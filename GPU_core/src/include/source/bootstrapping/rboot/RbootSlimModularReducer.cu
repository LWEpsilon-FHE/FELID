#include "RbootSlimModularReducer.cuh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <unistd.h>

#include "RbootSlimPolyEval.cuh"
#include "../RemezGeluAlgo6Even.cuh"
#include "../common/MinicompRemez.cuh"

using namespace NTL;
using namespace phantom;
using namespace moai;

namespace rboot_slim {
namespace {

std::vector<RR> load_rr_coefficients(const std::string &path, long expected_count);
std::vector<RR> load_all_rr_coefficients(const std::string &path);
void write_rr_coefficients(const std::string &path, const std::vector<RR> &coeffs);
void build_uniform_intervals(const RR &start, const RR &end, long interval_count, std::vector<RR> &starts, std::vector<RR> &ends);

double g_gelu_target_input_scale = 1.0;

std::filesystem::path executable_directory()
{
    std::vector<char> buffer(4096, '\0');
    const auto len = ::readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
    if (len <= 0)
    {
        return std::filesystem::current_path();
    }
    buffer[static_cast<size_t>(len)] = '\0';
    return std::filesystem::path(buffer.data()).parent_path();
}

long get_env_long_local(const char *name, long default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0')
    {
        throw std::runtime_error(std::string("Invalid integer value for ") + name + ": " + raw);
    }
    return value;
}

double get_env_double_local(const char *name, double default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return default_value;
    }
    char *end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0')
    {
        throw std::runtime_error(std::string("Invalid floating-point value for ") + name + ": " + raw);
    }
    return value;
}

bool get_env_bool_local(const char *name, bool default_value)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return default_value;
    }
    std::string lowered(raw);
    std::transform(lowered.begin(), lowered.end(), lowered.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on")
    {
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off")
    {
        return false;
    }
    throw std::runtime_error(std::string("Invalid boolean value for ") + name + ": " + raw);
}

std::string get_env_string_local(const char *name)
{
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0')
    {
        return {};
    }
    return raw;
}

RR gelu_even_remez_target(RR z)
{
    const RR alpha = sqrt(RR(2) / ComputePi_RR());
    const RR input_scale = RR(g_gelu_target_input_scale);
    const RR two = RR(2);
    const RR beta(NTL::INIT_VAL_TYPE{}, "0.044715");
    const RR x = input_scale * two * z;
    const RR cubic = x * x * x;
    const RR inner = alpha * (x + beta * cubic);
    const RR exp_pos = exp(inner);
    const RR exp_neg = exp(-inner);
    const RR tanh_inner = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    return input_scale * z * tanh_inner;
}

RR gelu_algo6_even_remez_target(RR y)
{
    const RR alpha = sqrt(RR(2) / ComputePi_RR());
    const RR beta(NTL::INIT_VAL_TYPE{}, "0.044715");
    const RR pi = ComputePi_RR();
    const RR quarter = RR(0.25);
    const RR half = RR(0.5);
    const RR x_abs = (arcsin(y) / (RR(2) * pi) + quarter) * half;
    const RR cubic = x_abs * x_abs * x_abs;
    const RR inner = alpha * (x_abs + beta * cubic);
    const RR exp_pos = exp(inner);
    const RR exp_neg = exp(-inner);
    const RR tanh_inner = (exp_pos - exp_neg) / (exp_pos + exp_neg);
    return half * x_abs * tanh_inner;
}

std::string resolve_arcsin_coeff_path(const std::string &configured_path)
{
    if (!configured_path.empty())
    {
        return configured_path;
    }

    {
        const auto candidate = executable_directory() / "arcsin_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "arcsin_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/arcsin_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    {
        const std::string candidate = "RBOOT_external/result/arcsin_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }

    return {};
}

std::string resolve_gelu_even_coeff_path()
{
    const std::string configured = get_env_string_local("RBOOT_GELU_EVEN_COEFF");
    if (!configured.empty())
    {
        const auto configured_path = std::filesystem::path(configured);
        if (configured_path.is_absolute())
        {
            return configured_path.string();
        }

        {
            const auto candidate = executable_directory() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

        {
            const auto candidate = std::filesystem::current_path() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

#ifdef FINETUNING_GPU_SOURCE_DIR
        {
            const auto candidate = configured_path.has_parent_path()
                                       ? (std::filesystem::current_path() / configured_path)
                                       : (std::filesystem::path(FINETUNING_GPU_SOURCE_DIR) / "src/data" / configured_path);
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
            return candidate.string();
        }
#else
        return (executable_directory() / configured_path).string();
#endif
    }

    {
        const auto candidate = executable_directory() / "gelu_even_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "gelu_even_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/gelu_even_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    {
        const std::string candidate = "RBOOT_external/result/gelu_even_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    return (executable_directory() / "gelu_even_d3.txt").string();
#else
    return (executable_directory() / "gelu_even_d3.txt").string();
#endif
}

std::string resolve_gelu_algo6_even_coeff_path()
{
    const std::string configured = get_env_string_local("RBOOT_GELU_ALGO6_EVEN_COEFF");
    if (!configured.empty())
    {
        const auto configured_path = std::filesystem::path(configured);
        if (configured_path.is_absolute())
        {
            return configured_path.string();
        }

        {
            const auto candidate = executable_directory() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

        {
            const auto candidate = std::filesystem::current_path() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

#ifdef FINETUNING_GPU_SOURCE_DIR
        {
            const auto candidate = configured_path.has_parent_path()
                                       ? (std::filesystem::current_path() / configured_path)
                                       : (std::filesystem::path(FINETUNING_GPU_SOURCE_DIR) / "src/data" / configured_path);
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
            return candidate.string();
        }
#else
        return (executable_directory() / configured_path).string();
#endif
    }

    {
        const auto candidate = executable_directory() / "gelu_even_algo6_baby_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "gelu_even_algo6_baby_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/gelu_even_algo6_baby_d127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    return (executable_directory() / "gelu_even_algo6_baby_d127.txt").string();
}

std::string resolve_tanh_coeff_path()
{
    const std::string configured = get_env_string_local("RBOOT_TANH_COEFF");
    if (!configured.empty())
    {
        const auto configured_path = std::filesystem::path(configured);
        if (configured_path.is_absolute())
        {
            return configured_path.string();
        }

        {
            const auto candidate = executable_directory() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

        {
            const auto candidate = std::filesystem::current_path() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

#ifdef FINETUNING_GPU_SOURCE_DIR
        {
            const auto candidate = configured_path.has_parent_path()
                                       ? (std::filesystem::current_path() / configured_path)
                                       : (std::filesystem::path(FINETUNING_GPU_SOURCE_DIR) / "src/data" / configured_path);
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
            return candidate.string();
        }
#else
        return (executable_directory() / configured_path).string();
#endif
    }

    {
        const auto candidate = executable_directory() / "tanh_deg128.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "tanh_deg128.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/tanh_deg128.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    {
        const auto candidate = executable_directory() / "tanh_odd_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "tanh_odd_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/tanh_odd_d3.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    return (executable_directory() / "tanh_odd_d3.txt").string();
}

std::string resolve_exp_coeff_path()
{
    const std::string configured = get_env_string_local("RBOOT_EXP_COEFF");
    if (!configured.empty())
    {
        const auto configured_path = std::filesystem::path(configured);
        if (configured_path.is_absolute())
        {
            return configured_path.string();
        }

        {
            const auto candidate = executable_directory() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

        {
            const auto candidate = std::filesystem::current_path() / configured_path;
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
        }

#ifdef FINETUNING_GPU_SOURCE_DIR
        {
            const auto candidate = configured_path.has_parent_path()
                                       ? (std::filesystem::current_path() / configured_path)
                                       : (std::filesystem::path(FINETUNING_GPU_SOURCE_DIR) / "src/data" / configured_path);
            std::ifstream ifs(candidate);
            if (ifs.good())
            {
                return candidate.string();
            }
            return candidate.string();
        }
#else
        return (executable_directory() / configured_path).string();
#endif
    }

    {
        const auto candidate = executable_directory() / "exp_deg127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

    {
        const auto candidate = std::filesystem::current_path() / "exp_deg127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate.string();
        }
    }

#ifdef FINETUNING_GPU_SOURCE_DIR
    {
        const std::string candidate = std::string(FINETUNING_GPU_SOURCE_DIR) + "/src/data/exp_deg127.txt";
        std::ifstream ifs(candidate);
        if (ifs.good())
        {
            return candidate;
        }
    }
#endif

    return (executable_directory() / "exp_deg127.txt").string();
}

std::vector<RR> generate_gelu_even_coefficients_via_remez(long gelu_even_deg, double zmax)
{
    constexpr long remez_iterations = 25;
    constexpr long remez_prec = 50;
    constexpr long rr_prec = 300;
    constexpr long scan_points = 1000;

    RR::SetPrecision(rr_prec);

    g_gelu_target_input_scale = get_env_double_local("RBOOT_GELU_TARGET_INPUT_SCALE", 1.0);
    if (!(g_gelu_target_input_scale > 0.0))
    {
        throw std::runtime_error("RBOOT_GELU_TARGET_INPUT_SCALE must be positive");
    }

    const RR z_min = RR(0);
    const RR z_max = RR(zmax);
    const RR scan_step = (z_max - z_min) / RR(scan_points);

    minicomp::Remez rm(gelu_even_remez_target,
                       1,
                       std::vector<RR>{z_min},
                       std::vector<RR>{z_max},
                       scan_step,
                       remez_prec,
                       gelu_even_deg,
                       remez_iterations,
                       0,
                       RR(1),
                       true);

    std::ostringstream sink;
    auto *old_buf = std::cout.rdbuf(sink.rdbuf());
    const bool ok = rm.test();
    std::cout.rdbuf(old_buf);
    if (!ok)
    {
        throw std::runtime_error("Failed to generate GELU-even minimax polynomial with Remez.");
    }

    std::vector<RR> coeffs;
    rm.getcoeff(coeffs);
    return coeffs;
}

std::vector<RR> generate_gelu_algo6_even_power_coefficients_via_remez(long degree, double epsilon)
{
    RemezParam params;
    params.log_scan_step_diff = get_env_double_local("RBOOT_GELU_ALGO6_LOG_SCAN_STEP_DIFF", 12.0);
    params.binary_prec = get_env_long_local("RBOOT_GELU_ALGO6_BINARY_PREC", 10);
    params.RR_prec = get_env_long_local("RBOOT_GELU_ALGO6_RR_PREC", 1000);
    params.log_approx_degree = get_env_long_local("RBOOT_GELU_ALGO6_LOG_APPROX_DEGREE", 120);
    params.log_round_prec = get_env_long_local("RBOOT_GELU_ALGO6_LOG_ROUND_PREC", 100);

    if (!(epsilon > 0.0 && epsilon < 1.0))
    {
        throw std::runtime_error("RBOOT_GELU_ALGO6_EVEN_EPS must be in (0, 1)");
    }

    const double log_width = -std::log2(1.0 - epsilon);
    RemezGeluAlgo6Even rm(params, log_width, degree);
    std::vector<RR> cheb_coeffs;
    rm.generate_optimal_cheb_coeff(cheb_coeffs);

    minicomp::Polynomial poly(static_cast<int>(degree), cheb_coeffs, "cheb");
    return poly.coeff;
}

std::vector<RR> generate_gelu_algo6_even_decomposition_via_remez(long degree,
                                                                 double epsilon,
                                                                 const std::string &path,
                                                                 minicomp::Tree &tree)
{
    auto coeffs = generate_gelu_algo6_even_power_coefficients_via_remez(degree, epsilon);
    upgrade_baby(degree, tree);

    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent);
    }

    std::ofstream out(path);
    if (!out.good())
    {
        throw std::runtime_error("Failed to write GeLU Algorithm 6 decomposition file: " + path);
    }
    std::ostringstream sink;
    auto *old_buf = std::cout.rdbuf(sink.rdbuf());
    minicomp::poly_decompose_integrate(static_cast<int>(degree),
                                       0,
                                       RR(1.0),
                                       coeffs,
                                       tree,
                                       evaltype::baby,
                                       1,
                                       RR(1.0),
                                       out);
    std::cout.rdbuf(old_buf);
    out.close();

    return load_rr_coefficients(path, coeff_number(degree, tree));
}

std::vector<RR> load_rr_coefficients(const std::string &path, long expected_count)
{
    std::ifstream coeff_stream(path);
    if (!coeff_stream.good())
    {
        throw std::runtime_error("Failed to open coefficient file: " + path);
    }

    std::vector<RR> coeffs;
    coeffs.reserve(static_cast<size_t>(expected_count));
    for (long i = 0; i < expected_count; ++i)
    {
        RR temp;
        coeff_stream >> temp;
        if (!coeff_stream.good())
        {
            throw std::runtime_error("Coefficient file has fewer entries than expected: " + path);
        }
        coeffs.emplace_back(temp);
    }
    return coeffs;
}

std::vector<RR> load_all_rr_coefficients(const std::string &path)
{
    std::ifstream coeff_stream(path);
    if (!coeff_stream.good())
    {
        throw std::runtime_error("Failed to open coefficient file: " + path);
    }

    std::vector<RR> coeffs;
    RR temp;
    while (coeff_stream >> temp)
    {
        coeffs.emplace_back(temp);
    }
    if (coeffs.empty())
    {
        throw std::runtime_error("Coefficient file is empty: " + path);
    }
    return coeffs;
}

void write_rr_coefficients(const std::string &path, const std::vector<RR> &coeffs)
{
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
    {
        std::filesystem::create_directories(parent);
    }
    std::ofstream ofs(path);
    if (!ofs.good())
    {
        throw std::runtime_error("Failed to write coefficient file: " + path);
    }
    for (const auto &coeff : coeffs)
    {
        ofs << coeff << '\n';
    }
}

std::vector<RR> generate_exp_taylor_coefficients(long degree)
{
    if (degree < 1)
    {
        throw std::runtime_error("RBOOT_EXP_DEG must be positive");
    }

    std::vector<RR> coeffs(static_cast<size_t>(degree + 1));
    RR factorial(1);
    for (long k = 0; k <= degree; ++k)
    {
        if (k > 0)
        {
            factorial *= RR(k);
        }
        coeffs[static_cast<size_t>(k)] = RR(1) / factorial;
    }
    return coeffs;
}

void build_uniform_intervals(const RR &start, const RR &end, long interval_count, std::vector<RR> &starts, std::vector<RR> &ends)
{
    if (interval_count <= 0)
    {
        throw std::runtime_error("RBOOT_GELU_ALGO6_INTERVALS must be positive");
    }
    starts.clear();
    ends.clear();
    starts.reserve(static_cast<size_t>(interval_count));
    ends.reserve(static_cast<size_t>(interval_count));
    const RR width = (end - start) / RR(interval_count);
    for (long i = 0; i < interval_count; ++i)
    {
        const RR seg_start = start + RR(i) * width;
        const RR seg_end = (i == interval_count - 1) ? end : (seg_start + width);
        starts.push_back(seg_start);
        ends.push_back(seg_end);
    }
}

}   

RbootSlimModularReducer::RbootSlimModularReducer(long boundary_K,
                                                 double log_width,
                                                 long deg,
                                                 long num_double_formula,
                                                 long inverse_deg,
                                                 CKKSEvaluator *ckks,
                                                 RbootSparseOps *ops,
                                                 std::string arcsin_coeff_path)
    : boundary_K(boundary_K),
      log_width(log_width),
      deg(deg),
      num_double_formula(num_double_formula),
      inverse_log_width(-std::log2(std::sin(2.0 * M_PI * std::pow(2.0, -log_width)))),
      inverse_deg(inverse_deg),
      ckks(ckks),
      ops(ops),
      arcsin_coeff_path(resolve_arcsin_coeff_path(arcsin_coeff_path)),
      gelu_even_coeff_path(resolve_gelu_even_coeff_path()),
      gelu_algo6_even_coeff_path(resolve_gelu_algo6_even_coeff_path()),
      tanh_coeff_path(resolve_tanh_coeff_path()),
      exp_coeff_path(resolve_exp_coeff_path()),
      dyn_scale_for_turn_back_q(std::pow(2.0, 46) / 69834700000000.0 / scale_for_boost_relu_range)
{
}

void RbootSlimModularReducer::double_angle_formula(PhantomCiphertext &cipher)
{
    ckks->evaluator.square_inplace(cipher);
    ckks->evaluator.relinearize_inplace(cipher, *(ckks->relin_keys));
    ckks->evaluator.rescale_to_next_inplace(cipher);
    ckks->evaluator.double_inplace(cipher);
    ops->add_const(cipher, -1.0, cipher);
}

void RbootSlimModularReducer::double_angle_formula_scaled(PhantomCiphertext &cipher, double scale_coeff)
{
    ckks->evaluator.square_inplace(cipher);
    ckks->evaluator.relinearize_inplace(cipher, *(ckks->relin_keys));
    ckks->evaluator.rescale_to_next_inplace(cipher);
    ckks->evaluator.double_inplace(cipher);
    ops->add_const(cipher, -scale_coeff, cipher);
}

void RbootSlimModularReducer::generate_sin_cos_polynomial()
{
    RR coeffs[] = {
        RR(0.92387953231773009),
        RR(15.027943245428581),
        RR(-712.36917917753132),
        RR(-3862.4962173546066),
        RR(91546.900696525565),
        RR(297822.73140750612),
        RR(-4705894.2117281199),
        RR(-10935240.05946226),
        RR(129590717.97240609),
        RR(234215477.02216466),
        RR(-2220501807.3704581),
        RR(-3283542596.6943443),
        RR(25941607670.155979),
        RR(32459195987.501604),
        RR(-219808768676.06232),
        RR(-238362547119.73398),
        RR(1412383968229.0714),
        RR(1351413420768.5189),
        RR(-7117851373948.3598),
        RR(-6093684508527.8351),
        RR(28885433488237.621),
        RR(22374069196578.916),
        RR(-96413939332866.742),
        RR(-68186629971363.139),
        RR(269324742471541.98),
        RR(175237895712024.71),
        RR(-638808512782011.86),
        RR(-384865939301985.97),
        RR(1302256999237959.8),
        RR(730509056263588.56),
        RR(-2304968261174938.7),
        RR(-1209713598732979.1),
        RR(3571666231875234.7),
        RR(1761338358467019.9),
        RR(-4875744024649225.6),
        RR(-2268100565747594.1),
        RR(5886770715832721.1),
        RR(2592558478326421.3),
        RR(-6292431181678847.7),
        RR(-2632767781192529),
        RR(5939686576589964.4),
        RR(2369063681350181),
        RR(-4918259710060236.7),
        RR(-1876272208761189.2),
        RR(3532216541468631),
        RR(1293070069731452.1),
        RR(-2164304632527603.8),
        RR(-762689882270018.15),
        RR(1106207490957092.5),
        RR(376362391898387.81),
        RR(-457397159398795.63),
        RR(-150657282523830.17),
        RR(146510702076755.46),
        RR(46835248558682.531),
        RR(-34007271107894.592),
        RR(-10574375024775.831),
        RR(5075533242659.2427),
        RR(1538211315599.5003),
        RR(-364844435510.66296),
        RR(-107962886396.68134)
    };

    sin_polynomial.set_polynomial(deg, coeffs, "power");
    sin_polynomial.generate_poly_heap();

    cos_polynomial.set_polynomial(deg, coeffs, "power");
    cos_polynomial.generate_poly_heap();
}

void RbootSlimModularReducer::generate_inverse_sine_polynomial()
{
    if (inverse_deg > 127)
    {
        throw std::runtime_error("We have only prepared taylor coeffs for terms of degree <= 127.");
    }
    if (inverse_deg < 3)
    {
        throw std::invalid_argument("We now only allow inverse deg >= 3.");
    }

    std::vector<RR> taylor_coeffs{
        RR(NTL::INIT_VAL_TYPE{}, "2.2947379471529382969744498600366971923620711188040720503135663285591120454439982e-76"),
        RR(NTL::INIT_VAL_TYPE{}, "0.15897595311853101274526061204647097374617576780504320373503002412804013125389352"),
        RR(NTL::INIT_VAL_TYPE{}, "-1.1608808021533176343645051916281395736534684849452555645808255157330666353747941e-73"),
        RR(NTL::INIT_VAL_TYPE{}, "0.056648697109133325293689187145117838558785315198698457540119087978676284245672466"),
        RR(NTL::INIT_VAL_TYPE{}, "9.7846038194851369194745179519276033267815972074371410866890581905249590678060137e-72"),
        RR(NTL::INIT_VAL_TYPE{}, "-1.4962576928053791590280746333430693060007070082898805468523500985795806225772547"),
        RR(NTL::INIT_VAL_TYPE{}, "-3.2702674279438227454484772562766054732640518481367510698173844152325652533644013e-70"),
        RR(NTL::INIT_VAL_TYPE{}, "35.362074308406350474792880925586208455112536866444251515111465396777026303611403"),
        RR(NTL::INIT_VAL_TYPE{}, "5.7551890541776576190524723561565756985691176458704944761227995371007212304187709e-69"),
        RR(NTL::INIT_VAL_TYPE{}, "-471.07953431217396424071402460253047716806523886881267880515901400181061914676474"),
        RR(NTL::INIT_VAL_TYPE{}, "-6.1385995412158862994464629918848808954017674485798557953343092811489608950401273e-68"),
        RR(NTL::INIT_VAL_TYPE{}, "3964.6161629320869022883621817561482183665609895483259339857446167047706330655413"),
        RR(NTL::INIT_VAL_TYPE{}, "4.3063391773465811423959945686875889964018085109242934483393710560231252306784904e-67"),
        RR(NTL::INIT_VAL_TYPE{}, "-22459.541248865152312938065178128097300367927572725676740568338390667504135425985"),
        RR(NTL::INIT_VAL_TYPE{}, "-2.0908474497159231683075942490197489538677345843971745704378214470396584316573739e-66"),
        RR(NTL::INIT_VAL_TYPE{}, "89123.731902018683809289470223194159766482580874743306244672533078890309896168463"),
        RR(NTL::INIT_VAL_TYPE{}, "7.2563980255277427934374583835867474857085908282004973666885313524025023335521144e-66"),
        RR(NTL::INIT_VAL_TYPE{}, "-253768.63233816677971544447178108247093939938513515465922939165725620651828358590"),
        RR(NTL::INIT_VAL_TYPE{}, "-1.8350287759899356541915493275239160064134455598891720786718897728659227461676995e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "524776.79801063595003095752312128769224864972771216682777456495127994560809612862"),
        RR(NTL::INIT_VAL_TYPE{}, "3.4109554196316155049189506446937865662178628136050185783972844603646948740133374e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "-789109.60198359304961497985472868477663171542552926472895053159878609101337416840"),
        RR(NTL::INIT_VAL_TYPE{}, "-4.6549606705774713570078140439899260027092906414990235601470982179195983272398013e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "853745.43290425213388163067627040607005174940281249565369620887280235781821082316"),
        RR(NTL::INIT_VAL_TYPE{}, "4.6067539112626614413484189888682410299940811380246496040148671655469170423923455e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "-647273.70324669429840150943198157914821646349529845866115260376069874133420227858"),
        RR(NTL::INIT_VAL_TYPE{}, "-3.2156205379114234927352893083936135992902282672923966492119385660829250635757931e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "326301.26986669056050070160365511101760051670477775281598819399244409811748101577"),
        RR(NTL::INIT_VAL_TYPE{}, "1.5006065588780182319983124977215458601564535200529088091947918519261862615559678e-65"),
        RR(NTL::INIT_VAL_TYPE{}, "-98218.984331747139107499019920039922801278956938360690938876111674031308704904622"),
        RR(NTL::INIT_VAL_TYPE{}, "-4.2005464365314365056914532097070591744796635056007673884285548727683343842567724e-66"),
        RR(NTL::INIT_VAL_TYPE{}, "13355.853558204680018492786448903280528249651256497222226676393541186558489225246")
    };

    inverse_sin_polynomial_v1.set_polynomial(static_cast<long>(taylor_coeffs.size()) - 1, taylor_coeffs.data(), "power");
    inverse_sin_polynomial_v1.generate_poly_heap();
    inverse_sin_polynomial_v1.constmul(RR(1.0 / scale_for_eval));
    inverse_sin_polynomial_v1.constmul(RR(0.5));
    inverse_sin_polynomial_v1.generate_poly_heap_odd();
    inverse_sin_polynomial_v1_original.copy(inverse_sin_polynomial_v1);

    upgrade_oddbaby(127, arcsin_tree);
    if (arcsin_coeff_path.empty())
    {
        throw std::runtime_error("Unable to locate arcsin_d127.txt.");
    }

    std::ifstream coeff_stream(arcsin_coeff_path);
    if (!coeff_stream.good())
    {
        throw std::runtime_error("Failed to open arcsin coefficient file: " + arcsin_coeff_path);
    }

    const long count = coeff_number(127, arcsin_tree);
    arcsin_decomp_coeff.clear();
    arcsin_decomp_coeff.reserve(static_cast<size_t>(count));
    for (long i = 0; i < count; ++i)
    {
        RR temp;
        coeff_stream >> temp;
        temp *= RR(0.25);
        temp *= RR(1.0 / scale_for_eval);
        arcsin_decomp_coeff.emplace_back(temp);
    }
    arcsin_decomp_coeff_original = arcsin_decomp_coeff;

    abs_lift = 0.125 * (1.0 / scale_for_eval) * 0.5;
}

void RbootSlimModularReducer::generate_gelu_even_polynomial()
{
    const long gelu_even_deg = get_env_long_local("RBOOT_GELU_EVEN_DEG", 3);
    const double gelu_even_zmax = get_env_double_local("RBOOT_GELU_EVEN_ZMAX", 0.125);
    gelu_target_input_scale = get_env_double_local("RBOOT_GELU_TARGET_INPUT_SCALE", 1.0);
    if (gelu_even_deg < 1)
    {
        throw std::runtime_error("RBOOT_GELU_EVEN_DEG must be positive");
    }
    if (!(gelu_even_zmax > 0.0))
    {
        throw std::runtime_error("RBOOT_GELU_EVEN_ZMAX must be positive");
    }
    if (!(gelu_target_input_scale > 0.0))
    {
        throw std::runtime_error("RBOOT_GELU_TARGET_INPUT_SCALE must be positive");
    }
    std::vector<RR> coeffs;
    std::ifstream coeff_stream(gelu_even_coeff_path);
    if (coeff_stream.good())
    {
        coeff_stream.close();
        coeffs = load_rr_coefficients(gelu_even_coeff_path, gelu_even_deg + 1);
    }
    else
    {
        coeffs = generate_gelu_even_coefficients_via_remez(gelu_even_deg, gelu_even_zmax);
        write_rr_coefficients(gelu_even_coeff_path, coeffs);
    }
    gelu_even_polynomial.set_polynomial(gelu_even_deg, coeffs.data(), "power");
    gelu_even_polynomial.generate_poly_heap();
}

void RbootSlimModularReducer::generate_tanh_polynomial()
{
    std::vector<RR> coeffs = load_all_rr_coefficients(tanh_coeff_path);
    long tanh_deg = static_cast<long>(coeffs.size()) - 1;
    const std::string configured_degree = get_env_string_local("RBOOT_TANH_DEG");
    if (!configured_degree.empty())
    {
        tanh_deg = get_env_long_local("RBOOT_TANH_DEG", tanh_deg);
        if (tanh_deg + 1 > static_cast<long>(coeffs.size()))
        {
            throw std::runtime_error("RBOOT_TANH_DEG requires more coefficients than available in " +
                                     tanh_coeff_path);
        }
        coeffs.resize(static_cast<size_t>(tanh_deg + 1));
    }

    if (tanh_deg < 1)
    {
        throw std::runtime_error("RBOOT_TANH_DEG must be positive");
    }

    const double tanh_input_scale = get_env_double_local("RBOOT_TANH_INPUT_SCALE", 0.69745);
    const double tanh_output_scale = get_env_double_local("RBOOT_TANH_OUTPUT_SCALE", 1.0);
    if (!(tanh_input_scale > 0.0))
    {
        throw std::runtime_error("RBOOT_TANH_INPUT_SCALE must be positive");
    }
    if (!(tanh_output_scale > 0.0))
    {
        throw std::runtime_error("RBOOT_TANH_OUTPUT_SCALE must be positive");
    }
    RR input_power(1.0);
    const RR input_scale_rr(tanh_input_scale);
    const RR output_scale_rr(tanh_output_scale);
    for (auto &coeff : coeffs)
    {
        coeff *= input_power;
        coeff *= output_scale_rr;
        input_power *= input_scale_rr;
    }

    tanh_polynomial.set_polynomial(tanh_deg, coeffs.data(), "power");
    tanh_polynomial.generate_poly_heap();
    
}

void RbootSlimModularReducer::generate_exp_polynomial()
{
    const bool explicit_coeff_path = !get_env_string_local("RBOOT_EXP_COEFF").empty();
    const bool use_coeff_file = get_env_bool_local("RBOOT_EXP_USE_COEFF_FILE", explicit_coeff_path);

    std::vector<RR> coeffs;
    long exp_deg = get_env_long_local("RBOOT_EXP_DEG", use_coeff_file ? -1 : 2);
    if (use_coeff_file)
    {
        coeffs = load_all_rr_coefficients(exp_coeff_path);
        if (exp_deg < 0)
        {
            exp_deg = static_cast<long>(coeffs.size()) - 1;
        }
    }
    else
    {
        if (exp_deg < 0)
        {
            exp_deg = 2;
        }
        coeffs = generate_exp_taylor_coefficients(exp_deg);
    }

    const std::string configured_degree = get_env_string_local("RBOOT_EXP_DEG");
    if (!configured_degree.empty())
    {
        exp_deg = get_env_long_local("RBOOT_EXP_DEG", exp_deg);
        if (exp_deg + 1 > static_cast<long>(coeffs.size()))
        {
            throw std::runtime_error("RBOOT_EXP_DEG requires more coefficients than available in " +
                                     exp_coeff_path);
        }
        coeffs.resize(static_cast<size_t>(exp_deg + 1));
    }

    if (exp_deg < 1)
    {
        throw std::runtime_error("RBOOT_EXP_DEG must be positive");
    }

    const double exp_input_scale = get_env_double_local("RBOOT_EXP_INPUT_SCALE", 0.708);
    const double exp_output_scale = get_env_double_local("RBOOT_EXP_OUTPUT_SCALE", 0.031255375);
    if (exp_input_scale == 0.0)
    {
        throw std::runtime_error("RBOOT_EXP_INPUT_SCALE must be non-zero");
    }
    if (exp_output_scale == 0.0)
    {
        throw std::runtime_error("RBOOT_EXP_OUTPUT_SCALE must be non-zero");
    }

    RR input_power(1.0);
    const RR input_scale_rr(exp_input_scale);
    const RR output_scale_rr(exp_output_scale);
    for (auto &coeff : coeffs)
    {
        coeff *= input_power;
        coeff *= output_scale_rr;
        input_power *= input_scale_rr;
    }

    exp_polynomial.set_polynomial(exp_deg, coeffs.data(), "power");
    exp_polynomial.generate_poly_heap();
    
}

void RbootSlimModularReducer::generate_gelu_algo6_even_decomposition()
{
    const long degree = get_env_long_local("RBOOT_GELU_ALGO6_EVEN_DEG", 127);
    const double epsilon = get_env_double_local("RBOOT_GELU_ALGO6_EVEN_EPS", 1.0e-4);
    if (degree < 1)
    {
        throw std::runtime_error("RBOOT_GELU_ALGO6_EVEN_DEG must be positive");
    }
    if (!(epsilon > 0.0 && epsilon < 1.0))
    {
        throw std::runtime_error("RBOOT_GELU_ALGO6_EVEN_EPS must be in (0, 1)");
    }

    upgrade_baby(degree, gelu_algo6_even_tree);

    std::ifstream coeff_stream(gelu_algo6_even_coeff_path);
    if (coeff_stream.good())
    {
        coeff_stream.close();
        gelu_algo6_even_decomp_coeff = load_rr_coefficients(gelu_algo6_even_coeff_path, coeff_number(degree, gelu_algo6_even_tree));
    }
    else
    {
        gelu_algo6_even_decomp_coeff =
            generate_gelu_algo6_even_decomposition_via_remez(degree, epsilon, gelu_algo6_even_coeff_path, gelu_algo6_even_tree);
    }
    gelu_algo6_even_decomp_coeff_original = gelu_algo6_even_decomp_coeff;
}

void RbootSlimModularReducer::scaling_for_turn_back_q()
{
    inverse_sin_polynomial_v1.copy(inverse_sin_polynomial_v1_original);
    inverse_sin_polynomial_v1.constmul(to_RR(dyn_scale_for_turn_back_q));
    inverse_sin_polynomial_v1.generate_poly_heap_odd();

    arcsin_decomp_coeff = arcsin_decomp_coeff_original;
    for (auto &coeff : arcsin_decomp_coeff)
    {
        coeff *= RR(dyn_scale_for_turn_back_q);
    }

    gelu_algo6_even_decomp_coeff = gelu_algo6_even_decomp_coeff_original;
    for (auto &coeff : gelu_algo6_even_decomp_coeff)
    {
        coeff *= RR(dyn_scale_for_turn_back_q);
    }

    abs_lift = 0.125 * (1.0 / scale_for_eval) * 0.5 * dyn_scale_for_turn_back_q;
}

void RbootSlimModularReducer::modular_reduction_split(PhantomCiphertext &odd_rtn,
                                                      PhantomCiphertext &even_rtn,
                                                      PhantomCiphertext &cipher)
{
    PhantomCiphertext sin_tmp2;
    PhantomCiphertext cos_tmp2;

    scaling_for_turn_back_q();

    homomorphic_poly_evaluation_sparse(sin_polynomial, *ops, cos_tmp2, cipher);
    for (int i = 0; i < num_double_formula + 1; ++i)
    {
        double_angle_formula(cos_tmp2);
        if (i == num_double_formula - 1)
        {
            sin_tmp2 = cos_tmp2;
        }
    }

    eval_polynomial_integrate(*ops, even_rtn, cos_tmp2, 127, arcsin_decomp_coeff, arcsin_tree);
    ops->add_const_inplace(even_rtn, abs_lift);

    homomorphic_poly_evaluation_sparse(inverse_sin_polynomial_v1, *ops, odd_rtn, sin_tmp2);
    ckks->evaluator.mod_switch_to_next_inplace(odd_rtn);

    const auto &context_data = ckks->context->get_context_data(odd_rtn.params_id());
    const double ql = static_cast<double>(context_data.parms().coeff_modulus().back().value());
    const double scale_adjust = even_rtn.scale() * ql / (odd_rtn.scale() * odd_rtn.scale());

    ops->multiply_const_inplace(odd_rtn, scale_adjust);
    ckks->evaluator.rescale_to_next_inplace(odd_rtn);
    odd_rtn.scale() = even_rtn.scale();
}

void RbootSlimModularReducer::modular_reduction(PhantomCiphertext &rtn, PhantomCiphertext &cipher)
{
    PhantomCiphertext odd_rtn;
    PhantomCiphertext even_rtn;
    modular_reduction_split(odd_rtn, even_rtn, cipher);
    ops->add_reduced_error(odd_rtn, even_rtn, rtn);
}

}
