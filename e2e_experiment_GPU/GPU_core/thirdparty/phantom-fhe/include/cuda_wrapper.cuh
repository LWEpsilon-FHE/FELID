#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

#define PHANTOM_CHECK_CUDA(val) check((val), #val, __FILE__, __LINE__)
#define PHANTOM_CHECK_CUDA_LAST() checkLast(__FILE__, __LINE__)

template <typename T>
inline void check(T err, const char *const func, const char *const file,
                  const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << std::endl
                  << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func
                  << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

inline void checkLast(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

namespace phantom::util
{

#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
    struct ciphertext_allocation_snapshot
    {
        size_t live_count = 0;
        size_t live_bytes = 0;
        size_t peak_count = 0;
        size_t bytes_at_peak_count = 0;
        size_t peak_bytes = 0;
        size_t count_at_peak_bytes = 0;
        size_t phase_peak_count = 0;
        size_t phase_bytes_at_peak_count = 0;
        size_t phase_peak_bytes = 0;
        size_t phase_count_at_peak_bytes = 0;
    };

    inline std::atomic<size_t> ciphertext_live_count{0};
    inline std::atomic<size_t> ciphertext_live_bytes{0};
    inline std::atomic<size_t> ciphertext_peak_count{0};
    inline std::atomic<size_t> ciphertext_bytes_at_peak_count{0};
    inline std::atomic<size_t> ciphertext_peak_bytes{0};
    inline std::atomic<size_t> ciphertext_count_at_peak_bytes{0};
    inline std::atomic<size_t> ciphertext_phase_peak_count{0};
    inline std::atomic<size_t> ciphertext_phase_bytes_at_peak_count{0};
    inline std::atomic<size_t> ciphertext_phase_peak_bytes{0};
    inline std::atomic<size_t> ciphertext_phase_count_at_peak_bytes{0};

    inline void update_ciphertext_peak_pair(size_t live_count,
                                            size_t live_bytes,
                                            std::atomic<size_t> &peak_count,
                                            std::atomic<size_t> &bytes_at_peak_count,
                                            std::atomic<size_t> &peak_bytes,
                                            std::atomic<size_t> &count_at_peak_bytes)
    {
        size_t observed_count = peak_count.load();
        while (live_count > observed_count &&
               !peak_count.compare_exchange_weak(observed_count, live_count))
        {
        }
        if (live_count >= peak_count.load())
        {
            bytes_at_peak_count.store(live_bytes);
        }

        size_t observed_bytes = peak_bytes.load();
        while (live_bytes > observed_bytes &&
               !peak_bytes.compare_exchange_weak(observed_bytes, live_bytes))
        {
        }
        if (live_bytes >= peak_bytes.load())
        {
            count_at_peak_bytes.store(live_count);
        }
    }

    inline void track_ciphertext_allocation(size_t bytes)
    {
        const size_t live_count = ciphertext_live_count.fetch_add(1) + 1;
        const size_t live_bytes = ciphertext_live_bytes.fetch_add(bytes) + bytes;

        update_ciphertext_peak_pair(live_count, live_bytes,
                                    ciphertext_peak_count,
                                    ciphertext_bytes_at_peak_count,
                                    ciphertext_peak_bytes,
                                    ciphertext_count_at_peak_bytes);
        update_ciphertext_peak_pair(live_count, live_bytes,
                                    ciphertext_phase_peak_count,
                                    ciphertext_phase_bytes_at_peak_count,
                                    ciphertext_phase_peak_bytes,
                                    ciphertext_phase_count_at_peak_bytes);
    }

    inline void track_ciphertext_release(size_t bytes)
    {
        ciphertext_live_count.fetch_sub(1);
        ciphertext_live_bytes.fetch_sub(bytes);
    }

    inline void reset_ciphertext_allocation_peaks()
    {
        ciphertext_peak_count.store(ciphertext_live_count.load());
        ciphertext_bytes_at_peak_count.store(ciphertext_live_bytes.load());
        ciphertext_peak_bytes.store(ciphertext_live_bytes.load());
        ciphertext_count_at_peak_bytes.store(ciphertext_live_count.load());
    }

    inline void reset_ciphertext_allocation_phase_peaks()
    {
        ciphertext_phase_peak_count.store(ciphertext_live_count.load());
        ciphertext_phase_bytes_at_peak_count.store(ciphertext_live_bytes.load());
        ciphertext_phase_peak_bytes.store(ciphertext_live_bytes.load());
        ciphertext_phase_count_at_peak_bytes.store(ciphertext_live_count.load());
    }

    inline ciphertext_allocation_snapshot get_ciphertext_allocation_snapshot()
    {
        return {
            ciphertext_live_count.load(),
            ciphertext_live_bytes.load(),
            ciphertext_peak_count.load(),
            ciphertext_bytes_at_peak_count.load(),
            ciphertext_peak_bytes.load(),
            ciphertext_count_at_peak_bytes.load(),
            ciphertext_phase_peak_count.load(),
            ciphertext_phase_bytes_at_peak_count.load(),
            ciphertext_phase_peak_bytes.load(),
            ciphertext_phase_count_at_peak_bytes.load(),
        };
    }
#endif

    class cuda_stream_wrapper
    {
    public:
        cuda_stream_wrapper()
        {
            cudaSetDevice(0);  
            cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        }

        ~cuda_stream_wrapper()
        {
            cudaStreamDestroy(stream);
        }

        [[nodiscard]] auto &get_stream() const
        {
            return stream;
        }

    private:
        cudaStream_t stream{};
    };

    template <class T>
    class cuda_auto_ptr
    {

    private:
        T *ptr_ = nullptr;
        size_t n_ = 0;
        cudaStream_t cudaStream_ = nullptr;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
        bool ciphertext_role_ = false;
        bool ciphertext_allocation_counted_ = false;
#endif

    public:
        cuda_auto_ptr() = default;

        explicit cuda_auto_ptr(T *ptr, size_t n, const cudaStream_t &stream)
        {
            ptr_ = ptr;
            n_ = n;
            cudaStream_ = stream;
        }

        cuda_auto_ptr(const cuda_auto_ptr &obj)
        {
            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            this->ciphertext_role_ = obj.ciphertext_role_;
            if (this->ciphertext_role_ && this->ptr_ != nullptr)
            {
                track_ciphertext_allocation(this->n_ * sizeof(T));
                this->ciphertext_allocation_counted_ = true;
            }
#endif
        }

        cuda_auto_ptr &operator=(const cuda_auto_ptr &obj)
        {
            if (this == &obj)
            {
                return *this;
            }

#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            const bool ciphertext_role = this->ciphertext_role_ || obj.ciphertext_role_;
#endif
            reset();

            PHANTOM_CHECK_CUDA(cudaMallocAsync(&this->ptr_, obj.n_ * sizeof(T), obj.cudaStream_));
            PHANTOM_CHECK_CUDA(cudaMemcpyAsync(this->ptr_, obj.ptr_, obj.n_ * sizeof(T), cudaMemcpyDeviceToDevice,
                                               obj.cudaStream_));
            this->n_ = obj.n_;
            this->cudaStream_ = obj.cudaStream_;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            this->ciphertext_role_ = ciphertext_role;
            if (this->ciphertext_role_ && this->ptr_ != nullptr)
            {
                track_ciphertext_allocation(this->n_ * sizeof(T));
                this->ciphertext_allocation_counted_ = true;
            }
#endif
            return *this;
        }

        cuda_auto_ptr(cuda_auto_ptr &&dyingObj) noexcept
        {
             
            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            this->ciphertext_role_ = dyingObj.ciphertext_role_;
            this->ciphertext_allocation_counted_ = dyingObj.ciphertext_allocation_counted_;
#endif

            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            dyingObj.ciphertext_allocation_counted_ = false;
#endif
        }

        cuda_auto_ptr &operator=(cuda_auto_ptr &&dyingObj) noexcept
        {
            if (this == &dyingObj)
            {
                return *this;
            }

#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            const bool ciphertext_role = this->ciphertext_role_ || dyingObj.ciphertext_role_;
#endif
            reset();

            this->ptr_ = dyingObj.ptr_;
            this->n_ = dyingObj.n_;
            this->cudaStream_ = dyingObj.cudaStream_;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            this->ciphertext_role_ = ciphertext_role;
            this->ciphertext_allocation_counted_ = dyingObj.ciphertext_allocation_counted_;
            if (this->ciphertext_role_ && this->ptr_ != nullptr &&
                !this->ciphertext_allocation_counted_)
            {
                track_ciphertext_allocation(this->n_ * sizeof(T));
                this->ciphertext_allocation_counted_ = true;
            }
#endif

            dyingObj.ptr_ = nullptr;
            dyingObj.n_ = 0;
            dyingObj.cudaStream_ = nullptr;
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            dyingObj.ciphertext_allocation_counted_ = false;
#endif

            return *this;
        }

        ~cuda_auto_ptr()  
        {
            reset();
        }

        T *get() const
        {
            return this->ptr_;
        }

        T *operator->() const
        {
            return this->ptr_;
        }

        T &operator*() const
        {
            return this->ptr_;
        }

        [[nodiscard]] auto &get_n() const
        {
            return this->n_;
        }

        [[nodiscard]] auto &get_stream() const
        {
            return this->cudaStream_;
        }

        void reset()
        {
            if (ptr_ == nullptr)
            {
                return;
            }
            auto err = cudaFreeAsync(ptr_, cudaStream_);
#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
            if (ciphertext_allocation_counted_)
            {
                track_ciphertext_release(n_ * sizeof(T));
                ciphertext_allocation_counted_ = false;
            }
#endif
            if (err != cudaSuccess)
            {
                std::cerr << "Error freeing " << n_ << " * " << sizeof(T) << " bytes at " << ptr_
                          << " on stream " << cudaStream_ << std::endl;
                std::cerr << "Error code: " << cudaGetErrorString(err) << std::endl;
            }
            n_ = 0;
            cudaStream_ = nullptr;
            ptr_ = nullptr;
        }

#ifdef PHANTOM_TRACK_CIPHERTEXT_ALLOCATIONS
        void mark_ciphertext_role()
        {
            ciphertext_role_ = true;
            if (ptr_ != nullptr && !ciphertext_allocation_counted_)
            {
                track_ciphertext_allocation(n_ * sizeof(T));
                ciphertext_allocation_counted_ = true;
            }
        }
#endif
    };

    template <class T>
    cuda_auto_ptr<T> make_cuda_auto_ptr(size_t n, const cudaStream_t &stream)
    {
        T *ptr;
        PHANTOM_CHECK_CUDA(cudaMallocAsync(&ptr, n * sizeof(T), stream));
        return cuda_auto_ptr<T>(ptr, n, stream);
    }

    class CUDATimer
    {
    public:
        explicit CUDATimer(std::string func_name, const phantom::util::cuda_stream_wrapper &stream_wrapper) : func_name_(std::move(func_name))
        {
            stream_ = stream_wrapper.get_stream();
            cudaEventCreate(&start_event_);
            cudaEventCreate(&stop_event_);
        }

        ~CUDATimer()
        {
            cudaEventDestroy(start_event_);
            cudaEventDestroy(stop_event_);
            auto n_trials = time_.size();
            auto mean_time = mean(time_);
            auto min_time = min(time_);
            auto median_time = median(time_);
            auto stddev = std_dev(time_);
            ;
        }

        inline void start() const
        {
            cudaEventRecord(start_event_, stream_);
        }

        inline void stop()
        {
            cudaEventRecord(stop_event_, stream_);
            cudaEventSynchronize(stop_event_);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start_event_, stop_event_);
            time_.push_back(milliseconds);
        }

    private:
        std::string func_name_;
        cudaStream_t stream_;
        cudaEvent_t start_event_{}, stop_event_{};
        std::vector<float> time_;

        static float mean(std::vector<float> const &v)
        {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            return std::reduce(v.begin(), v.end()) / count * 1000;
        }

        static float median(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v[size / 2] * 1000;
        }

        static float min(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.front() * 1000;
        }

        static float max(std::vector<float> v)
        {
            size_t size = v.size();

            if (size == 0)
                return 0;

            sort(v.begin(), v.end());
            return v.back() * 1000;
        }

        static double std_dev(std::vector<float> const &v)
        {
            if (v.empty())
                return 0;

            auto const count = static_cast<float>(v.size());
            float mean = std::reduce(v.begin(), v.end()) / count;

            std::vector<double> diff(v.size());

            std::transform(v.begin(), v.end(), diff.begin(), [mean](double x)
                           { return x - mean; });
            double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
            return std::sqrt(sq_sum / count);
        }
    };

}
