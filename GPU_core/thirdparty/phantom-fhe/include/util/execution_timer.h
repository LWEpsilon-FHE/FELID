#pragma once
#include <array>
#include <chrono>
#include <cuda_runtime.h>
#include <stdexcept>

namespace phantom::execution {
enum class Phase { Compute, Preparation, CacheTransfer, Setup, Validation, Cleanup, Count };
class Timer;
inline thread_local Timer *active_timer=nullptr;
inline thread_local bool encoding_cache_enabled=false;
class Timer {
    using Clock=std::chrono::steady_clock;
    Clock::time_point start_=Clock::now(), boundary_=start_;
    std::array<double,static_cast<unsigned>(Phase::Count)> seconds_{};
    Phase phase_=Phase::Setup;
public:
    Timer() { if(active_timer)throw std::logic_error("Execution timer already active");active_timer=this; }
    Timer(const Timer &)=delete;
    Timer &operator=(const Timer &)=delete;
    ~Timer() { if(active_timer==this)active_timer=nullptr; }
    Phase phase()const{return phase_;}
    void change(Phase next) {
        if(next==phase_)return;
        const auto status=cudaDeviceSynchronize();
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        const auto now=Clock::now();
        seconds_[static_cast<unsigned>(phase_)]+=std::chrono::duration<double>(now-boundary_).count();
        boundary_=now;phase_=next;
    }
    double compute_seconds() {
        const auto status=cudaDeviceSynchronize();
        if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
        const auto now=Clock::now();
        seconds_[static_cast<unsigned>(phase_)]+=std::chrono::duration<double>(now-boundary_).count();
        boundary_=now;return seconds_[0];
    }
    double finish() {
        compute_seconds();active_timer=nullptr;
        return std::chrono::duration<double>(boundary_-start_).count();
    }
    double seconds(Phase phase)const{return seconds_[static_cast<unsigned>(phase)];}
};
class Scope {
    Timer *timer_=active_timer;
    Phase previous_=timer_ ? timer_->phase() : Phase::Compute;
    bool changed_=false;
public:
    explicit Scope(Phase next) {
        if(timer_ && static_cast<unsigned>(next)>static_cast<unsigned>(previous_)) {
            timer_->change(next);changed_=true;
        }
    }
    ~Scope() noexcept(false) {if(changed_)timer_->change(previous_);}
    Scope(const Scope &)=delete;
    Scope &operator=(const Scope &)=delete;
};
class EncodingCacheScope {
    bool previous_=encoding_cache_enabled;
public:
    EncodingCacheScope(){encoding_cache_enabled=true;}
    ~EncodingCacheScope(){encoding_cache_enabled=previous_;}
    EncodingCacheScope(const EncodingCacheScope &)=delete;
    EncodingCacheScope &operator=(const EncodingCacheScope &)=delete;
};
void clear_encoding_cache();
unsigned long long encoding_cache_hits();
unsigned long long encoding_cache_misses();
}
