#pragma once

#include <iostream>
#include <chrono>

namespace sdot {

/** */
class Time {
public:
    using    TP = std::chrono::high_resolution_clock::time_point;

    double   operator-( const Time &that ) const { return std::chrono::duration_cast<std::chrono::microseconds>( time_point - that.time_point ).count() / 1e6; }

    TP       time_point;
};

inline Time time() {
    return { std::chrono::high_resolution_clock::now() };
}

/** */
class RaiiTime {
public:
    /**/        RaiiTime( const char *str ) : str( str ), t0( time() ) {}
    /**/       ~RaiiTime() { double dt = time() - t0; std::cout << "  " << str << " => " << dt << std::endl; }
    const char *str;
    Time        t0;
};

#define RDTSC_START(cycles)                                                   \
    do {                                                                      \
        unsigned cyc_high, cyc_low;                                           \
        __asm volatile(                                                       \
            "cpuid\n\t"                                                       \
            "rdtsc\n\t"                                                       \
            "mov %%edx, %0\n\t"                                               \
            "mov %%eax, %1\n\t"                                               \
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx"); \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                      \
        __asm volatile("" ::: /* pretend to clobber */ "memory");             \
    } while (0)

#define RDTSC_FINAL(cycles)                                                   \
    do {                                                                      \
        __asm volatile("" ::: /* pretend to clobber */ "memory");             \
        unsigned cyc_high, cyc_low;                                           \
        __asm volatile(                                                       \
            "rdtscp\n\t"                                                      \
            "mov %%edx, %0\n\t"                                               \
            "mov %%eax, %1\n\t"                                               \
            "cpuid\n\t"                                                       \
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx"); \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                      \
    } while (0)

#define BEST_TIME(test, pre, repeat, size)                                     \
    do {                                                                       \
        printf("%-60s: ", #test);                                              \
        fflush(NULL);                                                          \
        uint64_t cycles_start, cycles_final, cycles_diff;                      \
        uint64_t min_diff = (uint64_t)-1;                                      \
        for (int i = 0; i < repeat; i++) {                                     \
            pre;                                                               \
            __asm volatile("" ::: /* pretend to clobber */ "memory");          \
            RDTSC_START(cycles_start);                                         \
            test;                                                              \
            RDTSC_FINAL(cycles_final);                                         \
            cycles_diff = (cycles_final - cycles_start);                       \
            if (cycles_diff < min_diff)                                        \
              min_diff = cycles_diff;                                          \
        }                                                                      \
        uint64_t S = size;                                                     \
        float ns_per_op = (min_diff) / (double)S;                              \
        printf(" %.2f cycles per input key ", ns_per_op);                      \
        printf("\n");                                                          \
        fflush(NULL);                                                          \
    } while (0)

}

