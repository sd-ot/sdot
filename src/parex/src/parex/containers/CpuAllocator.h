#ifndef parex_CpuMemory_H
#define parex_CpuMemory_H

#include <cstdlib>
#include <atomic>
#include <memory>
#include <array>

namespace parex {

/**

 */
struct CpuAllocator {
    template<class T> struct Alignment { static constexpr std::size_t value = 64; };

    static constexpr bool    cpu       = true;
    using                    I         = std::size_t;

    template<class T> void   deallocate( T *ptr, I count );
    template<class T> T*     allocate  ( I count );
    template<class T> T      value     ( const T *ptr );

    std::atomic<I>           used      = 0;
};

template<class T,int n,class F>
void get_memory_values( CpuAllocator &, CpuAllocator &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace parex

#include "CpuAllocator.tcc"

#endif // parex_CpuMemory_H
