#ifndef asimd_CpuMemory_H
#define asimd_CpuMemory_H

#include <cstdlib>
#include <atomic>
#include <memory>
#include <array>

namespace asimd {

/**

 */
struct CpuMemory {
    template<class T> struct Alignment { static constexpr std::size_t value = 64; };
    static constexpr bool    cpu       = true;
    using                    TI        = std::size_t;

    template<class T> void   deallocate( T *ptr, TI count );
    template<class T> T*     allocate  ( TI count );
    template<class T> T      value     ( const T *ptr );

    std::atomic<TI>          used      = 0;
};

template<class T,int n,class F>
void get_memory_values( CpuMemory &, CpuMemory &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace asimd

#include "CpuMemory.tcc"

#endif // asimd_CpuMemory_H
