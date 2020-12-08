#ifndef parex_CpuMemory_H
#define parex_CpuMemory_H

#include <asimd/processing_units/LargestCpu.h>
#include <asimd/SimdSize.h>
#include <array>

#include "../data/TypeInfo.h"
#include "CpuMemory.h"

namespace parex {

/**

 */
class BasicCpuAllocator {
public:
    template<class T> struct         Alignment        { static constexpr std::size_t value = asimd::SimdSize<T,asimd::processing_units::LargestCpu>::value; };
    static constexpr bool            cpu              = true;
    using                            I                = std::size_t;

    /**/                             BasicCpuAllocator();

    template<class T> void           deallocate       ( T *ptr, I count );
    template<class T> T*             allocate         ( I count );
    CpuMemory*                       memory           () { return &mem; }

    CpuMemory                        mem;
};

template<>
struct TypeInfo<BasicCpuAllocator> {
    static std::string name() {
        return "parex::BasicCpuAllocator";
    }
};

template<class T>
void copy_memory_values( const BasicCpuAllocator &dst_alloc, T *dst_data, const BasicCpuAllocator &src_alloc, const T *src_data, std::size_t len );

template<class T,int n,class F>
void get_memory_values( BasicCpuAllocator &, BasicCpuAllocator &, std::array<std::pair<const T *,std::size_t>,n> ptrs, F &&f );

} // namespace parex

#include "BasicCpuAllocator.tcc"

#endif // parex_CpuMemory_H
