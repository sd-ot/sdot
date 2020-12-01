#pragma once

#define XTENSOR_USE_XSIMD 1

#include <asimd/AllocatorTrait.h>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp>
#include "../type_name.h"

inline std::string xt_layout_type_name( xt::layout_type lt ) {
    switch ( lt ) {
    case xt::layout_type::dynamic     : return "xt::layout_type::dynamic";
    case xt::layout_type::any         : return "xt::layout_type::any";
    case xt::layout_type::row_major   : return "xt::layout_type::row_major";
    case xt::layout_type::column_major: return "xt::layout_type::column_major";
    }
    return "unknow layout type";
}

template<class T,std::size_t alig>
std::string type_name( S<xsimd::aligned_allocator<T,alig>> ) {
    return "xsimd::aligned_allocator<" + type_name( S<T>() ) + "," + std::to_string( alig ) + ">";
}

template<class T,xt::layout_type lt,class A>
std::string type_name( S<xt::xarray<T,lt,A>> ) {
    return "xt::xarray<" + type_name( S<T>() ) + "," + xt_layout_type_name( lt ) + "," + type_name( S<A>() ) + ">";
}

template<class T,std::size_t N,xt::layout_type lt,class A>
std::string type_name( S<xt::xtensor<T,N,lt,A>> ) {
    return "xt::xtensor<" + type_name( S<T>() ) + "," + std::to_string( N ) + "," + xt_layout_type_name( lt ) + "," + type_name( S<A>() ) + ">";
}

// AllocatorTrait
namespace asimd {
template<class T,std::size_t alig>
    struct AllocatorTrait<xsimd::aligned_allocator<T,alig>> {
        enum {                          alignment = alig };
        static position::Cpu<alignment> position  ( const std::allocator<T> & = {} ) { return {}; }
    };
} // namespace asimd
