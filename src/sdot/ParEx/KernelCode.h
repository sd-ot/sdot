#pragma once

#include <dynalo/dynalo.hpp>
#include <functional>
#include "Kernel.h"
#include <map>

namespace parex {

/**
*/
class KernelCode {
public:
    using                 Func            = std::function<void(void **)>;

    Func                  func            ( const Kernel &kernel );

private:
    struct                Src             { std::string name; std::vector<std::string> flags;  auto tie() const { return std::tie( name, flags ); } bool operator<( const Src &t ) const { return tie() < t.tie(); } };
    using                 Lib             = dynalo::library;

    Lib*                  lib             ( const std::string &name, std::vector<std::string> flags = {} ); ///<

    Func                  make_func       ( const Kernel &kernel ); ///< make a Func from a Kernel (not cached)
    Lib                   make_lib        ( const std::string &name, const std::vector<std::string> &flags ); ///< make a lib from a Kernel (not cached)

    void                  make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &flags );
    void                  build_kernel    ( const std::string &dir );
    void                  exec            ( const std::string &cmd );

    std::map<Kernel,Func> funcs;
    std::map<Src,Lib>     libs;
};

extern KernelCode kernel_code;

} // namespace parex
