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
    std::function<void(void)> operator()     ( const Kernel &kernel );

private:
    struct                    Func           { Func( std::string lib, std::string func ); dynalo::library lib; std::function<void(void)> func; };
    Func                      func           ( const Kernel &kernel );

    void                      make_CMakeLists( const Kernel &kernel, const std::string &dir );
    void                      build          ( const Kernel &kernel, const std::string &dir );
    void                      exec           ( const std::string &cmd );

    std::map<Kernel,Func>     compilations;
};

extern KernelCode kernel_code;

} // namespace parex
