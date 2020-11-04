#pragma once

#include <dynalo/dynalo.hpp>
#include <functional>
#include "Kernel.h"
#include <memory>
#include <map>

namespace parex {

/**
*/
class KernelCode {
public:
    using                 Func            = std::function<void(void **)>;

    Func                  func            ( const Kernel &kernel, const std::vector<std::string> &input_types );

private:
    struct                Src             { Kernel kernel; std::vector<std::string> input_types, flags; auto tie() const { return std::tie( kernel, input_types, flags ); } bool operator<( const Src &t ) const { return tie() < t.tie(); } };
    struct                Code            { std::unique_ptr<dynalo::library> lib; Func func; };

    Code                  make_code       ( const Kernel &kernel, const std::vector<std::string> &input_types ); ///< make a Func from a Kernel (not cached)

    void                  make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &flags );
    void                  build_kernel    ( const std::string &dir );
    void                  make_cpp        ( const std::string &dir, const Kernel &kernel );
    void                  exec            ( const std::string &cmd );

    std::map<Src,Code>    code;
};

extern KernelCode kernel_code;

} // namespace parex
