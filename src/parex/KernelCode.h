#pragma once

#include <dynalo/dynalo.hpp>
#include <functional>
#include "Kernel.h"
#include <memory>
#include <map>
#include <set>

namespace parex {
class Task;

/**
*/
class KernelCode {
public:
    using                    MSVS            = std::map<std::string,std::vector<std::string>>;
    using                    Func            = std::function<void(Task*,void **)>;

    /**/                     KernelCode      ();

    void                     add_include_dir ( std::string name );
    Func                     func            ( const Kernel &kernel, const std::vector<std::string> &input_types );

private:
    struct                   Src             { Kernel kernel; std::vector<std::string> input_types, flags; bool operator<( const Src &t ) const { return std::tie( kernel, input_types, flags ) < std::tie( t.kernel, t.input_types, t.flags ); } };
    struct                   Code            { std::unique_ptr<dynalo::library> lib; Func func; };

    Code                     make_code       ( const Kernel &kernel, const std::vector<std::string> &input_types ); ///< make a Func from a Kernel (not cached)

    void                     make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &flags );
    void                     make_kernel_cpp ( const std::string &dir, const std::string &name, const std::vector<std::string> &input_types );
    void                     get_prereq_req  ( std::ostream &includes_os, std::ostream &src_heads_os, std::set<std::string> &includes_set, std::set<std::string> &src_heads_set, std::set<std::string> &seen_types, const std::string &type );
    void                     build_kernel    ( const std::string &dir );
    void                     exec            ( const std::string &cmd );

    std::vector<std::string> include_directories;
    MSVS                     src_heads;
    MSVS                     includes;
    std::map<Src,Code>       code;
};

extern KernelCode kernel_code;

} // namespace parex
