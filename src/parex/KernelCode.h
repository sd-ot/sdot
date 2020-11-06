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
    using                    Func            = std::function<void(Task*,void **)>;
    using                    MSVS            = std::map<std::string,std::vector<std::string>>;

    /**/                     KernelCode      ();

    Func                     func            ( const Kernel &kernel, const std::vector<std::string> &input_types );

private:
    struct                   Src             { Kernel kernel; std::vector<std::string> input_types, flags; auto tie() const { return std::tie( kernel, input_types, flags ); } bool operator<( const Src &t ) const { return tie() < t.tie(); } };
    struct                   Code            { std::unique_ptr<dynalo::library> lib; Func func; };

    Code                     make_code       ( const Kernel &kernel, const std::vector<std::string> &input_types ); ///< make a Func from a Kernel (not cached)

    void                     make_cmake_lists( const std::string &dir, const std::string &name, const std::vector<std::string> &flags );
    void                     make_kernel_cpp ( const std::string &dir, const std::string &name, const std::vector<std::string> &input_types );
    void                     get_prereq_req  ( std::ostream &includes, std::ostream &src_heads, std::set<std::string> &include_set, std::set<std::string> &src_head_set, std::set<std::string> &seen_types, const std::string &type );
    void                     build_kernel    ( const std::string &dir );
    void                     exec            ( const std::string &cmd );

    std::vector<std::string> include_directories;
    MSVS                     src_heads;
    MSVS                     includes;
    std::map<Src,Code>       code;
};

extern KernelCode kernel_code;

} // namespace parex