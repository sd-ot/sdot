#pragma once

#include <dynalo/dynalo.hpp>
#include "support/TmpDir.h"
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
    using                      MSVS              = std::map<std::string,std::vector<std::string>>;
    using                      Func              = std::function<void(Task*,void **)>;
    using                      path              = std::filesystem::path;

    /**/                       KernelCode        ();

    void                       add_include_dir   ( path name );
    Func                       func              ( const Kernel &kernel, const std::vector<std::string> &input_types );

    path                       object_dir;

private:
    struct                     Code              { std::unique_ptr<dynalo::library> lib; Func func; };

    void                       init_default_flags();
    Code                       load_or_make_code ( const std::string &kstr, const Kernel &kernel, const std::vector<std::string> &input_types ); ///< make a Func from a Kernel (not cached in memory)
    void                       init_base_types   ();
    void                       make_gen_cmk      ( TmpDir &tmp_dir );
    void                       make_gen_cpp      ( TmpDir &tmp_dir, const path &output_path, const std::string &bname, const std::string &param );
    void                       make_code         ( const std::string &shash, const std::string &kstr, const Kernel &kernel, const std::vector<std::string> &input_types );
    Code                       load_code         ( const std::string &shash );
    void                       make_lib          ( TmpDir &tmp_dir, const std::string &shash, const Kernel &kernel, const std::vector<std::string> &input_types );
    void                       make_cmk          ( TmpDir &tmp_dir, const std::string &shash );
    void                       make_cpp          ( TmpDir &tmp_dir, const Kernel &kernel, const std::vector<std::string> &input_types );
    void                       gen_code          ( const path &output_path, const std::string &bname, const std::string &param );
    void                       build             ( const path &dir, const std::string &build_opt );

    void                       make_cmake_lists  ( const std::string &dir, const std::string &name, const std::vector<std::string> &flags );
    void                       make_kernel_cpp   ( const std::string &dir, const std::string &name, const std::vector<std::string> &input_types, bool task_as_arg, bool local_inc );
    void                       get_prereq_req    ( std::ostream &includes_os, std::ostream &src_heads_os, std::set<std::string> &includes_set, std::set<std::string> &src_heads_set, std::set<std::string> &seen_types, const std::string &type );
    void                       exec              ( const std::string &cmd );

    std::vector<std::string>   include_directories;
    std::string                cpu_config;
    MSVS                       src_heads;
    MSVS                       includes;
    std::map<std::string,Code> code;
};

extern KernelCode kernel_code;

} // namespace parex
