#include "KernelWithCompiledCode.h"
#include "TODO.h"
#include "Task.h"

CompiledKernelCode::CompiledKernelCode( KernelWithCompiledCode *kernel ) : kernel( kernel ) {
}

CompiledKernelCode::Func *CompiledKernelCode::get_func( const Task *task ) {
    return symbol_for<Func>( kernel->kernel_parameters( task ) );
}

CompiledKernelCode::Path CompiledKernelCode::output_directory( const std::string &parameters ) const {
    return kernel->output_directory( parameters );
}

void CompiledKernelCode::make_srcs( SrcWriter &ff ) const {
    kernel->make_srcs( ff );
}
