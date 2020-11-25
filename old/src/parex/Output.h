#pragma once

#include "OutputFuncList.h"
#include "type_name.h"
#include <utility>
#include <string>

namespace parex {

/**
*/
class Output {
public:
    using        FuncList       = OutputFuncList;

    /**/         Output         ( std::size_t *ref_count, FuncList *func_list, const std::string &type, const void *data ) : ref_count( ref_count ), func_list( func_list ), type( type ), data( const_cast<void *>( data ) ) {}
    /**/         Output         () : ref_count( nullptr ), func_list( nullptr ), data( nullptr ) {}

    /**/         Output         ( const Output &that ) : ref_count( that.ref_count ), func_list( that.func_list ), type( that.type ), data( that.data ) { inc_ref(); }
    /**/         Output         ( Output &&that ) : ref_count( std::exchange( that.ref_count, nullptr ) ), func_list( std::exchange( that.func_list, nullptr ) ), type( std::exchange( that.type, {} ) ), data( std::exchange( that.data, nullptr ) ) {}

    /**/        ~Output         () { dec_ref(); }

    Output&      operator=      ( const Output &that ) { that.inc_ref(); dec_ref(); ref_count = that.ref_count; func_list = that.func_list; type = that.type; data = that.data; return *this; }
    Output&      operator=      ( Output &&that ) { ref_count = std::exchange( that.ref_count, nullptr ); func_list = std::exchange( that.func_list, nullptr ); type = std::exchange( that.type, {} ); data = std::exchange( that.data, nullptr ); return *this; }

    void         write_to_stream( std::ostream &os ) const { if ( func_list ) func_list->write_to_stream( os, data ); else os << "null"; }
    void         inc_ref        () const { if ( ref_count ) ++ *ref_count; }
    void         dec_ref        () const { if ( ref_count && ! -- *ref_count ) func_list->destroy( ref_count, data ); }

    std::size_t* ref_count;
    FuncList*    func_list;
    std::string  type;
    void*        data;
};

} // namespace parex
