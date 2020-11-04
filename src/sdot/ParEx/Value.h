#pragma once

#include "serialized.h"

namespace parex {
class Kernel;
class Task;

/**
*/
class Value {
public:
    /**/               Value          ( const Kernel &kernel, std::vector<Value> &&values );

    template           <class ...Args>
    /**/               Value          ( const Kernel &kernel, Args&& ...args ) : Value( kernel, _make_values( std::forward<Args>( args )... ) ) {  }

    template           <class T>
    /**/               Value          ( const T &value ) : serialized_value( serialized( value ) ), task( nullptr ) {}

    /**/               Value          ( const Value &that );
    /**/               Value          ( Value &&that );
    /**/               Value          ();

    /**/              ~Value          ();

    Value&             operator=      ( const Value &that );
    Value&             operator=      ( Value &&that );

    void               write_to_stream( std::ostream &os ) const;

    Task*              get_task       () const { return task; }

private:
    template           <class ...Args>
    std::vector<Value> _make_values   ( Args&& ...args ) { std::vector<Value> res; __make_values( res, args... ); return res; }

    template           <class Head,class ...Tail>
    void               __make_values  ( std::vector<Value> &res, Head &&head, Tail&& ...tail ) { res.push_back( std::forward<Head>( head ) ); __make_values( res, std::forward<Tail>( tail )... ); }
    void               __make_values  ( std::vector<Value> &/*res*/ ) {}

    static void        inc_ref        ( Task *task );
    static void        dec_ref        ( Task *task );

    std::string        serialized_value;
    Task*              task;
};


} // namespace parex
