#pragma once

#include "RefCount.h"
#include <vector>
#include <map>

template<class T> class TaskOut;
class ComputableTask;
class TypeFactory;
class Type;

/**
*/
class Task {
public:
    /**/                          Task                ();
    virtual                      ~Task                ();

    virtual void                  write_to_stream     ( std::ostream &os ) const = 0;
    virtual void                  get_front_rec       ( std::map<int,std::vector<ComputableTask *>> &front );
    virtual bool                  is_computed         () const;

    // how to get the global type_factory, even in dynamic libraries
    virtual TypeFactory&          type_factory_virtual(); ///< return type_factory without the need for Task.cpp
    static Type*                  type_factory        ( const std::string &name );
    static TypeFactory&           type_factory        ();

    // output data
    bool                          output_is_owned;    ///<
    Type*                         output_type;        ///<
    void*                         output_data;        ///<

    // graph data
    RefCount                      ref_count;          ///<
    std::vector<ComputableTask *> parents;            ///<
};

