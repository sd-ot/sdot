#pragma once

#include "RefCount.h"
#include "Type.h"

#include <vector>
#include <map>

class ComputableTask;
class TypeFactory;

/**
*/
class Task {
public:
    /**/                          Task                ();
    virtual                      ~Task                ();

    virtual void                  write_to_stream     ( std::ostream &os ) const = 0;
    virtual void                  get_front_rec       ( std::map<int,std::vector<ComputableTask *>> &front );
    virtual bool                  is_computed         () const = 0;
    virtual Type*                 output_type         () const = 0;
    virtual void*                 output_data         () const = 0;

    virtual TypeFactory&          type_factory_virtual(); ///< return type_factory without the need for Task.cpp
    static Type*                  type_factory        ( const std::string &name );
    static TypeFactory&           type_factory        ();

    RefCount                      ref_count;          ///<
    std::vector<ComputableTask *> parents;            ///<
};
