#pragma once

#include "../utility/RefCount.h"
#include <ostream>
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
    virtual Type*                 type_factory_virtual( const std::string &name ); ///<
    virtual TypeFactory&          type_factory_virtual(); ///< return type_factory without the need for Task.cpp
    static Type*                  type_factory        ( const std::string &name );
    static TypeFactory&           type_factory        ();

    //
    std::vector<bool>             possible_variants;  ///<
    std::size_t                   chosen_variant;     ///<
    std::size_t                   machine_id;         ///<

    // output data
    Type*                         output_type;        ///<
    void*                         output_data;        ///<
    bool                          output_own;         ///<

    // graph data
    RefCount                      ref_count;          ///<
    std::vector<ComputableTask *> parents;            ///<
};

