#ifndef PAREX_Task_HEADER
#define PAREX_Task_HEADER

#include "../utility/RefCount.h"
#include "../utility/Rc.h"
#include "../data/Data.h"
#include <ostream>
#include <vector>
#include <set>

namespace parex {

template<class T> class TaskOut;
class SchedulerFront;
class SchedulerFront;
class TypeFactory;
class Type;

/**
*/
class Task {
public:
    /**/                           Task                   ( std::string &&name, std::vector<Rc<Task>> &&children = {}, double priority = 0 );
    virtual                       ~Task                   ();

    // static ctors
    template<class T> static Task* new_src_from_ptr       ( T *data, bool own = true );
    static Task*                   new_src                ( Type *type, void *data, bool own = true );

    //
    virtual void                   check_input_same_memory();
    virtual void                   remove_from_parents    ( const Task *parent_to_remove );
    virtual bool                   all_ch_computed        () const;
    virtual void                   write_to_stream        ( std::ostream &os ) const;
    virtual void                   get_front_rec          ( SchedulerFront &front );
    virtual void                   insert_child           ( std::size_t num_child, const Rc<Task> &new_child );
    virtual Rc<Task>               move_child             ( std::size_t num_child ); ///<
    virtual void                   prepare                (); ///< done before execution, each time there's something new in one child. Can be used to check the input types. By default: check that data are allocated in the same space.
    virtual void                   exec                   ();

    void                           for_each_rec           ( const std::function<void( Task *task )> &f, std::set<Task *> &seen, bool go_to_parents = false );
    static void                    display_dot            ( const std::vector<Rc<Task>> &tasks, std::string f = ".tasks.dot", const char *prg = nullptr );

    // type factory
    virtual Type*                  type_factory_virtual   ( const std::string &name ); ///<
    virtual TypeFactory&           type_factory_virtual   (); ///< return type_factory without the need for symbols in Task.cpp
    static Type*                   type_factory           ( const std::string &name );
    static TypeFactory&            type_factory           ();

    // output creation
    template<class F> void         run_kernel_wrapper     ( const F &f ); ///< helper to create output from a class with a operator()( Task * ) method
    template<class F> void         run_void_or_not        ( std::integral_constant<bool,0>, const F &func );
    template<class F> void         run_void_or_not        ( std::integral_constant<bool,1>, const F &func );
    template<class T> void         make_outputs           ( TaskOut<T> &&ret );

    // variants
    std::vector<bool>              possible_variants;     ///<
    std::size_t                    chosen_variant;        ///<
    std::size_t                    machine_id;            ///<

    // graph data
    RefCount                       ref_count;             ///<
    std::vector<Rc<Task>>          children;              ///<
    double                         priority;              ///<
    std::vector<Task *>            parents;               ///<
    Data                           output;                ///<
    std::string                    name;                  ///<

    bool                           scheduled;             ///<
    bool                           in_front;              ///<
    bool                           computed;              ///<
};

} // namespace parex

#include "Task.tcc"

#endif // PAREX_Task_HEADER
