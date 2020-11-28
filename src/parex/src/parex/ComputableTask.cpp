#include "ComputableTask.h"
#include "TODO.h"
#include "P.h"

ComputableTask::ComputableTask( std::vector<Rc<Task>> &&children ) : children( std::move( children ) ) {
    scheduled = 0;
    in_front  = 0;
    computed  = 0;
    priority  = 0;

    type = nullptr;
    data = nullptr;

    for( Rc<Task> &ch : this->children )
        ch->parents.push_back( this );
}

ComputableTask::~ComputableTask() {
    if ( type && data )
        type->destroy( data );
}

bool ComputableTask::all_ch_computed() const {
    for( const Rc<Task> &child : children )
        if ( child && ! child->is_computed() )
            return false;
    return true;
}

void ComputableTask::get_front_rec( std::map<int,std::vector<ComputableTask *>> &front ) {
    // in_front
    if ( in_front || computed )
        return;

    if ( all_ch_computed() ) {
        front[ - priority ].push_back( this );
        in_front = true;
        return;
    }

    // in_schedule
    if ( scheduled )
        return;
    scheduled = true;

    for( const Rc<Task> &child : children )
        child->get_front_rec( front );
}

bool ComputableTask::is_computed() const {
    return computed;
}

Type *ComputableTask::output_type() const {
    return type;
}

void *ComputableTask::output_data() const {
    return data;
}

