#include "TypeFactory.h"
#include "Task.h"

Task::Task() {
    output_is_owned = true;
    output_type = nullptr;
    output_data = nullptr;
}

Task::~Task() {
    if ( output_is_owned && output_type && output_data )
        output_type->destroy( output_data );
}

void Task::get_front_rec( std::map<int,std::vector<ComputableTask *>> &/*front*/ ) {
}

bool Task::is_computed() const {
    return true;
}

TypeFactory &Task::type_factory_virtual() {
    return type_factory();
}

Type *Task::type_factory( const std::string &name ) {
    TypeFactory &tf = type_factory();
    return tf( name );
}

TypeFactory &Task::type_factory() {
    static TypeFactory res;
    return res;
}
