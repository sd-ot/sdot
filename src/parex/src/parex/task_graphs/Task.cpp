#include "TypeFactory.h"
#include "Task.h"

Task::Task() {
    output_type = nullptr;
    output_data = nullptr;
    output_own = true;
}

Task::~Task() {
    if ( output_own && output_type && output_data )
        output_type->destroy( output_data );
}

void Task::get_front_rec( std::map<int,std::vector<ComputableTask *>> &/*front*/ ) {
}

bool Task::is_computed() const {
    return true;
}

Type *Task::type_factory_virtual( const std::string &name ) {
    TypeFactory &tf = type_factory_virtual();
    return tf( name );
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
