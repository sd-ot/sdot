#include "TypeFactory.h"
#include "Task.h"

Task::Task() {
}

Task::~Task() {
}

void Task::get_front_rec( std::map<int,std::vector<ComputableTask *>> &/*front*/ ) {
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
