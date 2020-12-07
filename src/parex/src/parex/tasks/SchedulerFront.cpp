#include "SchedulerFront.h"

namespace parex {

void SchedulerFront::write_to_stream( std::ostream &os ) const {
    for( const auto &p : map )
        for( Task *t : p.second )
            os << t->name << " ";
}

void SchedulerFront::insert( parex::Task *task ) {
    map[ - task->priority ].push_back( task );
}

bool SchedulerFront::empty() const {
    return map.empty();
}

Task *SchedulerFront::pop() {
    if( empty() )
        return nullptr;

    std::map<double,std::vector<Task *>>::iterator first_in_front = map.begin();
    std::vector<Task *> &vf = first_in_front->second;
    Task *task = vf.back();
    vf.pop_back();

    if ( vf.empty() )
        map.erase( map.begin() );

    return task;
}

} // namespace parex
