#pragma once

#include "../data/TypeInfo.h"
#include "Task.h"

namespace parex {

/**
*/
class SrcTask : public Task {
public:
    /**/            SrcTask        ( Type *type, void *data, bool own );

    template        <class T>
    static SrcTask* from_ptr       ( T *data, bool own = true );

    virtual void    write_to_stream( std::ostream &os ) const override;
};

template<class T>
SrcTask *SrcTask::from_ptr( T *data, bool own ) {
    return new SrcTask( Task::type_factory( TypeInfo<T>::name() ), data, own );
}

} // namespace parex
