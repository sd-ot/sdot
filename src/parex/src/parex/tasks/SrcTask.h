#pragma once

#include "Task.h"

namespace parex {

/**
*/
class SrcTask : public Task {
public:
    /**/          SrcTask        ( Type *type, void *data, bool own );

    virtual void  write_to_stream( std::ostream &os ) const override;
};

} // namespace parex
