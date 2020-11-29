#pragma once

#include "../Task.h"

/**
*/
class SrcTask : public Task {
public:
    /**/          SrcTask        ( Type *type, void *data, bool own );
    virtual      ~SrcTask        ();

    virtual void  write_to_stream( std::ostream &os ) const override;

    Type*         type;
    void*         data;
    bool          own;
};

