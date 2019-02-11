#pragma once

#include "CbQueue.h"

namespace Hpipe {

/**
  Fake queue. Enable the computation of size needed for write stuff.
*/
class CfQueue {
public:
    CfQueue() : res( 0 ) {}

    // size
    bool empty() const { return res == 0; }
    ST   size () const { return res; }

    // writers
    void write_some( const void *data, ST size ) { res += size; }
    void write_some( const CbQueue &s ) { res += s.size(); }
    void write_byte( PI8 val ) { ++res; }
    void write_byte_wo_beg_test( PI8 val ) { ++res; }

    ST res;
};

} // namespace Hpipe
