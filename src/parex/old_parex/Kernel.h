#pragma once

#include <ostream>
class Task;

/**
*/
class Kernel {
public:
    virtual     ~Kernel         ();

    virtual void write_to_stream( std::ostream &os ) const = 0;
    virtual void exec           ( Task *task ) const = 0;
};

