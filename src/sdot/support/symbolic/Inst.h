#pragma once

#include <ostream>
#include <vector>

namespace Symbolic {
class Context;

/**
*/
class Inst {
public:
    /**/                Inst           ( Context *context );
    virtual            ~Inst           ();

    virtual void        write_to_stream( std::ostream &os ) const = 0;
    virtual void        write_code     ( std::ostream &os ) const = 0;
    virtual Inst*       simplify       ();

    std::vector<Inst *> children;
    std::vector<Inst *> parents;
    Context*            context;
    std::size_t         date = 0;
    std::string         reg;
};

}
