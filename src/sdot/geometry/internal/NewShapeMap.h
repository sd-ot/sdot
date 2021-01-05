#pragma once

#include <parex/instructions/CompiledInstruction.h>
#include <parex/resources/Memory.h>

#include "../ElementaryPolytopTypeSet.h"

namespace sdot {
class ElementaryPolytopTypeSet;

/**
*/
struct NewShapeMap : parex::CompiledInstruction {
    /**/                NewShapeMap     ( const ElementaryPolytopTypeSet &elementary_polytop_type_set, const parex::String &scalar_type, const parex::String &index_type, const parex::Scalar &dim, parex::Memory *dst );

    virtual void        get_src_content ( parex::Src &src, parex::SrcSet &, parex::TypeFactory * ) const override;
    virtual void        prepare         ( parex::TypeFactory *tf, parex::SchedulerSession *) override;
    virtual std::string summary         () const override;

    virtual std::string output_type_name() const;

    parex::Memory*      dst;
};

} // namespace sdot
