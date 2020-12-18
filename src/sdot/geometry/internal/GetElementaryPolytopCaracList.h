#pragma once

#include <parex/instructions/CompiledLambdaInstruction.h>

namespace sdot {
class SymbolicElementaryPolytop;

struct GetElementaryPolytopCaracList : parex::CompiledInstruction {
public:
    using parex::CompiledInstruction::CompiledInstruction;

    virtual void                    get_src_content( parex::Src &src, parex::SrcSet &/*sw*/, parex::TypeFactory *tf ) const override;
    virtual void                    prepare        ( parex::TypeFactory *tf, parex::SchedulerSession */*ss*/ ) override;
    virtual std::string             summary        () const override;

private:
    void                            write_carac    ( parex::Src &src, const SymbolicElementaryPolytop &se ) const;
    const std::vector<std::string> &shape_names    () const;
    std::string                     type_name      () const;
};

} // namespace sdot