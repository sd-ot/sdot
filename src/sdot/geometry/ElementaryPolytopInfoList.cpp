#include "ElementaryPolytopInfoList.h"
#include <parex/CompiledTask.h>

namespace sdot {

ElementaryPolytopInfoList::ElementaryPolytopInfoList( const Value &dim_or_shape_types ) {
    struct GetElementaryPolytopInfoList : CompiledTask {
        using CompiledTask::CompiledTask;

        virtual void write_to_stream( std::ostream &os ) const  override {
            os << "GetElementaryPolytopInfoList";
        }

        virtual void get_src_content( Src &src, SrcSet &/*sw*/ ) override {
            src << "TaskOut<ElementaryPolytopInfoListContent> kernel( const TaskOut<std::string> & ) {\n";
            src << "    ElementaryPolytopInfoListContent *res = new ElementaryPolytopInfoListContent;\n";
            src << "    return res;\n";
            src << "}\n";
            src << "\n";
        }

        virtual std::string summary() override {
            return "";
        }
    };

    task = new GetElementaryPolytopInfoList( { dim_or_shape_types.to_string() } );
}

void ElementaryPolytopInfoList::write_to_stream( std::ostream &os ) const {
    os << Value( task );
}

} // namespace sdot
