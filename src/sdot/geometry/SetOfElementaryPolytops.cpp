#include <parex/instructions/CompiledIncludeInstruction.h>
#include <parex/resources/default_cpu_memory.h>
//#include <parex/GeneratedSymbolSet.h>
//#include <parex/variable_encode.h>
//#include <parex/Scheduler.h>
//#include <parex/CppType.h>
//#include <parex/TODO.h>
//#include <parex/P.h>
//#include <sstream>

#include <parex/instructions/CompiledIncludeInstruction.h>
#include "SetOfElementaryPolytops.h"
#include "internal/NewShapeMap.h"

namespace sdot {

SetOfElementaryPolytops::SetOfElementaryPolytops( const ElementaryPolytopTypeSet &elementary_polytop_type_set, const CtorParameters &parameters ) : elem_info( elementary_polytop_type_set.carac ) {
    parex::Memory *dst = parameters.dst ? parameters.dst : parex::default_cpu_memory();
    shape_map = new parex::Variable( new NewShapeMap( elementary_polytop_type_set, parameters.scalar_type, parameters.index_type, parameters.dim, dst ), 5 );
}

void SetOfElementaryPolytops::write_to_stream( std::ostream &os ) const {
    shape_map->display_data( os );
}

void SetOfElementaryPolytops::add_repeated( const parex::String &shape_name, const parex::Scalar &count, const parex::Tensor<> &coordinates, const parex::Vector<> &face_ids, const parex::Scalar &beg_ids ) {
    shape_map->set( new parex::CompiledIncludeInstruction( "sdot/geometry/internal/add_repeated.h", {
        shape_map->get(), shape_name.variable->get(), count.variable->get(), coordinates.variable->get(),
        face_ids.variable->get(), beg_ids.variable->get()
    } ), 0 );
}

void SetOfElementaryPolytops::display_vtk( parex::Scheduler &scheduler, const parex::String &filename ) const {
    scheduler.append( new parex::CompiledIncludeInstruction( "sdot/geometry/internal/display_vtk.h", {
        filename.variable->get(), shape_map->get(), elem_info->get()
    } ) );
}

void SetOfElementaryPolytops::display_vtk( const parex::String &filename ) const {
    parex::Scheduler sch;
    display_vtk( sch, filename );
    sch.run();
}

//void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
//    P( normals );
//}

} // namespace sdot
