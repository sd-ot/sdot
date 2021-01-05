#include <parex/instructions/CompiledIncludeInstruction.h>
#include <parex/resources/default_cpu_memory.h>
//#include <parex/GeneratedSymbolSet.h>
//#include <parex/variable_encode.h>
//#include <parex/Scheduler.h>
//#include <parex/CppType.h>
//#include <parex/TODO.h>
//#include <parex/P.h>
//#include <sstream>

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

//void SetOfElementaryPolytops::add_repeated( const Value &shape_name, const Value &count, const Value &coordinates, const Value &face_ids, const Value &beg_ids ) {
//    shape_map = new CompiledIncludeTask( "sdot/geometry/internal/add_repeated.h", { shape_map, shape_name.task, count.task, coordinates.task, face_ids.task, beg_ids.task } );
//}

//void SetOfElementaryPolytops::display_vtk( const Value &filename ) const {
//    scheduler.run( new CompiledIncludeTask( "sdot/geometry/internal/display_vtk.h", { filename.task, shape_map, elem_info }, {}, std::numeric_limits<double>::max() ) );
//}

//void SetOfElementaryPolytops::plane_cut( const Value &normals, const Value &scalar_products, const Value &cut_ids ) {
//    P( normals );
//}

} // namespace sdot
