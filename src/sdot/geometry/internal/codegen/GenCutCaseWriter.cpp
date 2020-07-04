#include "../../../support/for_each_permutation.h"
#include "../../../support/range.h"
#include "../../../support/P.h"
#include "GenCutCaseWriter.h"
#include <ostream>

GenCutCaseWriter::GenCutCaseWriter() {

}

GenCutCaseWriter::ByOutputShape &GenCutCaseWriter::by_output_shape( std::string shape_name ) {
    for( ByOutputShape &os : by_output_shapes )
        if ( os.shape_name == shape_name )
            return os;
    by_output_shapes.push_back( { shape_name, {} } );
    return by_output_shapes.back();
}

void GenCutCaseWriter::write_func_name( std::ostream &os ) {
    os << "    recursive_polyhedron_cut";
    for( TI n = 0; n < by_output_shapes.size(); ++n ) {
        ByOutputShape &bos = by_output_shapes[ n ];
        os << ( n ? "__" : "_" ) << n;
        for( const Output &output : bos.outputs ) {
            for( const std::array<TI,2> inds : output.inds )
                for( TI i = 0; i < 2; ++i )
                    os << "_" << src_map[ inds[ i ] ];
        }
    }
}


void GenCutCaseWriter::write_func_args( std::ostream &os, TI num_case ) {
    os << "( beg_cut_cases[ " << num_case << " ], nb_cut_cases[ " << num_case << " ]";
    for( TI n = 0; n < by_output_shapes.size(); ++n ) {
        ByOutputShape &bos = by_output_shapes[ n ];
        os << ", shape_list( tmp_shape_map, \"" << bos.shape_name << "\" ), { ";
        for( TI i = 0; i < bos.outputs.size(); ++i )
            os << ( i ? ", " : "" ) << i;
        os << " }";
    }
    os << ", sc, { ";
    for( TI i = 0; i < num_src_nodes.size(); ++i )
        os << ( i ? ", " : "" ) << num_src_nodes[ i ];
    os << " }, N<dim>() );\n";
}

void GenCutCaseWriter::optimize() {
    // src_map
    for( ByOutputShape &bos : by_output_shapes )
        for( Output &output : bos.outputs )
            for( std::array<TI,2> inds : output.inds )
                for( TI i = 0; i < 2; ++i )
                    src_map.insert( { inds[ i ], src_map.size() } );

    //
    optimize_src_map();
}

void GenCutCaseWriter::write_to( std::ostream &os, TI num_case ) {
    write_func_name( os );
    write_func_args( os, num_case );
}

bool GenCutCaseWriter::operator<( const GenCutCaseWriter &that ) const {
    std::ostringstream a, b;
    const_cast<GenCutCaseWriter *>( this )->write_func_name( a );
    const_cast<GenCutCaseWriter &>( that ) .write_func_name( b );
    return a.str() < b.str();
}

void GenCutCaseWriter::optimize_src_map() {
    GenCutCaseWriter best = *this;
    for_each_permutation<TI>( range<TI>( src_map.size() ), [&]( const std::vector<TI> &perm ) {
        GenCutCaseWriter wr = *this;
        for( auto &p : wr.src_map )
            p.second = perm[ p.second ];

        if ( wr < best )
            best = wr;
    } );

    *this = best;
}



