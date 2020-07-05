#include "../../../support/for_each_permutation.h"
#include "../../../support/range.h"
#include "../../../support/P.h"
#include "GenCutCaseWriter.h"
#include <ostream>

GenCutCaseWriter::GenCutCaseWriter() {

}

GenCutCaseWriter::ByOutputShape &GenCutCaseWriter::by_output_shape( std::string shape_name, TI nb_vertices ) {
    for( ByOutputShape &os : by_output_shapes )
        if ( os.shape_name == shape_name )
            return os;
    by_output_shapes.push_back( { shape_name, {}, range<TI>( nb_vertices ) } );
    return by_output_shapes.back();
}

void GenCutCaseWriter::write_func_name( std::ostream &os, const ByOutputShape &bos, TI n ) const {
    os << ( n ? "__" : "_" ) << n;
    for( const Output &output : bos.outputs )
        for( const std::array<TI,2> inds : output.inds )
             for( TI i = 0; i < 2; ++i )
                os << "_" << src_map.find( inds[ i ] )->second;
}

void GenCutCaseWriter::write_func_name( std::ostream &os ) const {
    os << "    recursive_polyhedron_cut";
    for( TI n = 0; n < by_output_shapes.size(); ++n )
        write_func_name( os, by_output_shapes[ n ], n );
}

void GenCutCaseWriter::write_func_args( std::ostream &os, TI num_case ) const {
    std::vector<TI> inv_src_map( src_map.size() );
    for( const auto &p : src_map )
        inv_src_map[ p.second ] = p.first;

    os << "( beg_cut_cases[ " << num_case << " ], nb_cut_cases[ " << num_case << " ]";
    for( const ByOutputShape &bos : by_output_shapes ) {
        for( TI o = 0; o < bos.outputs.size(); ++o ) {
            os << ", shape_list( tmp_shape_map, \"" << bos.shape_name << "\" ), { ";
            for( TI i = 0; i < bos.num_dst_vertex.size(); ++i )
                os << ( i ? ", " : "" ) << bos.num_dst_vertex[ i ];
            os << " }";
        }
    }
    os << ", sc, { ";
    for( TI i = 0; i < inv_src_map.size(); ++i )
        os << ( i ? ", " : "" ) << inv_src_map[ i ];
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
    write_func_name( a );
    that.write_func_name( b );
    return a.str() < b.str();
}

void GenCutCaseWriter::optimize_src_map() {
    GenCutCaseWriter best = *this;
    for_each_permutation<TI>( range<TI>( src_map.size() ), [&]( const std::vector<TI> &perm ) {
        GenCutCaseWriter wr = *this;
        for( auto &p : wr.src_map )
            p.second = perm[ p.second ];

        // optimize separately for each output shape
        for( ByOutputShape &bos : wr.by_output_shapes )
            optimize( bos );

        // sort by_output_shapes
        std::sort( wr.by_output_shapes.begin(), wr.by_output_shapes.end(), [&]( const ByOutputShape &a, const ByOutputShape &b ) {
            std::ostringstream sa, sb;
            wr.write_func_name( sa, a, 0 );
            wr.write_func_name( sb, b, 0 );
            return sa.str() < sb.str();
        } );


        if ( wr < best )
            best = wr;
    } );

    *this = best;
}

void GenCutCaseWriter::optimize( ByOutputShape &bos ) {

}



