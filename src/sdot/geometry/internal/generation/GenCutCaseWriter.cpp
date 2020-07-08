#include "../../../support/for_each_permutation.h"
#include "../../../support/range.h"
#include "../../../support/P.h"
#include "GenCutCaseWriter.h"
#include <algorithm>
#include <ostream>

GenCutCaseWriter::GenCutCaseWriter( const GenCutCaseWriter &that ) {
    by_output_shapes = that.by_output_shapes;
    src_map          = that.src_map;

    for( ByOutputShape &bos : by_output_shapes ) {
        bos.src_map = &src_map;
        for( Output &output : bos.outputs ) {
            output.num_dst_vertex = &bos.num_dst_vertex;
            output.src_map = &src_map;
        }
    }
}

GenCutCaseWriter::GenCutCaseWriter() {
}

void GenCutCaseWriter::operator=( const GenCutCaseWriter &that ) {
    by_output_shapes = that.by_output_shapes;
    src_map          = that.src_map;

    for( ByOutputShape &bos : by_output_shapes ) {
        bos.src_map = &src_map;
        for( Output &output : bos.outputs ) {
            output.num_dst_vertex = &bos.num_dst_vertex;
            output.src_map = &src_map;
        }
    }
}

void GenCutCaseWriter::add_output( std::string shape_name, std::vector<Node> nodes ) {
    ByOutputShape &bos = by_output_shape( shape_name, nodes.size() );
    bos.outputs.push_back( { &bos.num_dst_vertex, &src_map, nodes } );

    // src_map
    for( const Node &node : nodes )
        for( TI i = 0; i < 2; ++i )
            src_map.insert( { node[ i ], src_map.size() } );
}

void GenCutCaseWriter::optimize() {
    // w.r.t. scr_map
    GenCutCaseWriter best = *this;
    for_each_permutation<TI>( range<TI>( src_map.size() ), [&]( const std::vector<TI> &perm ) {
        GenCutCaseWriter wr = *this;
        for( auto &p : wr.src_map )
            p.second = perm[ p.second ];

        // optimize separately for each output shape
        for( ByOutputShape &bos : wr.by_output_shapes )
            bos.optimize();

        // sort by_output_shapes
        std::sort( wr.by_output_shapes.begin(), wr.by_output_shapes.end(), [&]( const ByOutputShape &a, const ByOutputShape &b ) {
            return a.func_name() < b.func_name();
        } );

        if ( wr.func_name() < best.func_name() )
            best = wr;
    } );

    *this = best;
}

void GenCutCaseWriter::write( std::ostream &os, TI num_case ) {
    os << "    RVO::" << func_name() << "( " << func_args( num_case ) << " );\n";
}


GenCutCaseWriter::ByOutputShape &GenCutCaseWriter::by_output_shape( std::string shape_name, TI nb_nodes ) {
    for( ByOutputShape &os : by_output_shapes )
        if ( os.shape_name == shape_name )
            return os;
    by_output_shapes.push_back( { range<TI>( nb_nodes ), shape_name, &src_map, {} } );
    return by_output_shapes.back();
}

std::string GenCutCaseWriter::func_args( TI num_case ) const {
    // inv_src_map
    std::vector<TI> inv_src_map( src_map.size() );
    for( const auto &p : src_map )
        inv_src_map[ p.second ] = p.first;

    // outputs
    std::string res = "tmp_indices_bcc.data() + " + std::to_string( num_case ) + " * cut_chunk_size, tmp_offsets_bcc[ " + std::to_string( num_case ) + " ] - " + std::to_string( num_case ) + " * cut_chunk_size";
    for( const ByOutputShape &bos : by_output_shapes )
        res += bos.func_args();

    // input
    res += ", sc, { ";
    for( TI i = 0; i < inv_src_map.size(); ++i )
        res += ( i ? ", " : "" ) + std::to_string( inv_src_map[ i ] );
    res += " }";

    return res;
}

std::string GenCutCaseWriter::func_name() const {
    std::string res = "cut";
    for( TI n = 0; n < by_output_shapes.size(); ++n )
        res += by_output_shapes[ n ].func_name( n );
    return res;
}

std::string GenCutCaseWriter::Output::func_name() const {
    std::string res;
    for( TI num_ind : *num_dst_vertex ) {
        Node node = orig_node( num_ind );
        res += "_" + std::to_string( node[ 0 ] );
        res += "_" + std::to_string( node[ 1 ] );
    }
    return res;
}

GenCutCaseWriter::Node GenCutCaseWriter::Output::orig_node( TI num_ind ) const {
    using std::min;
    using std::max;
    TI n0 = src_map->find( nodes[ num_ind ][ 0 ] )->second;
    TI n1 = src_map->find( nodes[ num_ind ][ 1 ] )->second;
    return { min( n0, n1 ), max( n0, n1 ) };
}

std::string GenCutCaseWriter::ByOutputShape::func_name( TI num_output_shape ) const {
    std::string res;
    for( const Output &output : outputs )
        res += "_l" + std::to_string( num_output_shape ) + output.func_name();
    return res;
}

std::string GenCutCaseWriter::ByOutputShape::func_args() const {
    std::string res = ", shape_list( tmp_shape_map, \"" + shape_name + "\" ), { ";
    for( TI i = 0; i < num_dst_vertex.size(); ++i )
        res += ( i ? ", " : "" ) + std::to_string( num_dst_vertex[ i ] );
    res += " }";
    return res;
}

void GenCutCaseWriter::ByOutputShape::optimize() {
    ByOutputShape best = *this;
    for( TI first_choice = 0; first_choice < outputs.size(); ++first_choice ) {
        // for each choice of first output
        ByOutputShape wr = *this;
        std::swap( wr.outputs[ 0 ], wr.outputs[ first_choice ] );

        // udpate wr.num_dst_vertex based on this first choice
        std::sort( wr.num_dst_vertex.begin(), wr.num_dst_vertex.end(), [&]( TI a, TI b ) {
            return wr.outputs[ 0 ].orig_node( a ) < wr.outputs[ 0 ].orig_node( b );
        } );

        // sort outputs with the updated wr.num_dst_vertex
        std::sort( wr.outputs.begin(), wr.outputs.end(), [&]( const Output &a, const Output &b ) {
            return a.func_name() < b.func_name();
        } );

        //
        if ( wr.func_name() < best.func_name() )
            best = wr;
    }

    *this = best;
}



