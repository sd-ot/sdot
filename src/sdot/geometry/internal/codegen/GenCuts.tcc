#include "../../../support/P.h"
#include "GenCuts.h"

template<int dim>
GenCuts<dim>::GenCuts( GlobalGenCutData &gcd ) : gcd( gcd ) {
}

template<int dim>
void GenCuts<dim>::add_ref_shape( std::string name ) {
    ASSERT( name.size() == dim - 1 );
    using Node = typename RefRp::Node;

    // start with points on a circle
    TI nb_pts_circle = std::stoi( name );
    std::vector<Node> nodes( nb_pts_circle, Node{ Pt( TF( 0 ) ), 0 } );
    for( TI i = 0; i < nb_pts_circle; ++i ) {
        double a = 2 * M_PI * i / nb_pts_circle;
        nodes[ i ].pos[ 0 ] = int( std::round( 1000 * cos( a ) ) );
        nodes[ i ].pos[ 1 ] = int( std::round( 1000 * sin( a ) ) );
    }

    // make extrusion/simplex
    for( TI nl = 0, d = 2; nl < name.size(); ++nl ) {
        // extrusion
        if ( name[ nl ] == 'E' ) {
            TI os = nodes.size();
            nodes.resize( 2 * os );
            for( TI i = 0; i < os; ++i ) {
                nodes[ os + i ] = nodes[ i ];
                nodes[ os + i ].pos[ d ] = 1000;
            }
            ++d;
            continue;
        }

        // simplex
        if ( name[ nl ] == 'S' ) {
            Pt p( 0 );
            for( TI i = 0; i <= d; ++i )
                p[ d ] = 500;
            nodes.push_back( { p, 0 } );
            ++d;
            continue;
        }
    }

    // update node data
    for( TI i = 0; i < nodes.size(); ++i )
        nodes[ i ].data = i;

    // register the shape
    RefRp res = RefRp::convex_hull( nodes );
    res.name = name;
    ref_shapes.push_back( res );

    //        static int cpt = 0;
    //        VtkOutput vo;
    //        res.display_vtk( vo );
    //        vo.save( "out_" + std::to_string( cpt++ ) + ".vtk" );

    // global cut data
    gcd.parts[ name ].nb_nodes = nodes.size();
    gcd.parts[ name ].nb_faces = res.faces.size();
    gcd.parts[ name ].dims.insert( dim );
    if ( dim == 2 )
        gcd.parts[ name ].dims.insert( 3 );
}

template<int dim>
void GenCuts<dim>::setup_cut_nodes_for( const RefRp &ref_rp ) {
    using std::min;
    using std::max;

    cut_nodes.clear();
    for( TI i = 0; i < ref_rp.nodes.size(); ++i )
        cut_nodes.push_back( { ref_rp.nodes[ i ].pos, { i, i } } );

    ref_rp.for_each_faces_rec( [&]( auto face ) {
        if ( face.nvi == 1 )
            cut_nodes.push_back( { TF( 1 ) / 2 * ( face.nodes[ 0 ].pos + face.nodes[ 1 ].pos ), { min( face.nodes[ 0 ].data, face.nodes[ 1 ].data ), max( face.nodes[ 0 ].data, face.nodes[ 1 ].data ) } } );
    } );
}

template<int dim>
void GenCuts<dim>::setup_parts_from_cut_nodes( const std::vector<CutNode> &choices, const std::vector<CutNode> &possibilities, const RefRp &ref_shape ) {
    // is it possible to make something valid starting from what we have so far ?
    if ( choices.size() <= ref_shape.nodes.size() && ! ref_shape.valid_node_prop( choices ) )
        return;

    // we have chosen all the points ?
    if ( choices.size() == ref_shape.nodes.size() ) {
        //
        CutRp part = ref_shape.with_nodes( choices );

        // check if it is compatible with previous parts leads to something new
        P( part.measure() );
        return;
    }

    // add a new point
    for( TI i = 0; i < possibilities.size(); ++i ) {
        std::vector<CutNode> new_possibilities = possibilities;
        std::vector<CutNode> new_choices = choices;
        new_possibilities.erase( new_possibilities.begin() + i );
        new_choices.push_back( possibilities[ i ] );
        setup_parts_from_cut_nodes( new_choices, new_possibilities, ref_shape );
    }
}


template<int dim>
void GenCuts<dim>::setup_parts_from_cut_nodes() {
    for( const RefRp &ref_shape : ref_shapes )
        setup_parts_from_cut_nodes( {}, cut_nodes, ref_shape );
}


