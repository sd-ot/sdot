#include "../../../support/P.h"
#include "GenCutCaseWriter.h"
#include "GenSetVecOps.h"
#include "GenCuts.h"
#include <algorithm>

template<int dim>
GenCuts<dim>::GenCuts( GlobalGenCutData &gcd ) : global_cut_data( gcd ) {
}

template<int dim>
void GenCuts<dim>::add_ref_shape( std::string name ) {
    ASSERT( name.size() == dim - 1 );

    // start with points on a circle
    TI nb_pts_circle = std::stoi( name );
    std::vector<Pt> pts( nb_pts_circle, TF( 0 ) );
    for( TI i = 0; i < nb_pts_circle; ++i ) {
        double a = 2 * M_PI * i / nb_pts_circle;
        pts[ i ][ 0 ] = int( std::round( 2520 * cos( a ) ) ) / TF( 1000 );
        pts[ i ][ 1 ] = int( std::round( 2520 * sin( a ) ) ) / TF( 1000 );
    }

    // make extrusion or simplex
    for( TI nl = 1, d = 2; nl < name.size(); ++nl ) {
        // extrusion
        if ( name[ nl ] == 'E' ) {
            TI os = pts.size();
            pts.resize( 2 * os );
            for( TI i = 0; i < os; ++i ) {
                pts[ os + i ] = pts[ i ];
                pts[ os + i ][ d ] = 5040 / TF( 1000 );
            }
            ++d;
            continue;
        }

        // simplex
        if ( name[ nl ] == 'S' ) {
            Pt p( 0 );
            for( TI i = 0; i <= d; ++i )
                p[ d ] = 5040 / TF( 1000 );
            pts.push_back( p );
            ++d;
            continue;
        }

        // ??
        TODO;
    }

    // register the shape
    ref_shapes.push_back( { pts, name } );
    ref_shapes.back().rp.make_convex_hull();

    // global cut data
    global_cut_data.parts[ name ].nb_nodes = pts.size();
    // gcd.parts[ name ].nb_faces = res.faces.size();
    global_cut_data.parts[ name ].dims.insert( dim );
}

template<int dim>
void GenCuts<dim>::setup_cut_nodes_for( const Shape &ref_shape ) {
    using std::min;
    using std::max;

    ref_shape_to_cut = &ref_shape;

    cut_nodes.clear();
    for( TI i = 0; i < ref_shape.rp.nb_vertices(); ++i )
        cut_nodes.push_back( { { i, i }, ref_shape.rp.vertex( i ).pos } );

    ref_shape.rp.for_each_item_rec( [&]( const auto &edge ) {
        Pt pt = TF( 1 ) / 2 * ( edge.vertices[ 0 ]->pos + edge.vertices[ 1 ]->pos );
        TI n0 = min( edge.vertices[ 0 ]->num, edge.vertices[ 1 ]->num );
        TI n1 = max( edge.vertices[ 0 ]->num, edge.vertices[ 1 ]->num );
        cut_nodes.push_back( { { n0, n1 }, pt } );
    }, N<1>() );
}

template<int dim>
void GenCuts<dim>::setup_parts_from_cut_nodes() {
    // get valid parts
    std::vector<TI> indices( cut_nodes.size() );
    for( const Shape &ref_shape : ref_shapes ) {
        for_each_comb<TI>( ref_shape.rp.nb_vertices(), cut_nodes.size(), indices.data(), [&]( const TI *chosen_indices ) {
            // get coordinates of the choosen nodes
            std::vector<Pt> pts( ref_shape.rp.nb_vertices() );
            for( TI i = 0; i < pts.size(); ++i )
                pts[ i ] = cut_nodes[ chosen_indices[ i ] ].pos;

            // find if the points can be used to construct the ref_shape
            std::vector<TI> num_in_pts( pts.size() );
            if ( ref_shape.rp.can_use_perm_pts( pts.data(), num_in_pts.data() ) ) {
                std::vector<Pt> npts( pts.size() );
                for( TI i = 0; i < pts.size(); ++i )
                    npts[ i ] = pts[ num_in_pts[ i ] ];

                 parts.push_back( { ref_shape.rp.with_points( npts ), &ref_shape } );
                 Part *p = &parts.back();

                 p->measure = p->rp.measure();

                 p->cut_nodes.resize( pts.size() );
                 for( TI i = 0; i < pts.size(); ++i ) {
                     p->cut_nodes[ i ] = cut_nodes[ chosen_indices[ num_in_pts[ i ] ] ];
                     ASSERT( p->cut_nodes[ i ].pos == npts[ i ] );
                 }

                 // find compatible parts
                 for( TI num_part = 0; num_part < parts.size() - 1; ++num_part ) {
                     Part &q = parts[ num_part ];
                     if ( Rp::measure_intersection( q.rp, p->rp ) == 0 )
                         q.compatible_parts.push_back( p );
                 }
            }
        } );
    }

    // sort compatible_shapes
    for( Part &p : parts )
        std::sort( p.compatible_parts.begin(), p.compatible_parts.end() );
}

template<int dim>
void GenCuts<dim>::make_best_combs_from_parts( const std::vector<Part *> &chosen_parts, const std::vector<Part *> &compatible_parts, TF measure, const std::set<std::array<TI,2>> &used_points ) {
    if ( chosen_parts.empty() && compatible_parts.empty() ) {
        std::vector<Part *> new_compatible_parts;
        new_compatible_parts.reserve( parts.size() );
        for( Part &p : parts )
            new_compatible_parts.push_back( &p );
        return make_best_combs_from_parts( chosen_parts, new_compatible_parts, measure, used_points );
    }

    // register the comb.
    Comb comb{ chosen_parts, measure };
    auto iter = best_combs.find( used_points );
    if ( iter == best_combs.end() )
        best_combs.insert( iter, { used_points, comb } );
    else if ( iter->second.score() < comb.score() )
        iter->second = comb;

    // add another one if possible
    for( Part *part : compatible_parts ) {
        // new chosen_parts
        std::vector<Part *> new_chosen_parts = chosen_parts;
        new_chosen_parts.push_back( part );

        // new compatible_parts
        std::vector<Part *> new_compatible_parts;
        new_compatible_parts.reserve( compatible_parts.size() );
        std::set_intersection(
            compatible_parts.begin(),
            compatible_parts.end(),
            part->compatible_parts.begin(),
            part->compatible_parts.end(),
            std::back_inserter( new_compatible_parts )
        );

        // new measure
        TF new_measure = measure;
        new_measure += part->measure;

        // new user_points
        std::set<std::array<TI,2>> new_used_points = used_points;
        for( const CutNode &cn : part->cut_nodes )
            new_used_points.insert( cn.inds );

        make_best_combs_from_parts( new_chosen_parts, new_compatible_parts, new_measure, new_used_points );
    }
}

template<int dim>
void GenCuts<dim>::makes_comb_for_cases() {
    comb_for_cases.clear();
    for( std::vector<bool> outside_nodes( ref_shape_to_cut->rp.nb_vertices(), false ); ; ) {
        // needed nodes
        std::set<std::array<TI,2>> needed_inds;
        for( TI num_node = 0; num_node < cut_nodes.size(); ++num_node ) {
            // base node
            const CutNode &node = cut_nodes[ num_node ];
            if ( node.inds[ 0 ] == node.inds[ 1 ] ) {
                if ( ! outside_nodes[ node.inds[ 0 ] ] )
                    needed_inds.insert( node.inds );
                continue;
            }

            // edge
            if ( outside_nodes[ node.inds[ 0 ] ] != outside_nodes[ node.inds[ 1 ] ] )
                needed_inds.insert( node.inds );
        }

        // register
        const Comb &comb = best_combs[ needed_inds ];
        comb_for_cases.push_back( comb );

        // next set of outside nodes
        TI i = 0;
        for( ; i < outside_nodes.size(); ++i ) {
            if ( outside_nodes[ i ] == false ) {
                outside_nodes[ i ] = true;
                break;
            }
            outside_nodes[ i ] = false;
        }
        if ( i == outside_nodes.size() )
            break;
    }
}

template<int dim>
void GenCuts<dim>::write_code_for_cases() {
    std::ostream &os = std::cout;

    os << "if ( name == \"" << ref_shape_to_cut->name << "\" ) {\n";

    // get scalar products and num cases
    os << "    make_sp_and_cases( dirs, sps, sc, N<" << ref_shape_to_cut->rp.nb_vertices() << ">(), { ";
    for( TI j = 0; j < ref_shapes.size(); ++j ) {
        if ( j )
            os << ", ";

        // sum of nb ref_shape[ j ] created
        TI nec_ref_shape_j = 0;
        for( const Comb &comb : comb_for_cases )
            for( Part *part : comb.parts )
                nec_ref_shape_j += part->ref_shape == &ref_shapes[ j ];
        if ( nec_ref_shape_j == 0 )
            continue;

        // nb ref_shape[ j ] created for each cut case
        os << "{ \"" << ref_shapes[ j ].name << "\", { ";
        for( TI num_case = 0; num_case < comb_for_cases.size(); ++num_case ) {
            TI n = 0;
            for( Part *part : comb_for_cases[ num_case ].parts )
                n += part->ref_shape == &ref_shapes[ j ];
            os << ( num_case ? ", " : "" ) << n;
        }
        os << " } }";
    }
    os << " } );\n";

    // creation of new elements for each case
    os << "\n";
    for( TI num_case = 0; num_case < comb_for_cases.size(); ++num_case )
        write_case( os, num_case );
    os << "    continue;\n";
    os << "}\n";
}


template<int dim>
void GenCuts<dim>::write_case( std::ostream &os, TI num_case ) {
    Comb &comb = comb_for_cases[ num_case ];
    if ( comb.parts.empty() )
        return;

    GenCutCaseWriter gw;
    for( Part *part : comb.parts ) {
        std::vector<std::array<TI,2>> nodes;
        for( const CutNode &cn : part->cut_nodes )
            nodes.push_back( cn.inds );
        gw.add_output( part->ref_shape->name, nodes );
    }

    gw.optimize();
    gw.write( os, num_case );

    //
    cut_op_names.insert( gw.func_name() );
}

template<int dim>
void GenCuts<dim>::write_cut_op_funcs() {
    std::ostream &os = std::cout;
    os << "template<class TF,class TI,class Arch,class Pos,class Id>\n";
    os << "struct RecursivePolyhedronCutVecOp_" << dim << " {\n";
    for( std::string str : cut_op_names ) {
        GenSetVecOps gs( str, dim );
        gs.write( os );
    }
    os << "};\n";
}

template<int dim>
void GenCuts<dim>::display_best_combs() const {
    for( const auto &p : best_combs ) {
        std::cout << p.first;
        for( const Part *part : p.second.parts )
            std::cout << "\n  " << part->name << " " << part->cut_nodes;
    }
}

template<int dim>
void GenCuts<dim>::display_ref_shape() const {
    for( TI cpt = 0; cpt < ref_shapes.size(); ++cpt ) {
        VtkOutput vo;
        ref_shapes[ cpt ].rp.display_vtk( vo );
        vo.save( "ref_shape_" + std::to_string( cpt++ ) + ".vtk" );
    }
}

template<int dim>
void GenCuts<dim>::display_parts() const {
    for( TI cpt = 0; cpt < parts.size(); ++cpt ) {
        VtkOutput vo;
        parts[ cpt ].rp.display_vtk( vo );
        vo.save( "part_" + std::to_string( cpt ) + ".vtk" );
    }
}

