#include "../../../support/for_each_permutation.h"
#include "../../../support/binary_repr.h"
#include "../../../support/VtkOutput.h"
#include "../../../support/ASSERT.h"
#include "../../../support/TODO.h"
#include "../../../support/P.h"
#include "OutputNodeList.h"
#include "ConvexHull.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <map>

//// nsmake cpp_flag -O3
using TI = std::size_t;

struct GlobalSummary {
    struct PartInfo {
        TI nb_faces;
        TI nb_nodes;
        std::set<TI> dims;
    };

    std::map<std::string,PartInfo> parts;
    std::map<std::string,std::set<TI>> needed_vec_ops_function_names; // name => dims
};

template<int dim>
struct GenCuts {
    using Ch = ConvexHull<dim>;
    using TI = typename Ch::TI;
    using TF = typename Ch::TF;
    using Pt = typename Ch::Pt;

    struct Node {
        std::array<TI,2> inds;
        Pt pos;
    };

    struct Part {
        std::vector<TI> compatible_parts; // index in this list > index of this
        std::vector<TI> num_nodes;
        TI num_ref_shape;
        TF measure;
        Ch ch;
    };

    struct Comb {
        std::pair<TF,TF> score() const { return { measure, - TF( part_inds.size() ) }; }
        std::vector<TI> part_inds;
        TF measure;
    };

    GenCuts( GlobalSummary &gs ) : gs( gs ) {
    }

    void register_shape( std::string name ) {
        ASSERT( name.size() == dim - 1 );

        // start with points on a circle
        TI nb_pts_circle = std::stoi( name );
        P( nb_pts_circle );
        std::vector<Pt> pts( nb_pts_circle, Pt( 0 ) );
        for( TI i = 0; i < nb_pts_circle; ++i ) {
            double a = 2 * M_PI * i / nb_pts_circle;
            pts[ i ][ 0 ] = int( 1000 * cos( a ) );
            pts[ i ][ 1 ] = int( 1000 * sin( a ) );
        }

        // make extrusion/simplex
        for( TI nl = 0, d = 2; nl < name.size(); ++nl ) {
            // extrusion
            if ( name[ nl ] == 'E' ) {
                TI os = pts.size();
                pts.resize( 2 * os );
                for( TI i = 0; i < os; ++i ) {
                    pts[ os + i ] = pts[ i ];
                    pts[ os + i ][ d ] = 1000;
                }
                ++d;
                continue;
            }

            // simplex
            if ( name[ nl ] == 'S' ) {
                Pt p( 0 );
                for( TI i = 0; i <= d; ++i )
                    p[ d ] = 500;
                pts.push_back( p );
                ++d;
                continue;
            }
        }

        // start with a large (hyper-)cube
        std::sort( pts.begin(), pts.end() );
        Ch res( pts, name );

        //        static int cpt = 0;
        //        VtkOutput vo;
        //        res.display_vtk( vo );
        //        vo.save( "out_" + std::to_string( cpt++ ) + ".vtk" );

        gs.parts[ name ].nb_nodes = res.pts.size();
        gs.parts[ name ].nb_faces = res.chi.nexts.size();
        gs.parts[ name ].dims.insert( dim );
        if ( dim == 2 )
            gs.parts[ name ].dims.insert( 3 );

        ref_shapes.push_back( res );

        //        std::string friendly_name;
        //        if ( dim == 2 ) {
        //            switch ( nb_nodes ) {
        //            case 3 : friendly_name = "triangle"; break;
        //            case 4 : friendly_name = "quad"    ; break;
        //            case 5 : friendly_name = "penta"   ; break;
        //            case 6 : friendly_name = "hexa"    ; break;
        //            case 7 : friendly_name = "hepta"   ; break;
        //            case 8 : friendly_name = "octa"    ; break;
        //            case 9 : friendly_name = "nona"    ; break;
        //            case 10: friendly_name = "deca"    ; break;
        //            case 12: friendly_name = "dodeca"  ; break;
        //            default: break;
        //            }
        //        }
        //        ref_shapes.push_back( { pts, friendly_name } );
    }

    void make_cut_nodes_for_ref_shape( TI num_ref_shape ) {
        cut_nodes.clear();

        // nodes of the ref shape
        const Ch &ref_shape = ref_shapes[ num_ref_shape ];
        for( TI i = 0; i < ref_shape.pts.size(); ++i )
            cut_nodes.push_back( { { i, i }, ref_shape.pts[ i ] } );

        // nodes in the middle of the edges
        std::vector<std::set<TI>> links( ref_shape.pts.size() );
        ref_shape.chi.get_links_rec( links );
        for( TI i = 0; i < links.size(); ++i )
            for( TI j : links[ i ] )
                if ( j > i )
                    cut_nodes.push_back( { { i, j }, ( ref_shape.pts[ i ] + ref_shape.pts[ j ] ) / TF( 2 ) } );
    }

    ///
    void make_parts_from_cut_nodes() {
        parts.clear();

        std::map<TI,std::vector<TI>> nb_nodes_to_num_ref_shapes;
        for( TI i = 0; i < ref_shapes.size(); ++i )
            nb_nodes_to_num_ref_shapes[ ref_shapes[ i ].pts.size() ].push_back( i );

        for( auto p : nb_nodes_to_num_ref_shapes ) {
            TI nb_nodes = p.first;
            for_each_comb<TI>( cut_nodes.size(), nb_nodes, [&]( TI *num_nodes ) {
                // get the points
                std::vector<Pt> npts;
                for( TI n = 0; n < nb_nodes; ++n )
                    npts.push_back( cut_nodes[ num_nodes[ n ] ].pos );

                // make a convex hull from the point set
                Part part;
                part.ch = npts;

                // check measure
                part.measure = part.ch.measure();
                if ( part.measure == 0 )
                    return;

                // test if it can be a ref shape
                for( TI num_ref_shape : p.second ) {
                    part.num_ref_shape = num_ref_shape;

                    std::vector<TI> rpc_to_nch( nb_nodes );
                    if ( ref_shapes[ num_ref_shape ].chi.is_a_permutation_of( part.ch.chi, rpc_to_nch.data() ) ) {
                        part.num_nodes.clear();
                        for( TI num_in_nch : rpc_to_nch )
                            part.num_nodes.push_back( num_nodes[ num_in_nch ] );
                        parts.push_back( part );
                    }
                }
            } );
        }
    }

    void set_up_compatible_parts() {
        for( TI i = 0; i < parts.size(); ++i ) {
            parts[ i ].compatible_parts.clear();
            for( TI j = i + 1; j < parts.size(); ++j ) {
                Ch inter = parts[ i ].ch.intersection( parts[ j ].ch );
                if ( inter.measure() == 0 )
                    parts[ i ].compatible_parts.push_back( j );
            }
        }
    }

    void make_best_combs() {
        best_combs.clear();
        make_best_combs( {}, range<TI>( parts.size() ), 0 );
    }

    void make_best_combs( const std::vector<TI> &part_inds, const std::vector<TI> &compatible_parts, TF measure ) {
        // used points
        std::set<TI> points;
        for( TI part_ind : part_inds )
            for( TI num_node : parts[ part_ind ].num_nodes )
                points.insert( num_node );

        // register the comb.
        Comb comb{ part_inds, measure };
        auto iter = best_combs.find( points );
        if ( iter == best_combs.end() )
            best_combs.insert( iter, { points, comb } );
        else if ( iter->second.score() < comb.score() )
            iter->second = comb;

        // add another one if possible
        for( TI compatible_part : compatible_parts ) {
            std::vector<TI> new_compatible_parts;
            new_compatible_parts.reserve( compatible_parts.size() );
            std::set_intersection(
                compatible_parts.begin(), compatible_parts.end(),
                parts[ compatible_part ].compatible_parts.begin(),
                parts[ compatible_part ].compatible_parts.end(),
                std::back_inserter( new_compatible_parts )
            );

            std::vector<TI> new_part_inds = part_inds;
            new_part_inds.push_back( compatible_part );

            make_best_combs( new_part_inds, new_compatible_parts, measure + parts[ compatible_part ].measure );

            //            if ( part_inds.empty() )
            //                P( compatible_part, compatible_parts.size() );
        }
    }

    TI nb_base_nodes() const {
        for( TI i = 0; i < cut_nodes.size(); ++i )
            if ( cut_nodes[ i ].inds[ 0 ] != cut_nodes[ i ].inds[ 1 ] )
                return i;
        return cut_nodes.size();
    }

    void make_comb_for_cases() {
        comb_for_cases.clear();
        for( std::vector<bool> outside_nodes( nb_base_nodes(), false ); ; ) {
            // needed nodes
            std::set<TI> needed_nodes;
            for( TI num_node = 0; num_node < cut_nodes.size(); ++num_node ) {
                // base node
                const Node &node = cut_nodes[ num_node ];
                if ( node.inds[ 0 ] == node.inds[ 1 ] ) {
                    if ( ! outside_nodes[ node.inds[ 0 ] ] )
                        needed_nodes.insert( num_node );
                    continue;
                }

                // edge
                if ( outside_nodes[ node.inds[ 0 ] ] != outside_nodes[ node.inds[ 1 ] ] )
                    needed_nodes.insert( num_node );
            }

            // register
            const Comb &comb = best_combs[ needed_nodes ];
            comb_for_cases.push_back( comb );

            // make a convex hull to check if the measure is correct
            std::vector<Pt> pts;
            for( TI n : needed_nodes )
                pts.push_back( cut_nodes[ n ].pos );

            Ch ch( pts );
            P( outside_nodes, comb.measure, ch.measure() );

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

    void write_to_stream( std::ostream &os ) const {
        //    for( const auto &rs : gc.ref_shapes )
        //        P( rs );
        //    for( const auto &rs : gc.parts )
        //        P( rs.num_ref_shape, rs.measure, rs.num_nodes, rs.compatible_parts );
        //    for( const auto &p : gc.best_combs ) {
        //        P( p.first, p.second.measure, p.second.part_inds );
        //    }
    }

    OutputNodeList make_base_output_node_list( std::map<TI,TI> &src_node_map, const Comb &comb ) {
        OutputNodeList res;
        res.perm_src_nodes = range<TI>( src_node_map.size() );
        for( TI num_part_ind : comb.part_inds ) {
            const Part &part = parts[ num_part_ind ];

            auto iter = std::find_if( res.nbrs.rbegin(), res.nbrs.rend(), [&]( const OutputNodeList::ByRefShape &a ) {
                return a.num_dst_ref_shape == part.num_ref_shape;
            } );
            if ( iter == res.nbrs.rend() ) {
                res.perm_nbrs.push_back( res.nbrs.size() );
                res.nbrs.push_back( {} );
                iter = res.nbrs.rbegin();
                iter->num_dst_ref_shape = part.num_ref_shape;
                iter->perm_dst_nodes = range<TI>( part.num_nodes.size() );
            }

            OutputNodeList::ByRefShape &nbr = *iter;
            std::vector<std::pair<TI,TI>> src_node_list;
            for( TI num_node : part.num_nodes ) {
                TI n0 = src_node_map[ cut_nodes[ num_node ].inds[ 0 ] ];
                TI n1 = src_node_map[ cut_nodes[ num_node ].inds[ 1 ] ];
                src_node_list.push_back( {
                    std::min( n0, n1 ),
                    std::max( n0, n1 )
                } );
            }
            nbr.perm_dst_shapes.push_back( nbr.node_lists.size() );
            nbr.node_lists.push_back( src_node_list );
        }

        return res;
    }

    void write_case( std::ostream &os, TI num_case ) {
        const Comb &comb = comb_for_cases[ num_case ];
        if ( comb.part_inds.empty() )
            return;

        // index for each used src node
        std::map<TI,TI> src_node_map;
        for( TI num_part : comb.part_inds ) {
            const Part &part = parts[ num_part ];
            for( TI num_node : part.num_nodes ) {
                for( TI ind : cut_nodes[ num_node ].inds ) {
                    if ( ! src_node_map.count( ind ) ) {
                        TI n = src_node_map.size();
                        src_node_map[ ind ] = n;
                    }
                }
            }
        }

        //
        OutputNodeList onl = make_base_output_node_list( src_node_map, comb );
        onl.sort_with_fixed_src_node_perm();

        for_each_permutation<TI>( onl.perm_src_nodes, [&]( const std::vector<TI> &perm_src_nodes ) {
            OutputNodeList nonl = onl;
            nonl.perm_src_nodes = perm_src_nodes;
            nonl.sort_with_fixed_src_node_perm();
            if ( onl.summary() > nonl.summary() )
                onl = nonl;
        } );

        // register function name
        std::ostringstream ss;
        onl.write_function_name( ss );
        gs.needed_vec_ops_function_names[ ss.str() ].insert( dim );
        if ( dim == 2 )
            gs.needed_vec_ops_function_names[ ss.str() ].insert( dim + 1 );

        // function call
        onl.write_function_call( os, num_case, ref_shape_names(), src_node_map );
    }

    std::vector<std::string> ref_shape_names() const {
        std::vector<std::string> res;
        for( const Ch &rs : ref_shapes )
            res.push_back( rs.name() );
        return res;
    }

    void write_cases( std::ostream &os, TI num_ref_shape_to_cut ) {
        const Ch &ref_shape_to_cut = ref_shapes[ num_ref_shape_to_cut ];

        os << "if ( name == \"" << ref_shape_to_cut.name() << "\" ) {\n";

        // get scalar products and num cases
        os << "    make_sp_and_cases( dirs, sps, sc, N<" << ref_shape_to_cut.pts.size() << ">(), { ";
        for( TI j = 0; j < ref_shapes.size(); ++j ) {
            if ( j )
                os << ", ";

            // sum of nb ref_shape[ j ] created
            TI nec_ref_shape_j = 0;
            for( const Comb &comb : comb_for_cases )
                for( TI num_part : comb.part_inds )
                    nec_ref_shape_j += parts[ num_part ].num_ref_shape == j;
            if ( nec_ref_shape_j == 0 )
                continue;

            // nb ref_shape[ j ] created for each cut case
            os << "{ \"" << ref_shapes[ j ].name() << "\", { ";
            for( TI num_case = 0; num_case < comb_for_cases.size(); ++num_case ) {
                TI n = 0;
                for( TI num_part : comb_for_cases[ num_case ].part_inds )
                    n += parts[ num_part ].num_ref_shape == j;
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

    void generate( std::string filename ) {
        // std::ostream &os = std::cout;
        std::ofstream os( filename );
        os << "// This file is generated\n";
        for( TI i = 0; i < ref_shapes.size(); ++i ) {
            make_cut_nodes_for_ref_shape( i );
            make_parts_from_cut_nodes();
            P( parts.size() );
            set_up_compatible_parts();
            make_best_combs();
            make_comb_for_cases();
            write_cases( os, i );
        }
    }

    std::vector<Comb>           comb_for_cases;
    std::vector<Ch>             ref_shapes;
    std::map<std::set<TI>,Comb> best_combs;
    std::vector<Node>           cut_nodes;
    std::vector<Part>           parts;

    GlobalSummary&              gs;
};

void write_nb_faces( GlobalSummary &gs ) {
    std::ofstream os( "src/sdot/ConvexPolytop/internal/ConvexPolytopCutGen_nb_boundaries.h" );
    for( auto p : gs.parts )
        os << "if ( name == \"" << p.first << "\" ) return " << p.second.nb_faces << ";\n";
}

void write_nb_nodes( GlobalSummary &gs ) {
    std::ofstream os( "src/sdot/ConvexPolytop/internal/ConvexPolytopCutGen_nb_vertices.h" );
    for( auto p : gs.parts )
        os << "if ( name == \"" << p.first << "\" ) return " << p.second.nb_nodes << ";\n";
}

int write_vec_ops_function_names( GlobalSummary &gs ) {
    std::ofstream fnf( "src/sdot/ConvexPolytop/internal/ConvexPolytopNeededSetVecOps.py" );
    fnf << "lst = [";
    for( auto p : gs.needed_vec_ops_function_names )
        for( TI d : p.second )
            fnf << "\n  ( '" << p.first << "', " << d << " ),";
    fnf << "\n]";

    return system( "touch src/sdot/ConvexPolytop/internal/ConvexPolytopSetVecOps.py" );
}

void write_max_nb_vertices( GlobalSummary &gs ) {
    std::ofstream os( "src/sdot/ConvexPolytop/internal/ConvexPolytopCutGen_max_nb_vertices.h" );
    std::map<TI,TI> res;
    for( TI i = 1; i < 10; ++i )
        res[ i ] = 0;
    for( auto p : gs.parts )
        for( TI d : p.second.dims )
            res[ d ] = std::max( res[ d ], p.second.nb_nodes );

    for( auto p : res )
        os << "if ( dim == " << p.first << " ) return " << p.second << ";\n";
}

void run_2D( GlobalSummary &gs, unsigned max_nb_nodes = 4 ) {
    GenCuts<2> gc( gs );
    for( unsigned nb_nodes = 3; nb_nodes <= max_nb_nodes; ++nb_nodes )
        gc.register_shape( std::to_string( nb_nodes ) );

    gc.generate( "src/sdot/ConvexPolytop/internal/ConvexPolytopCutGen_2D.h" );
}

void run_3D( GlobalSummary &gs, unsigned max_nb_nodes_2D = 4 ) {
    GenCuts<3> gc( gs );
    for( unsigned nb_nodes = 3; nb_nodes <= max_nb_nodes_2D; ++nb_nodes )
        for( char t : std::string( "ES" ) )
            gc.register_shape( std::to_string( nb_nodes ) + t );

    gc.generate( "src/sdot/ConvexPolytop/internal/ConvexPolytopCutGen_3D.h" );
}

int main() {
    GlobalSummary gs;
    run_2D( gs );
    // run_3D( gs );

    write_nb_faces( gs );
    write_nb_nodes( gs );
    write_max_nb_vertices( gs );
    write_vec_ops_function_names( gs );
}





