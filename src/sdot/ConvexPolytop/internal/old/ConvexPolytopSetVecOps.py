import ConvexPolytopNeededSetVecOps
import sys

def flatten( l, a ):
    for i in l:
        if isinstance(i, list):
            flatten( i, a )
        else:
            a.append( i )
    return a

class Elem:
    def __init__( self, name ):
        self.num_in_dst_list = 0 # completed later

        n = [ int( v ) for v in name.split( "_" ) ]
        self.num_dst_list = n[ 0 ]

        self.ind_src_pairs = []
        for i in range( 1, len( n ), 2 ):
            self.ind_src_pairs.append( [ n[ i + 0 ], n[ i + 1 ] ] )


def get_case( dim, name ):
    elems = []
    nb_dst_lists = 0
    for l in name.split( "__" ):
        elem = Elem( l )
        elems.append( elem )
        nb_dst_lists = max( nb_dst_lists, elem.num_dst_list + 1 )

    nb_elem_by_dst_list = [ 0 for _ in range( nb_dst_lists ) ]
    nb_nodes_by_dst_list = [ 0 for _ in range( nb_dst_lists ) ]
    for elem in elems:
        nb_nodes_by_dst_list[ elem.num_dst_list ] = len( elem.ind_src_pairs )
        elem.num_in_dst_list = nb_elem_by_dst_list[ elem.num_dst_list ]
        nb_elem_by_dst_list[ elem.num_dst_list ] += 1

    nb_src_nodes = 0
    for elem in elems:
        for ind_src_pair in elem.ind_src_pairs:
            for ind in ind_src_pair:
                nb_src_nodes = max( nb_src_nodes, ind + 1 )
    # print( nb_nodes_by_dst_list, file=sys.stderr )

    needed_src_disjoint_pairs = []
    for elem in elems:
        for ind_src_pair in elem.ind_src_pairs:
            if ind_src_pair[ 0 ] != ind_src_pair[ 1 ]:
                if not ( ind_src_pair in needed_src_disjoint_pairs ):
                    needed_src_disjoint_pairs.append( ind_src_pair )

    needed_src_pos_inds = []
    for elem in elems:
        for ind_src_pair in elem.ind_src_pairs:
            for ind_src in ind_src_pair:
                if not ( ind_src in needed_src_pos_inds ):
                    needed_src_pos_inds.append( ind_src )

    needed_src_sp_inds = []
    for ind_src_pair in needed_src_disjoint_pairs:
        for ind_src in ind_src_pair:
            if not ( ind_src in needed_src_sp_inds ):
                needed_src_sp_inds.append( ind_src )

    case = ""
    case += "    template<class ShapeCoords>\n"
    case += "    static void run_{name}( const TI *indices_data, TI indices_size, {dst_list_decl}, const ShapeCoords &sc, std::array<TI,{nb_src_nodes}> src_indices, N<{dim}> ) {{\n".format( nb_src_nodes = nb_src_nodes, name = name, dim = dim, dst_list_decl = ",".join( "ShapeCoords &nc_{num}, std::array<TI,{nb_dst_nodes}> dst_{num}_indices".format( num = n, nb_dst_nodes = nb_nodes_by_dst_list[ n ] ) for n in range( nb_dst_lists ) ) )
    case += "        Pos pos;\n"
    case += "        Id id;\n"

    case += "\n"
    for ind_src in needed_src_pos_inds:
        for d in range( dim ):
            case += "        const TF *src_pos_{ind_src}_{ind_src}_{d}_ptr = sc[ pos ][ src_indices[ {ind_src} ] ][ {d} ].data;\n".format( d = d, ind_src = ind_src )

    case += "\n"
    case += "        const TI *src_id_ptr = sc[ id ].data;\n"

    if len( needed_src_sp_inds ):
        case += "\n"
        for ind_src in needed_src_sp_inds:
            case += "        const TF *src_sp_{ind_src}_ptr = sc[ pos ][ src_indices[ {ind_src} ] ][ {d} ].data;\n".format( d = dim, ind_src = ind_src )

    case += "\n"
    for num_dst_list in range( nb_dst_lists ):
        for num_dst_node in range( nb_nodes_by_dst_list[ num_dst_list ] ):
            for d in range( dim ):
                case += "        TF *dst_{num_dst_list}_pos_{num_dst_node}_{d}_ptr = nc_{num_dst_list}[ pos ][ dst_{num_dst_list}_indices[ {num_dst_node} ] ][ {d} ].data + nc_{num_dst_list}.size;\n".format( d = d, num_dst_node = num_dst_node, num_dst_list = num_dst_list )

    case += "\n"
    for num_dst_list in range( nb_dst_lists ):
        case += "        TI *dst_{num_dst_list}_id_ptr = nc_{num_dst_list}[ id ].data + nc_{num_dst_list}.size;\n".format( num_dst_list = num_dst_list )

    case += "\n"
    case += "        SimdRange<SimdSize<TF,Arch>::value>::for_each( indices_size, [&]( TI beg_num_ind, auto simd_size ) {\n"
    case += "            using VI = SimdVec<TI,simd_size.value,Arch>;\n"
    case += "            using VF = SimdVec<TF,simd_size.value,Arch>;\n"
    case += "\n"
    case += "            VI inds = VI::load_aligned( indices_data + beg_num_ind );\n"

    # load pos
    case += "\n"
    for ind_src in needed_src_pos_inds:
        for d in range( dim ):
            case += "            VF src_pos_{ind_src}_{ind_src}_{d} = VF::gather( src_pos_{ind_src}_{ind_src}_{d}_ptr, inds );\n".format( d = d, ind_src = ind_src )

    # load sp
    if len( needed_src_sp_inds ):
        case += "\n"
        for ind_src in needed_src_sp_inds:
            case += "            VF src_sp_{ind_src} = VF::gather( src_sp_{ind_src}_ptr, inds );\n".format( ind_src = ind_src )

    # mult coeffs
    if len( needed_src_disjoint_pairs ):
        case += "\n"
        for ind_src_pair in needed_src_disjoint_pairs:
            case += "            VF m_{ind_src_0}_{ind_src_1} = src_sp_{ind_src_0} / ( src_sp_{ind_src_0} - src_sp_{ind_src_1} );\n".format( ind_src_0 = ind_src_pair[ 0 ], ind_src_1 = ind_src_pair[ 1 ] )

        case += "\n"
        for ind_src_pair in needed_src_disjoint_pairs:
            for d in range( dim ):
                case += "            VF src_pos_{ind_src_0}_{ind_src_1}_{d} = src_pos_{ind_src_0}_{ind_src_0}_{d} + m_{ind_src_0}_{ind_src_1} * ( src_pos_{ind_src_1}_{ind_src_1}_{d} - src_pos_{ind_src_0}_{ind_src_0}_{d} );\n".format( ind_src_0 = ind_src_pair[ 0 ], ind_src_1 = ind_src_pair[ 1 ], d = d )

    # load id
    case += "\n"
    case += "            VI ids = VI::gather( src_id_ptr, inds );\n"

    # compute + store
    case += "\n"
    for elem in elems:
        for ind_dst, ind_src_pair in enumerate( elem.ind_src_pairs ):
            for d in range( dim ):
                case += "            VF::store( dst_{num_dst_list}_pos_{ind_dst}_{d}_ptr + {nb_added} * beg_num_ind + {num_in_dst_list} * simd_size.value, src_pos_{ind_src_0}_{ind_src_1}_{d} );\n".format( d = d, ind_dst = ind_dst, ind_src_0 = ind_src_pair[ 0 ], ind_src_1 = ind_src_pair[ 1 ], num_dst_list = elem.num_dst_list, num_in_dst_list = elem.num_in_dst_list, nb_added = nb_elem_by_dst_list[ elem.num_dst_list ] )

    case += "\n"
    for elem in elems:
        case += "            VI::store( dst_{num_dst_list}_id_ptr + {nb_added} * beg_num_ind + {num_in_dst_list} * simd_size.value, ids );\n".format( num_dst_list = elem.num_dst_list, num_in_dst_list = elem.num_in_dst_list, nb_added = nb_elem_by_dst_list[ elem.num_dst_list ] )
    case += "        } );\n"
    case += "\n"
    for num_dst_list in range( nb_dst_lists ):
        case += "        nc_{num_dst_list}.size += indices_size * {nb_added};\n".format( num_dst_list = num_dst_list, nb_added = nb_elem_by_dst_list[ num_dst_list ] )
    case += "    }\n"
    case += "\n"

    return case

cases = ""
for c in ConvexPolytopNeededSetVecOps.lst:
    cases += get_case( c[ 1 ], c[ 0 ] )

code = """#pragma once

#include "../../support/simd/SimdRange.h"
#include "../../support/simd/SimdVec.h"
#include "../../support/StaticRange.h"
#include <array>

/*
*/
template<class TF,class TI,class Arch,class Pos,class Id>
struct ConvexPolytopSetVecOps {
CASES};
"""

print( code.replace( "CASES", cases ) )

