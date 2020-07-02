#include "../../support/for_each_comb.h"
#include "../../support/StaticRange.h"
#include "../../support/ASSERT.h"
#include "RecursivePolytopImpl.h"
#include "../../support/P.h"

template<class Rp,int nvi>
RecursivePolytopImpl<Rp,nvi>::RecursivePolytopImpl() : next( nullptr ) {
}

template<class Rp>
RecursivePolytopImpl<Rp,1>::RecursivePolytopImpl() : next( nullptr ) {
}

template<class Rp,int nvi> template<class Fu,int n>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu, N<n> ) const {
    for( const Face &face : faces )
        face.for_each_item_rec( fu, N<n>() );
}

template<class Rp,int nvi> template<class Fu>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu, N<nvi> ) const {
    fu( *this );
}

template<class Rp,int nvi> template<class Fu>
void RecursivePolytopImpl<Rp,nvi>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
    for( const Face &face : faces )
        face.for_each_item_rec( fu );
}

template<class Rp> template<class Fu>
void RecursivePolytopImpl<Rp,1>::for_each_item_rec( const Fu &fu, N<1> ) const {
    fu( *this );
}

template<class Rp> template<class Fu>
void RecursivePolytopImpl<Rp,1>::for_each_item_rec( const Fu &fu ) const {
    fu( *this );
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::add_convex_hull( BumpPointerPool &pool, Vertex *vertices, TI *indices, TI nb_indices, Pt *normals, Pt *dirs ) {
    // try each possible face
    for_each_comb<TI>( nvi, nb_indices, indices + nb_indices, [&]( TI *chosen_num_indices ) {
        // normal
        Pt orig = vertices[ indices[ chosen_num_indices[ 0 ] ] ].pos;
        for( TI d = 0; d < nvi - 1; ++d )
            normals[ dim - nvi + d ] = vertices[ indices[ chosen_num_indices[ d + 1 ] ] ].pos - orig;
        Pt normal = cross_prod( normals );
        normals[ dim - nvi ] = normal;

        // test if we already have this face
        for( const Face &face : faces )
            if ( dot( face.normal, orig - face.center ) == 0 && colinear( face.normal, normal ) )
                return;

        // test in and out points
        bool has_ins = false;
        bool has_out = false;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice ) {
            TF d = dot( vertices[ indices[ num_indice ] ].pos - orig, normal );
            has_ins |= d < 0;
            has_out |= d > 0;
        }

        if ( has_ins && has_out )
            return;

        // update normal orientation if necessary
        if ( has_out )
            normal *= TF( -1 );

        // register the new face
        Face *face = pool.create<Face>();
        faces.push_front( face );
        face->normal = normal;

        // find all the points that belong to this face
        TI *new_indices = indices + nb_indices + nvi, new_nb_indices = 0;
        for( TI num_indice = 0; num_indice < nb_indices; ++num_indice )
            if ( dot( vertices[ indices[ num_indice ] ].pos - orig, normal ) == TF( 0 ) )
                new_indices[ new_nb_indices++ ] = indices[ num_indice ];

        // center
        face->center = TF( 0 );
        for( TI i = 0; i < new_nb_indices; ++i )
            face->center += vertices[ new_indices[ i ] ].pos;
        face->center /= TF( new_nb_indices );

        // update of prev_dirs
        dirs[ dim - nvi ] = face->center - center;

        // construct the new face
        face->add_convex_hull( pool, vertices, new_indices, new_nb_indices, normals, dirs );
    } );
}

template<class Rp>
void RecursivePolytopImpl<Rp,1>::add_convex_hull( BumpPointerPool &/*pool*/, Vertex *vertices, TI *indices, TI nb_indices, Pt */*normals*/, Pt *dirs ) {
    Pt d = cross_prod( dirs );
    TF s = dot( vertices[ indices[ 0 ] ].pos, d ), min_s = s, max_s = s;
    this->vertices[ 0 ] = vertices + indices[ 0 ];
    this->vertices[ 1 ] = vertices + indices[ 0 ];
    for( TI i = 1; i < nb_indices; ++i ) {
        TF s = dot( vertices[ indices[ i ] ].pos, d );
        if ( min_s > s ) { min_s = s; this->vertices[ 0 ] = vertices + indices[ i ]; }
        if ( max_s < s ) { max_s = s; this->vertices[ 1 ] = vertices + indices[ i ]; }
    }
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::write_to_stream( std::ostream &os ) const {
    os << "C: "  << center << " N: " << normal;
}

template<class Rp>
void RecursivePolytopImpl<Rp,1>::write_to_stream( std::ostream &os ) const {
    os << "C: "  << center << " N: " << normal << " A:" << vertices[ 0 ]->num << " B:" << vertices[ 1 ]->num;
}

template<class Rp,int nvi>
void RecursivePolytopImpl<Rp,nvi>::plane_cut( RecursivePolytopImpl &res, BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Vertex *> &new_vertices, TI &date ) const {
    ;
}


//template<class Rp,int nvi>
//void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, N<1> ) {
//    //    using std::min;
//    //    using std::max;

//    //    // scalar products
//    //    TF s0 = vertices[ 0 ]->tmp_f;
//    //    TF s1 = vertices[ 1 ]->tmp_f;

//    //    // all inside => nothing to do
//    //    if ( s0 <= 0 && s1 <= 0 )
//    //        return;

//    //    // all outside
//    //    if ( s0 > 0 && s1 > 0 ) {
//    //        vertices.clear();
//    //        faces.clear();
//    //        return;
//    //    }

//    //
//    //    auto set_rp = [&]( Vertex *nv0, Vertex *nv1, TI ind_new ) {
//    //        new_rp.vertices = { pool, 2 };
//    //        new_rp.vertices[ 0 ] = nv0;
//    //        new_rp.vertices[ 1 ] = nv1;
//    //        new_rp.center = TF( 1 ) / 2 * ( new_rp.vertices[ 0 ]->node.pos + new_rp.vertices[ 1 ]->node.pos );
//    //        new_rp.normal = normal;

//    //        for( const Face &face : faces ) {
//    //            Face *new_face = pool.create<Face>();
//    //            new_rp.faces.push_front( new_face );

//    //            new_face->vertices = { pool, 1 };
//    //            new_face->vertices[ 0 ] = new_rp.vertices[ face.vertices[ 0 ] != vertices[ 0 ] ];
//    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
//    //            new_face->normal = face.normal;
//    //        }

//    //        if ( ind_new < 2 ) {
//    //            Face *new_face = pool.create<Face>();
//    //            new_faces.push_front( new_face );

//    //            new_face->vertices = { pool, 1 };
//    //            new_face->vertices[ 0 ] = new_rp.vertices[ ind_new ];
//    //            new_face->center = new_face->vertices[ 0 ]->node.pos;
//    //        }
//    //    };

//    //    // all inside
//    //    if ( s0 <= 0 && s1 <= 0 )
//    //        set_rp( vertices[ 0 ]->tmp_v, vertices[ 1 ]->tmp_v, 2 );

//    //    // only n0 inside
//    //    if ( s0 <= 0 && s1 > 0 ) {
//    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        set_rp( vertices[ 0 ]->tmp_v, new_vertices[ n1 * ( n1 - 1 ) + n0 ], 1 );
//    //    }

//    //    // only n1 inside
//    //    if ( s0 > 0 && s1 <= 0 ) {
//    //        TI n0 = min( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        TI n1 = max( vertices[ 0 ]->tmp_i[ 0 ], vertices[ 1 ]->tmp_i[ 0 ] );
//    //        set_rp( new_vertices[ n1 * ( n1 - 1 ) + n0 ], vertices[ 1 ]->tmp_v, 0 );
//    //    }
//}

//template<class Rp,int nvi> template<class B>
//void RecursivePolytopImpl<Rp,nvi>::plane_cut( BumpPointerPool &pool, IntrusiveList<Face> &new_faces, std::vector<Node *> &new_vertices, TI &date, B ) {
//    //    IntrusiveList<typename Face::Face> new_new_faces;
//    //    faces.remove_if( [&]( Face &face ) {
//    //        face.plane_cut( pool, new_new_faces, new_vertices, date, N<nvi-1>() );
//    //        return face.faces.empty();
//    //    } );

//    //    if ( ! new_new_faces.empty() ) {
//    //        // new face to close new_rp
//    //        Face *new_face_i = pool.create<Face>();
//    //        new_rp.faces.push_front( new_face_i );

//    //        new_face_i->faces = new_new_faces;

//    //        TI tmp_date = date;
//    //        new_face_i->make_vertices_from_face( pool, tmp_date );
//    //        for( Vertex *v : new_face_i->vertices )
//    //            v->date = date;

//    //        // new face to close new_rp
//    //        if ( int( nvi ) < dim ) {
//    //            Face *new_face_o = pool.create<Face>();
//    //            new_faces.push_front( new_face_o );

//    //            new_face_o->faces = new_new_faces;

//    //            TI tmp_date = date;
//    //            new_face_o->make_vertices_from_face( pool, tmp_date );
//    //            for( Vertex *v : new_face_o->vertices )
//    //                v->date = date;
//    //        }
//    //    }

//    //    // update vertices for new_rp
//    //    TI tmp_date = date;
//    //    new_rp.make_vertices_from_face( pool, tmp_date );
//    //    for( Vertex *v : new_rp.vertices )
//    //        v->date = date;
//}

template<class Rp,int nvi>
typename Rp::TF RecursivePolytopImpl<Rp,nvi>::measure( std::array<Pt,dim> &dirs ) const {
    TF res = 0;
    for( const Face &face : faces ) {
        dirs[ dim - nvi ] = face.center - center;
        res += face.measure( dirs );
    }
    return res;
}

template<class Rp>
typename Rp::TF RecursivePolytopImpl<Rp,1>::measure( std::array<Pt,dim> &dirs ) const {
    dirs[ dim - nvi ] = vertices[ 1 ]->pos - vertices[ 0 ]->pos;
    return determinant( dirs[ 0 ].data, N<dim>() );
}

