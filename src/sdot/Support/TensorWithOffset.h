#pragma once

#include "Array.h"
#include "Math.h"
#include "N.h"

/**
*/
template<class T,int dim,class Coord=std::size_t,class Index=std::size_t>
class TensorWithOffset {
public:
    TensorWithOffset() {
        _r_beg.fill( std::numeric_limits<Coord>::max() );
        _r_end.fill( std::numeric_limits<Coord>::min() );
        _beg  .fill( std::numeric_limits<Coord>::max() );
        _end  .fill( std::numeric_limits<Coord>::min() );
        if ( dim == 0 )
            _content.resize( 1 );
    }

    ///
    void resize( Array<Coord,dim> beg, Array<Coord,dim> end, Coord rese_borders = 0 ) {
        reserve( beg - rese_borders, end + rese_borders );
        _beg = beg;
        _end = end;
    }

    /// allocate memory, but do not change the size
    void reserve( Array<Coord,dim> r_beg, Array<Coord,dim> r_end ) {
        r_beg = std::min( r_beg, _r_beg );
        r_end = std::max( r_end, _r_end );

        Index c = 1;
        Array<Index,dim> r_cum;
        for( int d = 0; d < dim; ++d ) {
            c *= r_end[ d ] - r_beg[ d ];
            r_cum[ d ] = c;
        }

        // if we had data
        if ( dim && _beg[ 0 ] <= _end[ 0 ] ) {
            std::vector<T> content( c );
            for_each_coords( [&]( Array<Coord,dim> pos ) {
                content[ _index_at( pos, r_beg, r_cum ) ] = _content[ _index_at( pos, _r_beg, _r_cum ) ];
            } );
            std::swap( content, _content );
        } else {
            _content.resize( c );
        }

        _r_beg = r_beg;
        _r_end = r_end;
        _r_cum = r_cum;
    }

    inline T &operator[]( Array<Coord,dim> pos ) {
        return _content[ index_at( pos ) ];
    }

    inline const T &operator[]( Array<Coord,dim> pos ) const {
        return _content[ index_at( pos ) ];
    }

    T &operator[]( Index index ) {
        return _content[ index ];
    }

    const T &operator[]( Index index ) const {
        return _content[ index ];
    }

    inline Index index_at( Array<Coord,dim> pos ) const {
        return _index_at( pos, _r_beg, _r_cum );
    }

    template<class Func>
    void for_each_coords( Func &&func ) const {
        _for_each_coords( std::forward<Func>( func ), N<dim>() );
    }

    void write_to_stream( std::ostream &os ) const {
        for_each_coords( [&]( Array<Coord,dim> pos ) {
            if ( dim >= 2 && ! ( pos == _beg ) )
                os << ( pos[ 0 ] == _beg[ 0 ] ? "\n" : " " );
            os << operator[]( pos );
        } );

    }

    inline Array<Coord,dim> beg() const {
        return _beg;
    }

    inline Array<Coord,dim> end() const {
        return _end;
    }

private:
    Index _index_at( Array<Coord,0> pos, Array<Coord,dim> r_beg, Array<Index,dim> r_cum ) const { return 0; }
    Index _index_at( Array<Coord,1> pos, Array<Coord,dim> r_beg, Array<Index,dim> r_cum ) const { return ( pos[ 0 ] - r_beg[ 0 ] ); }
    Index _index_at( Array<Coord,2> pos, Array<Coord,dim> r_beg, Array<Index,dim> r_cum ) const { return ( pos[ 0 ] - r_beg[ 0 ] ) + ( pos[ 1 ] - r_beg[ 1 ] ) * r_cum[ 0 ]; }
    Index _index_at( Array<Coord,3> pos, Array<Coord,dim> r_beg, Array<Index,dim> r_cum ) const { return ( pos[ 0 ] - r_beg[ 0 ] ) + ( pos[ 1 ] - r_beg[ 1 ] ) * r_cum[ 0 ] + ( pos[ 2 ] - r_beg[ 2 ] ) * r_cum[ 1 ]; }

    template<class Func>
    void _for_each_coords( Func &&func, N<0> ) const {
        func( Array<Coord,0>{} );
    }

    template<class Func>
    void _for_each_coords( Func &&func, N<1> ) const {
        for( Coord x = _beg[ 0 ]; x < _end[ 0 ]; ++x )
            func( Array<Coord,1>{ x } );
    }

    template<class Func>
    void _for_each_coords( Func &&func, N<2> ) const {
        for( Coord y = _beg[ 1 ]; y < _end[ 1 ]; ++y )
            for( Coord x = _beg[ 0 ]; x < _end[ 0 ]; ++x )
                func( Array<Coord,2>{ x, y } );
    }

    template<class Func>
    void _for_each_coords( Func &&func, N<3> ) const {
        for( Coord z = _beg[ 2 ]; z < _end[ 2 ]; ++z )
            for( Coord y = _beg[ 1 ]; y < _end[ 1 ]; ++y )
                for( Coord x = _beg[ 0 ]; x < _end[ 0 ]; ++x )
                    func( Array<Coord,3>{ x, y, z } );
    }

    std::vector<T>   _content; ///<
    Array<Coord,dim> _r_beg;   ///< reserved begin
    Array<Coord,dim> _r_end;   ///< reserved end
    Array<Index,dim> _r_cum;   ///< cumulative product of ( end - beg )
    Array<Coord,dim> _beg;     ///<
    Array<Coord,dim> _end;     ///<
};
