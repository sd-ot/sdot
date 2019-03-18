#pragma once

#include "Array.h"

/**
*/
template<class T,int dim>
struct CrossProdOfRanges {
    CrossProdOfRanges( Array<T,dim> beg, Array<T,dim> end ) : _beg( beg ), _end( end ) {
    }

    CrossProdOfRanges( Array<T,dim> end ) : CrossProdOfRanges( array_of<T,dim>( 0 ), end ) {
    }

    template<class Func>
    void for_each( Func &&func ) const {
        std::array<T,dim> pos;
        _for_each( func, pos, N<dim>() );
    }

    template<class Func>
    bool for_each_cont( Func &&func ) const {
        std::array<T,dim> pos;
        return _for_each_cont( func, pos, N<dim>() );
    }

private:
    template<class Func,int d>
    void _for_each( const Func &func, std::array<T,dim> pos, N<d> ) const {
        for( pos[ d - 1 ] = _beg[ d - 1 ]; pos[ d - 1 ] < _end[ d - 1 ]; ++pos[ d - 1 ] )
            _for_each( func, pos, N<d-1>() );
    }
    template<class Func>
    void _for_each( const Func &func, std::array<T,dim> pos, N<0> ) const {
        func( pos );
    }

    template<class Func,int d>
    bool _for_each_cont( const Func &func, std::array<T,dim> pos, N<d> ) const {
        for( pos[ d - 1 ] = _beg[ d - 1 ]; pos[ d - 1 ] < _end[ d - 1 ]; ++pos[ d - 1 ] )
            if ( _for_each_cont( func, pos, N<d-1>() ) == false )
                return false;
        return true;
    }
    template<class Func>
    bool _for_each_cont( const Func &func, std::array<T,dim> pos, N<0> ) const {
        return func( pos );
    }

    Array<T,dim> _beg;
    Array<T,dim> _end;
};
