#pragma once

#include <functional>
#include <vector>

namespace  {
    //template<class TI>
    //void _for_each_comb( TI rese_size, TI list_size, const std::function<void( const std::vector<TI> & )> &f, std::vector<TI> &res, TI n, bool ordered ) {
    //    if ( n == list_size )
    //        return f( res );

    //    if ( ordered ) {
    //        for( TI val = 0; val < rese_size; ++val ) {
    //            bool already_in = false;
    //            for( TI i = 0; i < n; ++i )
    //                already_in |= res[ i ] == val;
    //            if ( already_in )
    //                continue;

    //            res[ n ] = val;
    //            _for_each_comb( rese_size, list_size, f, res, n + 1, ordered );
    //        }
    //    } else {
    //        for( TI val = n ? res[ n - 1 ] + 1 : 0; val < rese_size; ++val ) {
    //            res[ n ] = val;
    //            _for_each_comb( rese_size, list_size, f, res, n + 1, ordered );
    //        }
    //    }
    //}

    //template<class TI>
    //bool _for_each_comb_cont( TI rese_size, TI list_size, const std::function<bool( const std::vector<TI> & )> &f, std::vector<TI> &res, TI n, bool ordered ) {
    //    if ( n == list_size )
    //        return f( res );

    //    if ( ordered ) {
    //        for( TI val = 0; val < rese_size; ++val ) {
    //            bool already_in = false;
    //            for( TI i = 0; i < n; ++i )
    //                already_in |= res[ i ] == val;
    //            if ( already_in )
    //                continue;

    //            res[ n ] = val;
    //            if ( ! _for_each_comb_cont( rese_size, list_size, f, res, n + 1, ordered ) )
    //                return false;
    //        }
    //    } else {
    //        for( TI val = n ? res[ n - 1 ] + 1 : 0; val < rese_size; ++val ) {
    //            res[ n ] = val;
    //            if ( ! _for_each_comb_cont( rese_size, list_size, f, res, n + 1, ordered ) )
    //                return false;
    //        }
    //    }

    //    return true;
    //}

    template<class TI>
    void _for_each_comb( TI selection_size, TI nb_available_items, TI *room_for_selection, const std::function<void( TI * )> &f, TI n ) {
        if ( n == selection_size )
            return f( room_for_selection );

        for( TI val = n ? room_for_selection[ n - 1 ] + 1 : 0; val < nb_available_items; ++val ) {
            room_for_selection[ n ] = val;
            _for_each_comb( selection_size, nb_available_items, room_for_selection, f, n + 1 );
        }
    }

} // namespace

//template<class TI>
//bool for_each_comb_cont( TI rese_size, TI list_size, const std::function<bool( const std::vector<TI> & )> &f, bool ordered = false ) {
//    if ( rese_size < list_size )
//        return true;
//    std::vector<TI> res( list_size );
//    return _for_each_comb_cont( rese_size, list_size, f, res, TI( 0 ), ordered );
//}

//template<class TI>
//bool for_each_comb_cont_with_init( TI *lst_data, TI lst_size, TI nb_items_to_take, const std::function<bool()> &f ) {
//    if ( rese_size < list_size )
//        return true;
//    while ( true ) {
//        // find the last value that can be changed
//        for( TI ind = list_size, max_value = rese_size; ; ) {
//            if ( ! ind-- )
//                return true;

//            if ( ++list_data[ ind ] < max_value-- ) {
//                while ( ++ind < list_size )
//                    list_data[ ind ] = list_data[ ind - 1 ] + 1;
//                if ( ! f() )
//                    return false;
//                break;
//            }
//        }
//    }
//}


template<class TI>
void for_each_comb( TI selection_size, TI nb_available_items, TI *room_for_selection, const std::function<void( TI * )> &f ) {
    if ( nb_available_items < selection_size )
        return;
    _for_each_comb<TI>( selection_size, nb_available_items, room_for_selection, f, 0 );
}
