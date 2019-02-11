#pragma once

#include "StaticRange.h"
#include "ThreadPool.h"
#include <array>

// #include <tbb/tbb.h>
// // nsmake lib_name tbb

/**
 * radix_sort on 1 thread
*/
template<class TI,int nb_bytes>
TI *radix_sort( TI *out, TI *inp, std::size_t len, N<nb_bytes>, std::vector<std::array<std::size_t,256>> &tmps ) {
    std::size_t nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    tmps.resize( nb_jobs );

    StaticRange<nb_bytes>::for_each( [&]( auto n ) {
        constexpr int shift = 8 * n.val;
        
        // get count for each bucket
        thread_pool.execute( nb_threads, [&]( std::size_t num_job, int ) {
            std::array<std::size_t,256> &tmp = tmps[ num_job ];
            for( std::size_t i = 0; i < 256; ++i )
                tmp[ i ] = 0;
            std::size_t beg = ( num_job + 0 ) * len / nb_jobs;
            std::size_t end = ( num_job + 1 ) * len / nb_jobs;
            for( std::size_t i = beg; i < end; ++i ) {
                auto s = inp[ i ] >> shift;
                if ( n != nb_bytes - 1 )
                    s &= 0xFF;
                ++tmp[ s ];
            }
        } );

        // suffix sum
        for( std::size_t i = 0, acc = 0; i < 256; ++i ) {
            for( size_t num_job = 0; num_job < nb_jobs; ++num_job ) {
                std::array<std::size_t,256> &tmp = tmps[ num_job ];
                std::size_t v = acc;
                acc += tmp[ i ];
                tmp[ i ] = v;
            }
        }

        // save
        thread_pool.execute( nb_threads, [&]( std::size_t num_job, int ) {
            std::array<std::size_t,256> &tmp = tmps[ num_job ];
            std::size_t beg = ( num_job + 0 ) * len / nb_jobs;
            std::size_t end = ( num_job + 1 ) * len / nb_jobs;
            for( std::size_t i = beg; i < end; ++i ) {
                auto s = inp[ i ] >> shift;
                if ( n != nb_bytes - 1 )
                    s &= 0xFF;
                out[ tmp[ s ]++ ] = inp[ i ];
            }
        } );
        
        std::swap( out, inp );
    } );
    
    return inp;
}

template<class TI,int nb_bytes>
TI *radix_sort( TI *out, TI *inp, std::size_t len, N<nb_bytes> n ) {
    std::vector<std::array<std::size_t,256>> tmps;
    return radix_sort( out, inp, len, n, tmps );
}
