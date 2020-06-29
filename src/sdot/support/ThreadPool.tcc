#include "ThreadPool.h"

template<class TF>
void ThreadPool::execute( std::size_t nb_jobs, const TF &f ) {
    struct DataPerThread {
        std::atomic<std::size_t> cur;
        char                     pd0[ 64 - sizeof( std::size_t ) ];
        std::size_t              end;
        char                     pd1[ 64 - sizeof( std::size_t ) ];
    };


    // initialization of intervals
    int nt = nb_threads();
    std::vector<DataPerThread> dpts( nt );
    for( int n = 0; n < nt; ++n ) {
        dpts[ n ].cur = ( n + 0 ) * nb_jobs / nt;
        dpts[ n ].end = ( n + 1 ) * nb_jobs / nt;
    }

    // function that will be called in each thread
    auto exec = [&]( int num_thread ) {
        // own job
        for( DataPerThread &dpt = dpts[ num_thread ]; ; ) {
            std::size_t cur = dpt.cur++;
            if ( cur >= dpt.end )
                break;
            f( cur, num_thread );
        }
        // look in the other threads if there's something to do
        for( int o = 1; o < nt; ++o ) {
            int t = ( num_thread + 1 ) % nt;
            DataPerThread &dpt = dpts[ t ];
            std::size_t cur = dpt.cur++;
            if ( cur < dpt.end )
                f( cur, num_thread );
        }
    };

    // launch
    std::vector<std::thread> threads;
    for( int n = 1; n < nt; ++n )
        threads.emplace_back( exec, n );
    exec( 0 );
    for( std::thread &th : threads )
        th.join();
}
