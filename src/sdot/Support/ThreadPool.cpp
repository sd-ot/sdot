#include "ThreadPool.h"

//// nsmake lib_name pthread
ThreadPool thread_pool;


void ThreadPool::init( int nb_threads ) {
    _init_if_not_done( nb_threads );
}

int ThreadPool::nb_threads() {
    _init_if_not_done();
    return _nb_threads;
}

void ThreadPool::_init_if_not_done( int nb_threads ) {
    if ( _nb_threads > 0 )
        return;

    //  nb threads to launch
    _nb_threads = nb_threads <= 0 ?
        std::thread::hardware_concurrency() :
        nb_threads;

    // _nb_threads = 1;
}

