#include "ThreadPool.h"
#include <ostream>

//// nsmake lib_name pthread

ThreadPool thread_pool;

ThreadPool::ThreadPool() : initialized( false ), end_threads( false ) {
}

ThreadPool::~ThreadPool() {
    _clear();
}

void ThreadPool::init( int nb_threads ) {
    _clear();
    _init_if_not_done( nb_threads );
}

void ThreadPool::execute( size_t nb_jobs, const TF &f ) {
    _init_if_not_done();
    this->cur_f = f;

    // assign tasks to each thread
    for( size_t nt = 0; nt < thread_data.size(); ++nt ) {
        thread_data[ nt ].beg = ( nt + 0 ) * nb_jobs / thread_data.size();
        thread_data[ nt ].end = ( nt + 1 ) * nb_jobs / thread_data.size();
    }

    // execute f locally, and remotely
    nb_waiting_threads = thread_data.size();
    cbw.notify_all();
    _execute_loc( 0 );

    // wait for all tasks to be done
    std::unique_lock<std::mutex> lk( m );
    if ( nb_waiting_threads )
        cew.wait( lk, [&]() { return nb_waiting_threads == 0; } );

    // clean
    cur_f = {};
}

int ThreadPool::nb_threads() {
    _init_if_not_done();
    return thread_data.size();
}

void ThreadPool::_init_if_not_done( int _nb_threads ) {
    if ( initialized )
        return;
    initialized = true;

    int nb_threads = _nb_threads ? _nb_threads : std::thread::hardware_concurrency();
    thread_data.resize( nb_threads );

    nb_waiting_threads = nb_threads - 1;
    for( int i = 1; i < nb_threads; ++i ) {
        std::thread t( [this]( size_t i ) {
            // say that this thread is started
            std::unique_lock<std::mutex> lk( m );
            if ( --nb_waiting_threads == 0 ) {
                m.unlock();
                cew.notify_one();
            } else
                m.unlock();

            // wait for some work
            while ( true ) {
                m.lock();
                auto &td = thread_data[ i ];
                if ( td.beg == td.end && end_threads == false )
                    cbw.wait( lk );
                m.unlock();

                if ( end_threads ) {
                    std::unique_lock<std::mutex> lk( m );
                    if ( --nb_waiting_threads == 0 )
                        cew.notify_one();
                    break;
                }
                _execute_loc( i );
            }
        }, i );
        t.detach();
    }

    // wait for threads to be started
    std::unique_lock<std::mutex> lk( m );
    cew.wait( lk, [this]{ return nb_waiting_threads == 0; } );
}

void ThreadPool::_execute_loc( int num_thread ) {
    // take one item at a time, in the local list
    while ( true ) {
        ThreadData &td = thread_data[ num_thread ];
        std::unique_lock<std::mutex> lk( td.m );
        if ( td.beg == td.end )
            break;
        size_t ind = td.beg++;
        lk.unlock();

        cur_f( ind, num_thread );
    }

    // is it possible to steal an item ?
    while ( true ) {
        bool found_something_to_steal = false;
        for( int alt_num_thread = 0; alt_num_thread < (int)thread_data.size(); ++alt_num_thread ) {
            if ( alt_num_thread != num_thread ) {
                ThreadData &alt_td = thread_data[ alt_num_thread ];
                std::unique_lock<std::mutex> lk( alt_td.m );
                if ( alt_td.beg == alt_td.end )
                    continue;
                found_something_to_steal = true;
                size_t ind = --alt_td.end;
                lk.unlock();

                cur_f( ind, num_thread );
            }
        }
        if ( found_something_to_steal == false )
            break;
    }

    // the last thread notifies thread 0 to say that it is ended
    std::unique_lock<std::mutex> lk( m );
    if ( --nb_waiting_threads == 0 && num_thread )
        cew.notify_one();
}

void ThreadPool::_clear() {
    if ( initialized ) {
        nb_waiting_threads = thread_data.size() - 1;
        end_threads = true;
        cbw.notify_all();

        std::unique_lock<std::mutex> lk( m );
        cew.wait( lk, [this]{ return nb_waiting_threads == 0; } );

        initialized = false;
        end_threads = false;
        thread_data.clear();
    }
}
