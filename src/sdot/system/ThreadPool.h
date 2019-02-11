#pragma once

#include <condition_variable>
#include <functional>
#include <atomic>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>

/**
  TODO: use atomics
*/
class ThreadPool {
public:
    int                     nb_threads       (); ///<
    template<class TF> void execute          ( size_t nb_jobs, const TF &f ); ///< nb_jobs should be > something like 4*nb_thread, but not too high, too avoid overhead
    void                    init             ( int nb_threads = 0 ); ///< call to init is not mandatory (checked during `execute` anymway) but is a way to define explicitly nb_threads

private:
    void                    _init_if_not_done( int nb_threads = 0 );
    int                     _nb_threads      = 0;
};

#include "ThreadPool.tcc"

extern ThreadPool thread_pool;
