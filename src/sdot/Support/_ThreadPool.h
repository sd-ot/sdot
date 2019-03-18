#pragma once

#include <condition_variable>
#include <functional>
#include <thread>
#include <vector>
#include <deque>
#include <mutex>

/**
  TODO: use atomics
*/
class ThreadPool {
public:
    using TF = std::function<void( size_t job_index, int num_thread )>;

    ThreadPool();
    ~ThreadPool();

    /// call to init is not mandatory (checked if done during execute anymway) but is a way to define explicitly nb_threads
    void                    init              ( int nb_threads = 0 );

    /// nb_jobs should be > something like 4*nb_thread, but not too high, too avoid overhead. f( job_index, num_thread )
    void                    execute           ( size_t nb_jobs, const TF &f );

    ///
    int                     nb_threads        ();

private:
    struct ThreadData {
        std::size_t beg;
        std::size_t end;
        std::mutex  m;
    };

    void                    _init_if_not_done( int nb_threads = 0 );
    void                    _execute_loc     ( int num_thread );
    void                    _clear           ();

    size_t                  nb_waiting_threads;
    std::deque<ThreadData>  thread_data;
    bool                    initialized;
    bool                    end_threads;
    TF                      cur_f;
    std::condition_variable cbw;             ///< wait for beginning of work
    std::condition_variable cew;             ///< wait for end of work
    std::mutex              m;
};

extern ThreadPool thread_pool;
