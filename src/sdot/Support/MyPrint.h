#pragma once

#include <sstream>
#include <mutex>

struct WithSep {
    const char *sep;
};

template<class OS,class T0>               void __my_print( OS &os, const char *curr_sep, const char *next_sep, const T0      &t0                      ) { os << curr_sep << t0 << std::endl; }
template<class OS,class T0,class... Args> void __my_print( OS &os, const char *curr_sep, const char *next_sep, const T0      &t0, const Args &...args ) { os << curr_sep << t0; __my_print( os, next_sep, next_sep, args... ); }
template<class OS,         class... Args> void __my_print( OS &os, const char *curr_sep, const char *next_sep, const WithSep &ws, const Args &...args ) { __my_print( os, "", ws.sep, args... ); }

template<class OS,class... Args> void ___my_print( OS &os, const char *str, const Args &...args ) {
    static std::mutex m;
    m.lock();
    __my_print( os, str, ", ", args... );
    os.flush();
    m.unlock();
}
