#include <parex/support/S.h>
#include <utility>

template<class T,class... Args>
T *New_brace( parex::S<T>, Args&& ...args ) {
    return new T{ std::forward<Args>( args )... };
}
