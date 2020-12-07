#ifndef PAREX_BumpPointerPool_HEADER
#define PAREX_BumpPointerPool_HEADER

#include <cstdint>
#include <utility>

namespace parex {

/**
  Simple object pool.

  If object not "is_trivially_destructible", it is stored internally in a linked list of objects with virtual destructors.

  Beware: allocation is NOT thread_safe
*/
class BumpPointerPool {
public:
    /* */    BumpPointerPool();
    /* */   ~BumpPointerPool();

    char*    allocate       ( std::size_t size, std::size_t alig );

    template                <class T,class... Args>
    T*       create         ( Args &&...args );

private:
    struct   Frame          { Frame *prev_frame; char content[ 8 ]; };
    struct   Item           { virtual ~Item() {} Item *prev; };
    union    Ptr            { char *cp; std::size_t vp; };

    template <class T>
    struct   Inst : Item    { template<class... Args> Inst( Args &&...args ) : object( std::forward<Args>( args )... ) {} virtual ~Inst() {} T object; };

    Ptr      current_ptr;
    char*    ending_ptr;
    Frame*   last_frame;
    Item*    last_item;
};

} // namespace parex

#include "BumpPointerPool.tcc"

#endif // PAREX_BumpPointerPool_HEADER

