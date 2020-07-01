#ifndef SDOT_BumpPointerPool_H
#define SDOT_BumpPointerPool_H

#include <cstdint>
#include <utility>

/**
  Simple object pool
*/
class BumpPointerPool {
public:
    /* */    BumpPointerPool( BumpPointerPool &&that );
    /* */    BumpPointerPool( const BumpPointerPool &that ) = delete;
    /* */    BumpPointerPool();
    /* */   ~BumpPointerPool();

    void     operator=      ( const BumpPointerPool &that ) = delete;

    char*    allocate       ( std::size_t size, std::size_t alig );
    char*    allocate       ( std::size_t size );

    template                <class T,class... Args>
    T*       create         ( Args &&...args );

    void     clear          ();
    void     free           ();

private:
    struct   Frame          { Frame *prev_frame; char *ending_ptr; char content[ 8 ]; };
    struct   Item           { virtual ~Item() {} Item *prev; };
    union    Ptr            { char *cp; std::size_t vp; };

    template <class T>
    struct   Inst : Item    { template<class... Args> Inst( Args &&...args ) : object{ std::forward<Args>( args )... } {} virtual ~Inst() {} T object; };

    Ptr      current_ptr;
    char*    ending_ptr;
    Frame*   last_frame;
    Item*    last_item;
};

#include "BumpPointerPool.tcc"

#endif // SDOT_BumpPointerPool_H

