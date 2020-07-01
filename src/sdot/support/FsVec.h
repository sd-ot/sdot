#ifndef FS_VEC_H
#define FS_VEC_H

#include <cstdint>

/**
  Fixed size vector: size is set during construction, there's no dtor
*/
template<class T,class ST=std::size_t>
class FsVec {
public:
    template<class Pool> FsVec     ( Pool &pool, ST size, const T *content );
    template<class Pool> FsVec     ( Pool &pool, ST size, ST alig );
    template<class Pool> FsVec     ( Pool &pool, ST size );
    /**/                 FsVec     ( T *data, ST size );
    /**/                 FsVec     ();

    const T&             operator[]( ST index ) const;
    T&                   operator[]( ST index );

    const T*             begin     () const;
    T*                   begin     ();

    const T*             data      () const;
    T*                   data      ();

    const T*             end       () const;
    T*                   end       ();

    const T&             front     () const;
    T&                   front     ();

    ST                   size      () const;

    void                 downsize  ( ST size );
    void                 clear     () { _size = 0; }

private:
    T*                   _data;
    ST                   _size;
};

#include "FsVec.tcc"

#endif // FS_VEC_H

