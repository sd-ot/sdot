#pragma once

#include "Stream.h"
#include <functional>

/**
  Allows to use mpi semantic even if mpi is not loaded.

  Use #include "Mpi.h" and mpi.init( argc, argv ) to use MPI
*/
class Mpi {
public:
    bool                  master                 () const { return rank() == 0; }
    virtual int           rank                   () const = 0;
    virtual int           size                   () const = 0;

    virtual void          send                   ( const std::  int8_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const std:: uint8_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const std:: int32_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const std::uint32_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const std:: int64_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const std::uint64_t *data, std::size_t count, int destination, int tag = 0 ) = 0;
    virtual void          send                   ( const double        *data, std::size_t count, int destination, int tag = 0 ) = 0;

    virtual void          recv                   ( std::  int8_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( std:: uint8_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( std:: int32_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( std::uint32_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( std:: int64_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( std::uint64_t *data, std::size_t count, int source, int tag = 0 ) = 0;
    virtual void          recv                   ( double        *data, std::size_t count, int source, int tag = 0 ) = 0;

    virtual void          gather                 ( std:: int32_t *dst, const std:: int32_t *src, std::size_t count, int root = 0 ) = 0;
    virtual void          gather                 ( std:: int64_t *dst, const std:: int64_t *src, std::size_t count, int root = 0 ) = 0;
    virtual void          gather                 ( std::uint32_t *dst, const std::uint32_t *src, std::size_t count, int root = 0 ) = 0;
    virtual void          gather                 ( std::uint64_t *dst, const std::uint64_t *src, std::size_t count, int root = 0 ) = 0;
    virtual void          gather                 ( double        *dst, const double        *src, std::size_t count, int root = 0 ) = 0;

    virtual void          all_gather             ( std::vector<std::vector<char>> &dst, const char *src, std::size_t count ) = 0;
    virtual void          all_gather             ( std::vector<std::vector<int >> &dst, const int  *src, std::size_t count ) = 0;

    virtual void          bcast                  ( std::  int8_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( std:: uint8_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( std:: int32_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( std:: int64_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( std::uint32_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( std::uint64_t *vec, std::size_t count, int root = 0 ) = 0;
    virtual void          bcast                  ( double        *vec, std::size_t count, int root = 0 ) = 0;

    virtual void          barrier                () = 0;

    virtual void          selective_send_and_recv( std::vector<std::vector<char>> &ext, const std::vector<std::vector<int>> &needs, std::vector<char> &to_send ) = 0;

    virtual void          cross_sends            ( std::vector<std::vector<std::uint8_t >> &dst, const std::vector<std::vector<std::uint8_t >> &src ) = 0;
    virtual void          cross_sends            ( std::vector<std::vector<std::uint32_t>> &dst, const std::vector<std::vector<std::uint32_t>> &src ) = 0;
    virtual void          cross_sends            ( std::vector<std::vector<std::uint64_t>> &dst, const std::vector<std::vector<std::uint64_t>> &src ) = 0;

    virtual std::size_t   probe_size             ( int source, int tag = 0 ) = 0;

    virtual std:: int32_t reduction              ( std:: int32_t value, const std::function<std:: int32_t(std:: int32_t,std:: int32_t)> &f ) = 0;
    virtual std:: int64_t reduction              ( std:: int64_t value, const std::function<std:: int64_t(std:: int64_t,std:: int64_t)> &f ) = 0;
    virtual std::uint32_t reduction              ( std::uint32_t value, const std::function<std::uint32_t(std::uint32_t,std::uint32_t)> &f ) = 0;
    virtual std::uint64_t reduction              ( std::uint64_t value, const std::function<std::uint64_t(std::uint64_t,std::uint64_t)> &f ) = 0;
    virtual double        reduction              ( double        value, const std::function<double       (double       ,double       )> &f ) = 0;

    //    virtual void          partition  ( std::vector<int> &partition, const std::vector<std::size_t> &node_offsets, const std::vector<std::size_t> &edge_indices, const std::vector<std::size_t> &edge_values, const std::vector<int> &edge_costs, const std::vector<double> &xyz, int dim, bool full_redistribution ) = 0;
    //    virtual void          partition  ( std::vector<int> &partition, const std::vector<std::size_t> &node_offsets, const std::vector<double> &xyz, int dim ) = 0;
};

extern Mpi *mpi;

template<class OS,class... Args> void ___my_mpi_print( OS &os, const char *str, const Args &...args ) {
    // get local msg
    std::ostringstream ss;
    __my_print( ss, str, ", ", args... );
    std::string msg = ss.str();

    // send data (if rank != 0) or display it
    if ( mpi->rank() ) {
        mpi->send( (const std::int8_t *)msg.data(), msg.size(), 0, mpi->rank() );
    } else {
        // message for rank 0
        if ( mpi->size() )
            os << "rank 0: ";
        os << msg;

        // other messages
        for( int rank = 1; rank < mpi->size(); ++rank ) {
            std::string str( mpi->probe_size( rank, rank ), ' ' );
            mpi->recv( (std::int8_t *)str.data(), str.size(), rank, rank );
            os << "rank " << rank << ": " << str;
        }
    }

    // we don't want other displays to interfere
    mpi->barrier();
}

template<class OS,class... Args> void ___my_mpi_print_0( OS &os, const char *str, const Args &...args ) {
    // get local msg
    std::ostringstream ss;
    __my_print( ss, str, ", ", args... );

    if ( mpi->rank() == 0 )
        os << ss.str();
}

#define PNMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " ->\n" , __VA_ARGS__ )
#define PMPI( ... ) ___my_mpi_print( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
#define PMPI_0( ... ) ___my_mpi_print_0( std::cout, #__VA_ARGS__ " -> " , __VA_ARGS__ )
