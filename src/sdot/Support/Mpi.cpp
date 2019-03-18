#include "Mpi.h"
#include <string.h>

//
class NoMpi : public Mpi {
public:
    virtual int           rank                   () const override { return 0; }
    virtual int           size                   () const override { return 1; }

    virtual void          send                   ( const std::  int8_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const std:: uint8_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const std:: int32_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const std::uint32_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const std:: int64_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const std::uint64_t *data, std::size_t count, int destination, int tag = 0 ) override {}
    virtual void          send                   ( const double        *data, std::size_t count, int destination, int tag = 0 ) override {}

    virtual void          recv                   ( std::  int8_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( std:: uint8_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( std:: int32_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( std::uint32_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( std:: int64_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( std::uint64_t *data, std::size_t count, int source, int tag = 0 ) override {}
    virtual void          recv                   ( double        *data, std::size_t count, int source, int tag = 0 ) override {}

    virtual void          gather                 ( std:: int32_t *dst, const std:: int32_t *src, std::size_t count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( std:: int64_t *dst, const std:: int64_t *src, std::size_t count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( std::uint32_t *dst, const std::uint32_t *src, std::size_t count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( std::uint64_t *dst, const std::uint64_t *src, std::size_t count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }
    virtual void          gather                 ( double        *dst, const double        *src, std::size_t count, int root = 0 ) override { memcpy( dst, src, count * sizeof( *dst ) ); }

    virtual void          all_gather             ( std::vector<std::vector<char>> &dst, const char *src, std::size_t count ) override { dst.resize( 1 ); }
    virtual void          all_gather             ( std::vector<std::vector<int >> &dst, const int  *src, std::size_t count ) override { dst.resize( 1 ); }

    virtual void          bcast                  ( std::  int8_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( std:: uint8_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( std:: int32_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( std:: int64_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( std::uint32_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( std::uint64_t *vec, std::size_t count, int root = 0 ) override {}
    virtual void          bcast                  ( double        *vec, std::size_t count, int root = 0 ) override {}

    virtual std::size_t   probe_size             ( int source, int tag = 0 ) override { return 0; }

    virtual void          selective_send_and_recv( std::vector<std::vector<char>> &ext, const std::vector<std::vector<int>> &needs, std::vector<char> &to_send ) override {}

    virtual void          cross_sends            ( std::vector<std::vector<std::uint8_t >> &dst, const std::vector<std::vector<std::uint8_t >> &src ) override { dst[ 0 ] = src[ 0 ]; }
    virtual void          cross_sends            ( std::vector<std::vector<std::uint32_t>> &dst, const std::vector<std::vector<std::uint32_t>> &src ) override { dst[ 0 ] = src[ 0 ]; }
    virtual void          cross_sends            ( std::vector<std::vector<std::uint64_t>> &dst, const std::vector<std::vector<std::uint64_t>> &src ) override { dst[ 0 ] = src[ 0 ]; }

    virtual void          barrier                () override {}

    virtual std:: int32_t reduction              ( std:: int32_t value, const std::function<std:: int32_t(std:: int32_t,std:: int32_t)> &f ) override { return value; }
    virtual std:: int64_t reduction              ( std:: int64_t value, const std::function<std:: int64_t(std:: int64_t,std:: int64_t)> &f ) override { return value; }
    virtual std::uint32_t reduction              ( std::uint32_t value, const std::function<std::uint32_t(std::uint32_t,std::uint32_t)> &f ) override { return value; }
    virtual std::uint64_t reduction              ( std::uint64_t value, const std::function<std::uint64_t(std::uint64_t,std::uint64_t)> &f ) override { return value; }
    virtual double        reduction              ( double        value, const std::function<double       (double       ,double       )> &f ) override { return value; }

    //    virtual void          partition  ( std::vector<int> &partition, const std::vector<std::size_t> &node_off, const std::vector<std::size_t> &edge_indices, const std::vector<std::size_t> &edge_values, const std::vector<int> &edge_costs, const std::vector<double> &xyz, int dim, bool full_redistribution ) override { partition.resize( edge_indices.size() - 1 ); for( int &p : partition ) p = 0; }
    //    virtual void          partition  ( std::vector<int> &partition, const std::vector<std::size_t> &node_off, const std::vector<double> &xyz, int dim ) override { partition.resize( xyz.size() / dim ); }
};

static NoMpi inst_mpi_abstraction;
Mpi *mpi = &inst_mpi_abstraction;


