#include "../../src/sdot/support/Vec.h"
#include "../../src/sdot/support/P.h"
using TI = std::size_t;
using namespace sdot;

//// nsmake cxx_name nvcc
//// nsmake cpp_flag --expt-extended-lambda
//// nsmake cpp_flag -O3
//// nsmake cpp_flag --x
//// nsmake cpp_flag cu

//// nsmake cpp_flag -gencode=arch=compute_75,code=compute_75 

// ------------------
__global__
void fill_cases_global_kernel( int *offsets, int *indices, const int *ids, int nb_ids ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < nb_ids )
        indices[ atomicAdd( offsets + ids[ i ], 1 ) ] = i;
}

void fill_cases_global( int *offsets, int *indices, const int *ids, int nb_ids, int nb_threads ) {
    TI nb_groups = ( nb_ids + nb_threads - 1 ) / nb_threads;
    fill_cases_global_kernel<<<nb_groups,nb_threads>>>( offsets, indices, ids, nb_ids );
}

// ------------------
__global__
void fill_cases_block_kernel( int *offsets, int *indices, const int *ids, int nb_ids ) {
    constexpr std::size_t nb_cases = 32;
    __shared__ int local_offsets[ nb_cases ];
    for( int i = threadIdx.x; i < nb_cases; i += blockDim.x )
        local_offsets[ i ] = i * nb_ids; // collisions
    __syncthreads();

    for( int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb_ids; i += blockDim.x * gridDim.x )
        indices[ atomicAdd( local_offsets + ids[ i ], 1 ) ] = i;
}

void fill_cases_block( int *offsets, int *indices, const int *ids, int nb_ids, int nb_threads ) {
    fill_cases_block_kernel<<<128,nb_threads>>>( offsets, indices, ids, nb_ids );
}

// ------------------
template<class TF>
void test_func( const char *name, const TF &func, int nb_threads ) {
    using Arch = MachineArch::Gpu;
    std::size_t nb_cases = 32;

    // preliminaries
    Vec<int> case_ids_cpu( 1024 * 1024 * 10 );
    for( TI i = 0; i < case_ids_cpu.size(); ++i )
        case_ids_cpu[ i ] = rand() % nb_cases;

    Vec<int,Arch> case_ids = case_ids_cpu;

    Vec<int,Arch> case_indices( nb_cases * case_ids.size() );
    Vec<int,Arch> case_offsets( nb_cases );
    for( TI i = 0; i < nb_cases; ++i )
        case_offsets[ i ] = i * case_ids.size();

    // run
    cudaEvent_t start, stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );

    cudaEventRecord( start );
    func( case_offsets.ptr(), case_indices.ptr(), case_ids.ptr(), case_ids.size(), nb_threads );
    cudaEventRecord( stop );

    // display
    cudaEventSynchronize( stop );
    float milliseconds = 0;
    cudaEventElapsedTime( &milliseconds, start, stop );

    for( TI i = 0; i < nb_cases; ++i )
        case_offsets[ i ] -= i * case_ids.size();

    // for( TI i = 0; i < nb_cases; ++i )
    //     std::cout << " " << case_offsets[ i ];
    // std::cout << "\n";

    float bw = 2 * sizeof( int ) * case_ids.size() / milliseconds * 1e3 / 1e9;
    std::cout << name << " nt:" << std::setw( 3 ) << nb_threads << " ck:" << thrust::reduce( case_offsets.begin(), case_offsets.end() ) << " bw:" << bw << " GB/s\n";
}

#define TST( FUNC ) \
    test_func( #FUNC, FUNC, 512 ); \
    test_func( #FUNC, FUNC, 256 ); \
    test_func( #FUNC, FUNC, 128 ); \
    test_func( #FUNC, FUNC, 32  )

int main() {
    // Ref: 616 GB/s ou 448 ?? ROP = 64
    // https://www.guru3d.com/articles-pages/geforce-rtx-2080-super-review,4.html
    TST( fill_cases_global );
    TST( fill_cases_block );
}

