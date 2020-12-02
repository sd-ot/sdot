#pragma once

#ifdef __CUDACC__
#define HaD __host__ __device__
#else
#define HaD
#endif
