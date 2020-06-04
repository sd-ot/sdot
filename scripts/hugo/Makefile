# PRG = ./bench_FP64_U64_AVX2 && ./bench_FP64_U64_SSE2
PRG = ./test_FP64_U64_AVX2 --gtest_filter=ZGrid_2_FP64_U64_AVX2*

all:
	cd ../build-sdot-Desktop-Debug && make -j8 && ${PRG}

