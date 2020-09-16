NSMAKE = nsmake # --cpp-flag  "-std=c++17"
# Dg676-yu

all:
	# nsmake run -g3 --exec-with "valgrind --leak-check=full" tests/test_VolumeAssembly.cpp
	# ${NSMAKE} run tests/test_SimplexSumPolytop.cpp
	# ${NSMAKE} run tests/test_RecursivePolytopConnectivity.cpp
	# ${NSMAKE} run tests/test_RecursiveConvexPolytop.cpp
	# ${NSMAKE} run tests/test_SetOfElementaryPolytops.cpp
	${NSMAKE} run tests/prel_bench/bench_case_dispatch.cpp

gen:
	${NSMAKE} -O3 run src/sdot/geometry/internal/generation/gen_cuts.cpp 2; ${NSMAKE} run -O3 src/sdot/geometry/internal/generation/gen_cuts.cpp 3

bench:
	${NSMAKE} run tests/bench_ConvexPolytopSet.cpp

bld:
	${NSMAKE} exe src/sdot/geometry/internal/codegen/gen_cuts.cpp

tst:
	${NSMAKE} run tests/test_ConvexPolytopSet.cpp

asm:
	# create assembler code:
	g++ -S -fverbose-asm -g3 -O3 -march=native tests/bench_ConvexPolytopSet.cpp -o tests/bench_ConvexPolytopSet.s
	# create asm interlaced with source lines:
	as -alhnd tests/bench_ConvexPolytopSet.s > tests/bench_ConvexPolytopSet.lst
