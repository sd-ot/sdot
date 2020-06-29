NSMAKE = nsmake -g3 # --exec-with valgrind --cxx clang++ 
# Dg676-yu

all:
	# nsmake run -g3 --exec-with "valgrind --leak-check=full" tests/test_VolumeAssembly.cpp
	# ${NSMAKE} src/sdot/volume_assembly/generation/test_ConvexHull.cpp
	# ${NSMAKE} tests/test_Simd.cpp
	# ${NSMAKE} tests/test_StructOfArrays.cpp
	# ${NSMAKE} run tests/bench_ConvexPolytopSet.cpp
	${NSMAKE} run tests/test_ConvexPolytopSet.cpp

gen:
	${NSMAKE} run src/sdot/ConvexPolytop/internal/generation/gen_cuts.cpp

bench:
	${NSMAKE} run tests/bench_ConvexPolytopSet.cpp

bld:
	${NSMAKE} exe tests/bench_ConvexPolytopSet.cpp

tst:
	${NSMAKE} run tests/test_ConvexPolytopSet.cpp

asm:
	# create assembler code:
	g++ -S -fverbose-asm -g3 -O3 -march=native tests/bench_ConvexPolytopSet.cpp -o tests/bench_ConvexPolytopSet.s
	# create asm interlaced with source lines:
	as -alhnd tests/bench_ConvexPolytopSet.s > tests/bench_ConvexPolytopSet.lst
