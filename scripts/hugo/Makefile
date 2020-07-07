NSMAKE = time nsmake -g3 -O3 # --cxx clang++ --cpp-flag  "-std=c++17"
# Dg676-yu

all:
	# nsmake run -g3 --exec-with "valgrind --leak-check=full" tests/test_VolumeAssembly.cpp
	# ${NSMAKE} run src/sdot/ConvexPolytop/internal/generation/test_ConvexHull.cpp
	# ${NSMAKE} tests/test_Simd.cpp
	# ${NSMAKE} tests/test_StructOfArrays.cpp
	# ${NSMAKE} run tests/bench_ConvexPolytopSet.cpp
	# ${NSMAKE} run tests/test_IntrusiveList.cpp
	# ${NSMAKE} run tests/test_RecursivePolytop.cpp
	# ${NSMAKE} run src/sdot/geometry/internal/generation/gen_cuts.cpp
	${NSMAKE} run tests/test_SetOfElementaryPolytops.cpp
	# ${NSMAKE} run --exec-with "valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes" tests/bench_SetOfElementaryPolytops.cpp
	${NSMAKE} run tests/bench_SetOfElementaryPolytops.cpp

gen:
	${NSMAKE} run src/sdot/ConvexPolytop/internal/generation/gen_cuts.cpp

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
