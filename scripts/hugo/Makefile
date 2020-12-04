NSMAKE = nsmake run
NSMAKE += -DPAREX_DIR=\"`pwd`/src/parex\"
NSMAKE += -DSDOT_DIR=\"`pwd`\"
NSMAKE += --cpp-flag -std=c++17
NSMAKE += -Isrc/parex/src
NSMAKE += -Isrc/asimd/src
NSMAKE += -Isrc
NSMAKE += -g3

all:
	# ${NSMAKE} tests/test_PowerDiagram.cpp
	# ${NSMAKE} tests/test_radix_sort.cpp
	# ${NSMAKE} tests/test_parex.cpp
	# ${NSMAKE} tests/parex/test_GeneratedLibrarySet.cpp
	# ${NSMAKE} tests/parex/test_CompiledTask.cpp
	# ${NSMAKE} tests/test_ElementaryPolytopInfo.cpp
	# ${NSMAKE} tests/test_SetOfElementaryPolytops.cpp
	# ${NSMAKE} tests/asimd/test_asimd.cpp
	# ${NSMAKE} tests/asimd/test_asimd_gpu.cpp
	# ${NSMAKE} tests/parex/test_starpu.cpp
	# ${NSMAKE} tests/parex/test_HwGraph.cpp
	# ${NSMAKE} tests/parex/test_gtensor.cpp
	# ${NSMAKE} tests/parex/test_gvector.cpp
	${NSMAKE} tests/parex/test_Scalar.cpp

c:
	rm -rf .generated_libs
	rm -rf /tmp/parex_*
