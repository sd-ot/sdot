NSMAKE = nsmake run
NSMAKE += -DPAREX_DIR=\"`pwd`\"
NSMAKE += -DSDOT_DIR=\"`pwd`\"
NSMAKE += --cpp-flag -std=c++17
NSMAKE += -Isrc
NSMAKE += -g3

all:
	# ${NSMAKE} tests/test_simd.cpp
	# ${NSMAKE} tests/test_parex.cpp
	# ${NSMAKE} tests/test_SetOfElementaryPolytops.cpp
	${NSMAKE} tests/test_PowerDiagram.cpp

c:
	rm objects/lib/* objects/*.info
	rm -rf /tmp/parex_*
