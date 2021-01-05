NSMAKE = nsmake run
NSMAKE += -DPAREX_DIR=\"`pwd`/ext/parex\"
# NSMAKE += -DASIMD_DIR=\"`pwd`/ext/asimd\"
NSMAKE += -DSDOT_DIR=\"`pwd`\"
NSMAKE += --cpp-flag -std=c++17
NSMAKE += -Iext/parex/src
# NSMAKE += -Iext/asimd/src
NSMAKE += -Isrc
NSMAKE += -g3

all:
	# ${NSMAKE} tests/test_ElementaryPolytopTypeSet.cpp
	${NSMAKE} tests/test_SetOfElementaryPolytops.cpp

mrf:
	${NSMAKE} tests/test_PowerDiagram.cpp
	${NSMAKE} tests/test_radix_sort.cpp
	${NSMAKE} tests/test_parex.cpp

c:
	rm -rf .generated_libs
	rm -rf /tmp/parex_*
