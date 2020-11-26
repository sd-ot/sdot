NSMAKE = nsmake run
NSMAKE += -DPAREX_DIR=\"`pwd`\"
NSMAKE += --cpp-flag -std=c++17
NSMAKE += -Isrc
NSMAKE += -g3

all:
	# ${NSMAKE} tests/test_CompiledSymbolMap.cpp
	${NSMAKE} tests/test_Value.cpp

c:
	rm -rf .generated_libs/
	rm -rf /tmp/parex_*
