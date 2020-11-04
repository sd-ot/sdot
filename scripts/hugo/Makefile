all: src/sdot/kernels/VecTF.h src/sdot/kernels/VecTF.cpp src/sdot/kernels/VecTI.h src/sdot/kernels/VecTI.cpp # gen
	# nsmake run -g3 tests/test_SetOfElementaryPolytops.cpp
	# nsmake run -g3 tests/test_PowerDiagram.cpp
	nsmake run -g3 tests/test_ParEx.cpp

gen: 
	nsmake run -g3 src/sdot/generation/generation_geometry.cpp

src/sdot/kernels/Vec%.h: src/sdot/kernels/gen/make_VecTx.py src/sdot/kernels/gen/VecTT.h
	python $^ $@

src/sdot/kernels/Vec%.cpp: src/sdot/kernels/gen/make_VecTx.py src/sdot/kernels/gen/VecTT.cpp
	python $^ $@
