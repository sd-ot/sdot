all: src/sdot/kernels/VecTF.h src/sdot/kernels/VecTF.cpp src/sdot/kernels/VecTI.h src/sdot/kernels/VecTI.cpp
	nsmake run -g3 src/sdot/generation/generation_geometry.cpp
	nsmake run -g3 tests/test_SetOfElementaryPolytops.cpp

src/sdot/kernels/Vec%.h: src/sdot/kernels/gen/make_VecTx.py src/sdot/kernels/gen/VecTT.h
	python $^ $@

src/sdot/kernels/Vec%.cpp: src/sdot/kernels/gen/make_VecTx.py src/sdot/kernels/gen/VecTT.cpp
	python $^ $@
