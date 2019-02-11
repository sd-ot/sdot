CXX = g++
BLD = build

all: # py_test # coeffs
	# nsmake run -g3 benchmarks/bench_solve_2D.cpp --distribution eur://data/fof_boxlen656-25_n256_lcdmw7_00007_cube_00000 --nb-diracs 500 --max-dirac-per-cell 60 --vtk-output vtk/pd
	# nsmake run -g3 benchmarks/bench_solve_2D.cpp --distribution data/SIMpositionsz_0_cropped.xyz --nb-diracs 500 --max-dirac-per-cell 60 --vtk-output vtk/pd
	# nsmake run -g3 benchmarks/bench_solve_2D.cpp --distribution concentration --nb-diracs 1000 --max-dirac-per-cell 60 --vtk-output vtk/pd
	# python3 benchmarks/bench_solve_2D.py --distribution concentration --nb-diracs 500 --max-dirac-per-cell 60 --vtk-output vtk/pd

	# nsmake run -g3 tests/test_ConvexPolyhedron2.cpp
	nsmake run -g3 tests/test_ZGrid.cpp
	# nsmake run -g3 tests/test_get_der_measures.cpp
	# nsmake run -g3 tests/test_IntSpiral.cpp
	# nsmake run -g3 benchmarks/bench_solve_2D.cpp --distribution random --nb-diracs 11 --max-dirac-per-cell 200 --vtk-output vtk/pd
	# nsmake gtest -O3 -g3 tests/test_LaguerreCell3.cpp
	# nsmake gtest -g3 tests/test_SpRegularGrid.cpp
	# nsmake gtest -g3 tests/test_PowerDiagram.cpp
	# nsmake gtest -g3 tests/test_RadixSort.cpp
	# nsmake run -g3 benchmarks/bench_traversal.cpp random 1000 pd.vtk
	# nsmake run -g3 benchmarks/bench_traversal.cpp random 2000000 => 2.90
	# nsmake run -g3 --exec-with valgrind benchmarks/bench_traversal_2D.cpp random 100 pd.vtk
	# nsmake run -g3 --exec-with time benchmarks/bench_solve_2D.cpp random 101 pd.vtk
	# nsmake run -g3 benchmarks/bench_traversal_2D.cpp lines 38 2000000
	# nsmake run -g3 benchmarks/bench_traversal_2D.cpp random 100 pd.vtk
	# nsmake run --exec-with time examples/converging_corridor.cpp
	# nsmake run -g3 --exec-with "valgrind --tool=callgrind --dump-instr=yes" benchmarks/bench_traversal_2D.cpp random 10000
	# sudo operf -g /home/leclerc/.nsmake/build/bench_traversal.exe random 1000000

	# nsmake run -DBUILD_WITH_EASY_PROFILER examples/bench_TraversalByZIndexGrid.cpp
	# nsmake gtest -g3 tests/test_PointGrid.cpp
	# nsmake gtest -g3 tests/test_IntSpiral.cpp
	# nsmake gtest -g3 tests/test_ThreadPool.cpp
	# nsmake run -g3 samples/gravity_2D.cpp
	# nsmake run -g3 src/PowerDiagram/offline_integration/gen_approx_integration.cpp --function Gaussian --end-log-scale 10 --precision 1e-10
	# nsmake run -g3 src/PowerDiagram/offline_integration/gen_approx_integration.cpp --function Unit --end-log-scale 10 --precision 1e-10 --centroid

bench_cgal:
	g++ -c -o scalability_2D_cgal.o -march=native -O3 -lCGAL examples/scalability_2D_cgal.cpp
	g++ -o scalability_2D_cgal.exe scalability_2D_cgal.o -lCGAL -lgmp
	time ./scalability_2D_cgal.exe sin 10e6

coeffs:
	nsmake run src/PowerDiagram/offline_integration/exp_mr2.cpp

py_test: 
	PYTHONPATH=`pwd`/python python3 python/test_PowerDiagram.py

lib_python3:
	nsmake lib `python3 -m pybind11 --includes` -o python/power_diagram`python3-config --extension-suffix` src/PowerDiagram/pybind.cpp

lib_python3_manual:
	mkdir -p build
	${CXX} -o ${BLD}/pybind.o -c src/PowerDiagram/pybind.cpp -Ipybind11/include -march=native -Wattributes -O3 -g3 -Iext/pybind11/include `python3-config --includes` -fpic
	${CXX} -o ${BLD}/Assert-3.o -c src/PowerDiagram/system/Assert.cpp -Ipybind11/include  `python3-config --includes` -fpic
	${CXX} -o ${BLD}/ThreadPool.o -c src/PowerDiagram/system/ThreadPool.cpp -Ipybind11/include `python3-config --includes` -fpic
	${CXX} -o ${BLD}/IntSpiral.o -c src/PowerDiagram/IntSpiral.cpp `python3-config --includes` -Ipybind11/include -fpic
	${CXX} -shared -o python/power_diagram`python3-config --extension-suffix` ${BLD}/pybind.o ${BLD}/ThreadPool.o ${BLD}/IntSpiral.o ${BLD}/Assert-3.o -lpthread
# was `python3 -m pybind11 --includes`

.PHONY: lib_python cpp_test py_test
