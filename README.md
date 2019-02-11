PowerDiagram with sphere/disc intersections...

See the `tests` and `samples` directories for examples in C++.

For python, the bindings have first to be compiled. It can be done manually using `make lib_python3_manual`. For a more modern compilation toolchain, one can use nsmake as in the target `lib_python3` in `Makefile`.

Note: `PowerDiagram.add_..._shape` adds a shape in the integration space (it's cumulative).

