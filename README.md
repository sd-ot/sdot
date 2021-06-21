PowerDiagram with sphere/disc intersections...

Note: this code is being deeply reorganized. In the meantime, do not hesitate to contact us, e.g. if you're looking for recent benchmarks or more specific stuff (out-of-core, GPU, ...).

See the `tests` and `samples` directories for examples in C++.

For python, the bindings have first to be compiled. It can be done manually using `make lib_python3_manual`. For a more modern compilation toolchain, one can use nsmake as in the target `lib_python3` in `Makefile`.

Note: `PowerDiagram.add_..._shape` adds a shape in the integration space (it's cumulative).

