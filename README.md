## Compile
```bash
mpicc test_module.c HyperCubeLib.c  -o run
```
## Run
```bash
mpirun -n 64 run
```
## More
### Why I did this?
- Inspired by CS525 at Purdue, [here](./Basic_Communication_Operations.pdf) is the related slide.
### Complexity
- see it at the top of each function [here](./HyperCubeLib.c)
### Different implementations
- see the MACROs [here](./HyperCubeLib.h), and you could compare the complexities among different implementations.