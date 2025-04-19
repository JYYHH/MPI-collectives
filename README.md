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
### (New feature) Weighted Scan
- `definition`: $S[i] = \sum_{k=0}^i sum[k] (\Pi_{l=k+1}^i mul[l])$
- `sequential handling`: 
```cpp
for(int i = 1; i < n; i++)
    S[i] = sum[i] + mul[i] * S[i - 1];
```
- Which is built for [Mamba](https://arxiv.org/pdf/2312.00752), that Scan