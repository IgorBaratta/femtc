# Finite Element - Tensor Contraction Experiments

## Building

```
git clone --recurse-submodules git@github.com:IgorBaratta/femtc.git
mkdir build && cd build

cmake -DCMAKE_CXX_FLAGS="-march=native -Ofast"
```