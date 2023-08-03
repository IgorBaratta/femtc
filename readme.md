# Finite Element - Tensor Contraction Experiments

## Building



```
git clone --recurse-submodules git@github.com:IgorBaratta/femtc.git
mkdir build && cd build



cmake -DCMAKE_CXX_FLAGS="-march=native -Ofast" ..
```

## Using spack to manage dependencies
```
source spack/dir ..
source load.sh `spack_env`
```