# Telegram ML Competition 2023

### Boostrap dependencies:
- install python3
- install your favorite compiler (for example gcc/clang)
- install cmake 

### Build:
By default build artifacts will be inside build/${target_name} directory:
- library itself will be located at `build/libtglang/libtglang.so`
- test binary will be located at `build/tglang-tester/tglang-tester`

Examples:
'''
#all commands must be run from project root directory

#build all targets
./build.py

#remove artifacts/cache/config from previous build and rebuild from scratch
./build.py --clean

#build just required target, for example libtglang
./build.py --target libtglang
'''

### Run & export:
Before run or export build all targets, see `build` section for details how to build and where to find artifacts.

Run example:
```
# must be run from project root, will detect build script language
./build/tglang-tester/tglang-tester build.py
```

For export:
- copy artifact `libtglang.so` from `libtglang` target to your library path
- add libtglang sources dir to your include path

