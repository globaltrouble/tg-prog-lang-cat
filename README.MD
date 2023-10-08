# Telegram ML Competition 2023

### Boostrap dependencies:
- install python3
- install your favorite compiler (for example gcc/clang)
- install cmake 
See `docker/builder.dockerfile` example for Debian-10

### Build:
By default build artifacts will be inside build/${target_name} directory:
- library itself will be located at `build/libtglang/libtglang.so`
- test binary will be located at `build/tglang-tester/tglang-tester`

Examples:
```
#all commands must be run from project root directory

#build all targets
./build.py

#remove artifacts/cache/config from previous build and rebuild from scratch
./build.py --clean

#build just required target, for example libtglang
./build.py --target libtglang
```

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


### Docker build & run:
- Install docker
- Build container:
    ```
    #build dev container (for build,validate)
    docker build ./docker/ -f docker/builder.dockerfile -t tglang-builder

    #build test container (for validation before submit)
    docker build ./docker/ -f docker/tester.dockerfile -t tglang-tester
    ```
- Run build script from docker:
    ```
    #build and link binaries, then run tester on test file
    #use build script as an examples, 
    #but keep free to add your examples to `examples` dir in project root and run test on it.
    docker run -v .:/app -it tglang-builder --test-file /app/build.py
    ```
- Run tester in production like environment:
    ```
    #single file, original tg tester
    docker run -v ./build-docker/bin:/app/bin -v ./examples:/app/examples -it tglang-tester single /app/examples/build.py

    #multi file, modified tg tester accept directory as an argument and print stats about each file (name, tag, size, duration)
    #to profile timing, please use '--build-type Relase' argument for builder for maximum optimizations
    #please copy your files into examples dir inside this project before run
    docker run -v ./build-docker/bin:/app/bin -v ./examples:/app/examples -it tglang-tester multi /app/examples/
    ```


Keep in mind, you need to forward project root directory to container via `-v` argument.
Base command is `docker run -v .:/app -it tglang-builder`, after it will run build script with next arguments.
So you can work with container as with build script (pass same arguments).
To run shell in docker container instead of build script:
```
docker run --entrypoint bash -v .:/app -it tglang-builder
```