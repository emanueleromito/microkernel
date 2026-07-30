# microkernel — Workflow

## First Time Setup

```bash
git clone <repo-url>
cd microkernel
mkdir build && cd build
cmake ..
make -j$(nproc)
ctest
```


## Daily Workflow

```bash
cd build
make && ctest
```

That's it. Edit code, run `make && ctest`, repeat.


## When to re-run cmake

Only when you change the build configuration:

- Added or removed a `.cpp` / `.cu` file
- Edited `CMakeLists.txt`
- Added a new dependency

```bash
cd build
cmake ..
make && ctest
```

For normal code edits, just `make`.


## Running a single test

```bash
./tests/test_tensor              # run one binary directly
ctest -R TensorTest              # run tests matching a name
ctest --output-on-failure        # show details when something fails
```


## Clean rebuild

If something breaks or feels stale:

```bash
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)
```


## Adding a new source file

1. Create the file in `src/` or `kernels/`
2. Add it to `CMakeLists.txt`
3. Re-run `cmake ..` from `build/`
4. `make && ctest`


## Adding a new test file

1. Create `tests/test_something.cpp`
2. Add it to `CMakeLists.txt`
3. Re-run `cmake ..` from `build/`
4. `make && ctest`