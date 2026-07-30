# Code Review Skill — microkernel

Review C++ and CUDA code written by a beginner learning both languages through this project.
The goal is to teach, not just fix. Every issue found should explain *why* it matters.


## Review Checklist

### 1. Memory

- Raw `new`/`delete` used instead of RAII (std::vector, std::unique_ptr, std::shared_ptr)
- Memory leaks: allocated but never freed
- Dangling pointers or references to destroyed objects
- Buffer overflows: accessing past array bounds
- Uninitialized variables read before being set
- Unnecessary copies where a reference or move would work


### 2. Correctness

- Off-by-one errors in loops and index math
- Integer overflow in size/stride calculations (e.g. large tensor shapes)
- Signed/unsigned comparison warnings
- Narrowing conversions (double → float, size_t → int)
- Floating point comparison with `==` instead of tolerance
- Missing `const` on methods that don't modify state
- Missing `const&` on function parameters that shouldn't be copied


### 3. C++ Idioms

- Rule of five: if you define a destructor, you probably need copy/move constructors and assignment operators
- Prefer `std::vector` over raw arrays
- Prefer `size_t` for sizes and indices
- Use initializer lists in constructors
- Mark single-argument constructors `explicit`
- Use `#pragma once` or include guards
- Prefer `nullptr` over `NULL` or `0`
- Use `auto` where the type is obvious from context
- Avoid `using namespace std` in headers


### 4. Performance

- Unnecessary copies (passing large objects by value)
- Repeated allocation inside loops
- Cache-unfriendly access patterns (column-major iteration over row-major data)
- `std::endl` vs `'\n'` (endl flushes the buffer, almost never needed)
- Reserve vector capacity when size is known ahead of time


### 5. CUDA Specific (when applicable)

- Missing error checking on CUDA API calls
- Kernel launch with wrong grid/block dimensions
- Uncoalesced global memory access
- Shared memory bank conflicts
- Missing `__syncthreads()` where needed
- Host/device pointer confusion
- Forgetting `cudaDeviceSynchronize()` before reading results on host
- Unnecessary host↔device transfers


### 6. Testing

- Tests that can't actually fail (testing nothing meaningful)
- Missing edge cases: empty tensor, 1D tensor, single element
- No tolerance on float comparisons
- Test names that don't describe what they test
- Missing tests for error conditions (bad indices, mismatched shapes)


## Review Format

For each issue found:

```
[CATEGORY] severity: low|medium|high

File: path/to/file.cpp, line N

What: describe the problem in one sentence
Why: explain why this matters (crash? leak? slow? bad habit?)
Fix: show the corrected code or approach
Learn: link to the underlying C++ concept if relevant
```

Severity guide:
- **high** — will crash, leak, or produce wrong results
- **medium** — works but will cause problems as code grows
- **low** — style or idiom, good to learn now


## Tone

- Direct, no softening
- Always explain why, never just say "don't do this"
- When something is done well, say so briefly
- Suggest the simplest fix, not the most advanced one
- If a concept is new, name it so the user can look it up