---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

<style>pre.highlight { line-height: 1.2em; max-height: 25em; overflow: scroll; }</style>

# What does Exo do?

Exo is a domain-specific programming language that helps low-level
performance engineers transform very simple programs that specify
what they want to compute into very complex programs that do the
same thing as the specification, only much, much faster.


# Background & Motivation

The highest performance hardware made today (such as Google's TPU, 
Apple's Neural Engine, or Nvidia's Tensor Cores) power key 
scientific computing and machine learning kernels: the Basic 
Linear Algebra Subroutines (BLAS) library, for example. However, 
these new chips—which take hundreds of engineers to design—are 
only as good (i.e. high performance) for application developers as 
these kernels allow.

Unlike other programming languages and compilers, Exo is built around 
the concept of _exocompilation_. Traditionally, compilers are built 
to automatically optimize programs for running on some piece of 
hardware. This is great for most programmers, but for performance 
engineers the compiler gets in the way as often as it helps. Because 
the compiler's optimizations are totally automatic, there's no good 
way to fix it when it does the wrong thing and gives you 45% 
efficiency instead of 90%.

# Optimizing Code

With exocompilation, we put the performance engineer back in the 
driver's seat. Responsibility for choosing which optimizations to 
apply, when, and in what order is externalized from the compiler, 
back to the performance engineer. This way they don't have to waste 
time fighting the compiler on the one hand, or doing everything 
totally manually on the other. At the same time, Exo takes 
responsibility for ensuring that all of these optimizations are 
correct. As a result, the performance engineer can spend their time 
improving performance, rather than debugging the complex, optimized 
code.

Another key part of exocompilation is that performance engineers can 
describe the new chips they want to optimize for, without having to 
modify the compiler. Traditionally, the definition of the hardware 
interface is maintained by the compiler developers. However, for most 
new accelerator chips, the hardware interface is proprietary. It also 
changes more frequently than for general purpose chips. Currently, 
companies have to maintain their own fork of a whole traditional 
compiler, modified to support their particular chip. This requires 
hiring teams of compiler developers in addition to the performance 
engineers.

We've shown that we can use Exo to quickly write code that's as 
performant as Intel's hand-optimized Math Kernel Library. We also 
have an ongoing collaboration with UC Berkeley to create code for 
GEMMINI, their open-source machine learning accelerator. We're 
actively working with engineers and researchers at a number of 
companies, and are eager to work with more!

# Videos

We gave a talk at PLDI 2022. When a video of the talk is available,
we will post it here!

# Getting Started

## Setting up

Working with Exo is easy! We'll follow a standard workflow for creating 
a new Python project. Start by creating a virtual environment for your 
project. Note that Exo requires Python 3.9

```
$ python3 -m venv venv
$ . venv/bin/activate
```

Now, install `exo-lang`:

```
$ python -m pip install -U setuptools wheel
$ python -m pip install exo-lang
```

## Hello Exo!

Let's write a naive matrix multiply function in Exo. Put the following
code in a file called `example.py`:

```
# example.py
from __future__ import annotations
from exo import *

@proc
def example_sgemm(
    M: size,
    N: size,
    K: size,
    C: f32[M, N] @ DRAM,
    A: f32[M, K] @ DRAM,
    B: f32[K, N] @ DRAM,
):
    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]
```

And now we can run the exo compiler:

```
$ exocc -o out --stem example example.py
$ ls out
example.c  example.h
```

These can either be compiled into a library (static or shared) or 
compiled directly into your application. You will need to write a
short runner program yourself to test this code. For example:

```
// main.c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "example.h"

float* new_mat(int size, float value) {
  float* mat = malloc(size * sizeof(*mat));
  for (int i = 0; i < size; i++) {
    mat[i] = value;
  }
  return mat;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s M N K\n", argv[0]);
    return EXIT_FAILURE;
  }

  int M = atoi(argv[1]);
  int N = atoi(argv[2]);
  int K = atoi(argv[3]);

  if (M < 1 || N < 1 || K < 1) {
    printf("M, N, and K must all be positive!\n");
    return EXIT_FAILURE;
  }

  float* A = new_mat(M * K, 2.0);
  float* B = new_mat(K * N, 3.0);
  float* C = new_mat(M * N, 0.0);

  const int n_trials = 1000;

  clock_t start = clock();
  for (int i = 0; i < n_trials; i++) {
    example_sgemm(NULL, M, N, K, C, A, B);
  }
  clock_t end = clock();

  int msec = (end - start) * n_trials / CLOCKS_PER_SEC;

  printf("Each iteration ran in %d milliseconds\n", msec);
}
```

Then this can be easily compiled and run:

```
$ gcc -I out/ -o runner main.c out/example.c
$ ./runner 128 128 128
Each iteration ran in 11590 milliseconds
```

Now it's time to write a schedule to make this fast. See our [PLDI
artifact] for a tutorial on using AVX2 to optimize the innermost 
kernel. 

# Is Exo Right for Me?

1. Are you optimizing numerical programs?
2. Are you targeting uncommon accelerator hardware or even 
   developing your own?
3. Do you need to get as close as possible to the physical limits 
   of the hardware you're targeting?

If you answered "yes!" to all of three questions, then Exo might 
be right for you!

In particular, if you just want to optimize image processing code 
for consumer CPUs and GPUs, then [Halide](https://halide-lang.org) 
might be a better fit.

# 🚧 Under Construction! 🚧 

Exo is a young research project and under active development. Here 
are some of the things we'd like to add to Exo (or perhaps know we 
cannot).

1. Automatic generation of runner programs to make benchmarking 
   easier.
2. A brand new and more robust scheduling API based on a new AST 
   pointing technology we're calling "cursors".
3. Support for data-dependent accesses (e.g. histograms).
4. An LLVM or MLIR backend

# Publications & Learning More

So far, we have published the following papers on Exo:

1. **[PLDI 22]:** [Yuka Ikarashi][yuka-web], [Gilbert Louis 
   Bernstein][gilbert-web], [Alex Reinking][alex-web], [Hasan 
   Genc][hasan-web], and [Jonathan Ragan-Kelley][jrk-web]. 2022. 
   [Exocompilation for productive programming of hardware 
   accelerators.][exo-acm] In Proceedings of the 43rd ACM SIGPLAN 
   International Conference on Programming Language Design and 
   Implementation (PLDI 2022). Association for Computing 
   Machinery, New York, NY, USA, 703–718.

[exo-acm]: https://dl.acm.org/doi/abs/10.1145/3519939.3523446
[yuka-web]: https://people.csail.mit.edu/yuka/
[gilbert-web]: http://www.gilbertbernstein.com/
[alex-web]: https://alexreinking.com
[hasan-web]: https://hngenc.github.io/
[jrk-web]: https://people.csail.mit.edu/jrk/
[PLDI artifact]: https://github.com/ChezJrk/exo-artifact/blob/main/examples/x86_matmul.py
[Halide]: https://halide-lang.org
[TVM]: https://tvm.apache.org/
[Tiramisu]: http://tiramisu-compiler.org/