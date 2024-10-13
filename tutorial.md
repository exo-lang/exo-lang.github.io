---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

<style>pre.highlight { line-height: 1.2em; max-height: 25em; overflow: scroll; }</style>

# Exo Case Study: Custom accelerator for RISC-V CPU

_Author: Julien de Castelnau_

This is a tutorial introducing Exo from the perspective of a case study on writing an optimized math kernel for a custom hardware accelerator. The tutorial is adapted from [this blog post](https://systemf.epfl.ch/blog/riscv-optimization/), which frames the problem of high-performance software engineering to a broader audience and contrasts Exo with other solutions too. We will skip an introduction of this problem and focus on how Exo can be used in our workflow, starting with a discussion of how we would write software by hand for our accelerator.

## Case Study Overivew

The hardware in question accelerates 4x4 dense matrix multiplications with a systolic array. It is implemented as an ISA extension to RISC-V, RVM (RISC-V Matrix) [^1], allowing it to be programmed through CPU instructions. In particular, our implementation is connected to a microcontroller called [X-HEEP](https://github.com/esl-epfl/x-heep). For those familiar with RVV, the RISC-V Vector extension, RVM works similarly. There are a set of 8 "tile" registers in the accelerator, each storing 4x4 matrices of 32-bit values. Arithmetic instructions compute operations on these tile registers, and load/store instructions transfer data between the register file and main memory. These instructions are issued by the CPU. For example, `mmasa.w m2, m0, m1` computes \\(m_2 += m_0 \cdot {m_1}^T\\). [The full ISA listing is available here](https://github.com/esl-epfl/xheep_matrix_spec).

[^1]: Several ISA extensions for RISC-V implementing matrix operations have been proposed, and the RVM name is used more than once. [Our "RVM"](https://github.com/esl-epfl/xheep_matrix_spec) is mostly based off the [RVM from Xuantie](https://github.com/XUANTIE-RV/riscv-matrix-extension-spec), although ours is much simplified for the purposes of implementation on a resource-constrained embedded device.

One notable property of this accelerator combined with the CPU is the way its instructions are handled. The CPU issues instructions in-order, including RVM instructions, but the accelerator is allowed to complete out of order. Consequently, the CPU can perform and retire instructions while the accelerator completes operations. As we will see, optimizing software for this accelerator is in large part a matter of devising how to best overlap the time spent on the CPU with the accelerator.

On the software side, we were mainly interested in embedded machine learning applications (tinyML). A common workload in this space is convolution, specifically the 1D variant. This routine is a core part of neural networks designed to recognize speech, or monitor EKGs in wearable devices. Thus, we chose to study a simplified 1D convolution layer. Basic familiarity of convolution is assumed, but the mathematical specification for this particular operator is as follows. Given an input stream \\(I[I_C][N]\\) of length \\(N\\) with \\(I_C\\) input channels, and a array of kernels \\(K[O_C][I_C][W]\\) of width \\(W\\) with \\(O_C\\) output channels, we intend to compute an output \\(O[O_C][N]\\) such that

$$   O[i][j] = \sum_{c = 0}^{I_C} \sum_{r = 0}^{W} \begin{cases} I[c][j+r] \cdot K[i][c][r] & \text{if } j+r<N \\ 0 & \text{otherwise} \end{cases}, 0 \leq i \leq O_C, 0 \leq j \leq N $$

Before we jump into the process using Exo to make this development process easier, let's look at how it is to write this software by hand, to serve as a baseline.

## Manual optimization

```c
void conv1d_cpu(int32_t *data, int32_t *kernels, int32_t *out) {
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = 0;
            for (int c = 0; c < IC; c++) {
                for (int r = 0; r < W; r++) {
                    if ((j + r) < N) {
                        out[i][j] += data[c][j + r] * kernels[i][c][r];
                    }                    
                }
            }
        }
    }
}
```
The above code is more or less a direct translation of the formula written above. While the arrays \\(I,K,O\\) from the formula are passed as pointers `data`, `kernels,out`, note that the accesses use bracket syntax for multidimensional arrays for readability. In reality, an expression such as `out[i*IW+j]` would be needed instead of `out[i][j]`.

For simplicity, we'll make a lot of assumptions in the following sections. Notably, whenever we tile a loop by some factor, we assume the factor divides evenly. Compensating for this is usually a matter of adding some loop epilogue to handle the remainder. Another large assumption is that the size of a kernel matches the tile size supported by the accelerator. Usually, the kernel will be smaller, so the tile needs to be padded. While this assumption is unrealistic, we will see that there is still a great deal of nuance to study in optimizing this simplified routine.

Our first order of business is to actually make use of our accelerator hardware. Currently, everything is running on the CPU. For that, we need to somehow express this computation in terms of matrix multiplication. Notice that the inner loops vaguely resemble the accumulation of a matrix multiply. For each element of the `out` matrix, we are doing a dot product between a row and column of `data` and `kernels` respectively. The one caveat is that we don't access one row of `data` at a time, but rather an irregular pattern given by `j+r`. We can thus separate out this access into its own loop, storing the result as another array, which we will call `y`:

```c
void conv1d_im2col(int32_t *data, int32_t *kernels, int32_t *out) {
    int32_t y[N][IC][W];
    // perform im2col
    for (int j = 0; j < N; j++) {
        for (int c = 0; c < IC; c++) {
            for (int r = 0; r < W; r++) {
                if ((j + r) < N) {
                    y[j][c][r] = data[c][j+r];
                } else {
                    y[j][c][r] = 0;
                }                    
            }
        }
    }
    // matrix multiplication
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < N; j++) {
            out[i][j] = 0;
            for (int c = 0; c < IC; c++) {
                for (int r = 0; r < W; r++) {
                    out[i][j] += y[j][c][r] * kernels[i][c][r];
                }
            }
        }
    }
}
```

This transformation is so common it has the name im2col, as matrix multiply hardware is often used to accelerate convolutions. Having separated out these sections we could consider writing an optimized tiled matrix multiplication routine and invoking that, the same way that OpenBLAS solves this problem. However, this would be missing a key property of our hardware. Recall that accelerator instructions do not block the CPU. So, we could overlap the time spent computing results with time spent repacking the data (im2col). Instead of doing im2col on the entire array, we should repack only a tile (the size the accelerator can handle) at a time. This means we are computing as much as possible as soon as the data is ready.

```c
#define TILE 4
void conv1d_im2col_tile(int32_t *data, int32_t *kernels, int32_t *out) {
    for (int tile_i = 0; tile_i < OC/TILE; tile_i++) {
        for (int tile_j = 0; tile_j < N/TILE; tile_j++) {
            for (int c = 0; c < IC; c++) {
                int32_t y[TILE][TILE];
                // perform im2col
                for (int j = 0; j < TILE; j++) {
                    // assumed that W == TILE!
                    for (int r = 0; r < TILE; r++) {
                        if (((tile_j*TILE + j) + r) < N) {
                            y[j][r] = data[c][(tile_j*TILE + j)+r];
                        } else {
                            y[j][r] = 0;
                        }                    
                    }
                }
                // matrix multiplication
                for (int i = 0; i < TILE; i++) {
                    for (int j = 0; j < TILE; j++) {
                        out[i][j] = 0;
                        for (int r = 0; r < TILE; r++) {
                            out[i][j] += y[j][r] 
                             * kernels[tile_i*TILE + i][c][r];
                        }
                    }
                }
            }
        }
    }
}
```

Note some subtle changes in this process: the order of loops has been changed from the original program. Before, for each input channel (`c`), we accumulated one scalar output result. Now, after having tiled the `i` and `j` loops, we are accumulating a 4x4 tile.

Now, the operations in the routine correspond nicely with the instructions supported by our accelerator. Instead of performing the 4x4 matrix multiplication on the CPU, we can directly offload this to the accelerator. We can also hold the intermediate result `out[i][j]` until the end of the loop, when we can store the accumulated register to main memory. To properly load the subset of the matrices into tile registers, we used the stride parameter of the load instruction, which represents the width of a row in bytes. For instance, when loading kernels, the width of a row is \\(IC\cdot W\\), so we pass \\(4 \cdot IC\cdot W\\)   (4 = `sizeof(int32)`).

```c
#define TILE 4
void conv1d_im2col_tile(int32_t *data, int32_t *kernels, int32_t *out) {
    for (int tile_i = 0; tile_i < OC/TILE; tile_i++) {
        for (int tile_j = 0; tile_j < IW/TILE; tile_j++) {
            asm volatile ("mzero m2");
            for (int c = 0; c < IC; c++) {
                int32_t y[TILE][TILE];
                // perform im2col
                for (int j = 0; j < TILE; j++) {
                    for (int r = 0; r < TILE; r++) {
                        if (((tile_j*TILE + j) + r) < N) {
                            y[j][r] = data[c][(tile_j*TILE + j)+r];
                        } else {
                            y[j][r] = 0;
                        }                    
                    }
                }
                // matrix multiplication
                asm volatile ("mld.w m0, (%0), %1"
                    :: "r"(y), "r"(TILE*4));
                asm volatile ("mld.w m1, (%0), %1"
                    :: "r"(&kernels[tile_i*TILE][c][0]), "r"(IC * W * 4));
                asm volatile ("mmasa.w m2, m0, m1");
            }
            asm volatile ("mst.w m2, (%0), %1"
                :: "r"(&out[tile_i*TILE][tile_j*TILE]), "r"(IW * 4));
        }
    }
}
```

At this point, this code performs around 4x faster than the scalar  code we started with. [^2] Still, we can further optimize it. Profiling the code reveals that the majority of the time is still spent simply doing im2col, and that the computation practically adds nothing to the total runtime, once again due to the nonblocking nature of the instructions. There is ample time for the matrix load and multiply to compute before the im2col loop provides the next piece of data. However, notice that the im2col result is unnecessarily computed for every `tile_i` iteration: the result is not dependent on `tile_i` at all. If we could reorder the loops and share the value of `y` for every iteration of `tile_i`, then in theory we may be able to speed up by a factor of up to \\(\frac{OC}{TILE}\\). In reality, since we are reducing over the `c` loop, we would need to store the tiles for each `tile_i` in different registers, which is not feasible as we only have 8. But 8 registers is still enough registers to store 4 different `tile_i` iterations, so we can unroll by a factor of 4.

[^2]: Performance is measured by measuring number of cycles to [execute the function on a fixed input size](https://github.com/esl-epfl/xheep_matrix_spec/blob/25ff6a6a04acdb47b421fa64d454052a0d6b2a15/examples/conv1d/main.c). Test is only ran once since the system has no caches.

```c
#define TILE 4
void conv1d_im2col_tile(int32_t *data, int32_t *kernels, int32_t *out) {
    for (int tile_i = 0; tile_i < OC/(TILE*4); tile_i++) {
        for (int tile_j = 0; tile_j < IW/TILE; tile_j++) {
            asm volatile("mzero m1");
            asm volatile("mzero m2");
            asm volatile("mzero m3");
            asm volatile("mzero m4");
            for (int c = 0; c < IC; c++) {
                int32_t y[TILE][TILE];
                for (int j = 0; j < TILE; j++) {
                    for (int r = 0; r < TILE; r++) {
                        if (((tile_j*TILE + j) + r) < N) {
                            y[j][r] = data[c][(tile_j*TILE + j)+r];
                        } else {
                            y[j][r] = 0;
                        }                    
                    }
                }
                // matrix multiplication
                asm volatile("mld.w m0, (%0), %1"
                    :: "r"(y), "r"(TILE * 4));
                asm volatile("mld.w m5, (%0), %1"
                    :: "r"(kernel_base), "r"(IC * KW * 4));
                asm volatile("mmasa.w m1, m0, m5");
                asm volatile("mld.w m6, (%0), %1"
                    :: "r"(kernel_base+TILE * IC * KW), "r"(IC * KW * 4));
                asm volatile("mmasa.w m2, m0, m6");
                asm volatile("mld.w m7, (%0), %1"
                    :: "r"(kernel_base+TILE * IC * KW*2), "r"(IC * KW * 4));
                asm volatile("mmasa.w m3, m0, m7");
                asm volatile("mld.w m5, (%0), %1"
                    :: "r"(kernel_base+TILE * IC * KW*3), "r"(IC * KW * 4));
                asm volatile("mmasa.w m4, m0, m5");      
            }
            asm volatile("mst.w m1, (%0), %1"
                :: "r"(&out[tile_i*TILE][tile_j*TILE]), "r"(IW * 4));
            asm volatile("mst.w m2, (%0), %1"
                :: "r"(&out[tile_i*TILE+TILE*1][tile_j*TILE]), "r"(IW * 4));
            asm volatile("mst.w m3, (%0), %1"
                :: "r"(&out[tile_i*TILE+TILE*2][tile_j*TILE]), "r"(IW * 4));
            asm volatile("mst.w m4, (%0), %1"
                :: "r"(&out[tile_i*TILE+TILE*3][tile_j*TILE]), "r"(IW * 4));
        }
    }
}
```

As we would expect, the new code yields another roughly 4x speedup from the previous iteration.

Here we saw firsthand the impacts of optimizing for specialized hardware. We saw a roughly 16x speedup from our final to initial routines, but it was also 5x the number of lines of code. Notably, we lost the connection to the mathematical formula we started with. While the correctness of the original code was straightforward to audit at a glance, our final routine employed bespoke inline assembly and data movement techniques which don't readily correspond to the original specification. Any further maintenance on this code requires thinking through these techniques, and understanding their correctness, to manipulate it, which only becomes a bigger problem as more optimizations are applied.

Now, let's see how Exo can improve this workflow.

## Exo

With Exo, the programmer starts by writing their algorithm in a language which resembles plain Python. Then, they manipulate this program by providing a *schedule*, which is composed of a handful of *scheduling directives*. These are transformations describing rewrites on the program, such as unrolling a loop, or reordering two loops in a nest. The goal is to compose these to yield a program which is optimized for the target hardware, starting with the (typically much simpler) algorithm as the base. The idea is that exposing the schedule as a separate language construct enables the programmer to treat correctness and performance as independent concerns. We will see how this is applied in Exo.

In Exo, scheduling directives are written as ordinary Python functions manipulating the object corresponding to the program, which inform changes in the program's AST. Finally, the code is lowered down to C, which can be processed with a standard C compiler.

Let's dive right in with the convolution routine. To follow along with this tutorial, see [the setup document in the Exo repo](https://github.com/exo-lang/exo/tree/main/examples/rvm_conv1d). We'll start by expressing our algorithm, which corresponds to the "direct translation" version we started with in C:

```python
@proc
def generic_conv1d(
    data: i32[IC, N],
    kernels: i32[OC, IC, W],
    out: i32[OC, N],
):
    # do the convolution
    for i in seq(0, OC):
        for j in seq(0, N):
            # zero out the result memory
            out[i, j] = 0.0
            for c in seq(0, IC):
                for r in seq(0, W):
                    y: i32
                    if j + r < W:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
```

To optimize this routine, we'll pass the newly defined function `generic_conv1d` as an object to Exo's scheduling directives, which are just Python functions. The return value is a new procedure with the rewrite applied, which we can pass to further directives. We continue the process until we have arrived at a satisfactory schedule.

Along with the program itself, Exo's scheduling directives often need to be passed locations in the program to manipulate. For example, we may want to tell Exo to unroll one specific loop, rather than all of the loops in the program. One option Exo provides is to pass a string which is pattern matched against the program. So, to unroll a for loop with index "i", one could write `p = unroll_loop(p, "for i in _:_")` where `p` is the procedure object.

Better yet, Exo provides a system called "cursors," which lets you maintain a reference to a certain location, carried throughout the transformations you make on the program. In our case, we'll be repeatedly manipulating a lot of the same loops, so grabbing some cursors early on will help a lot with readability:

```python
# Before scheduling, grab cursors to the object code.
i_loop = p.find("for i in _:_")
j_loop = p.find("for j in _:_")
c_loop = p.find("for c in _:_")
y_alloc = p.find("y : _")
y_assign = p.find("y = data[_]")
```

Note that in this snippet and those that follow, `p` refers to the generic_conv1d routine. We've shortened it for brevity; these snippets belong to a function taking `p` as a parameter, which we are passing `generic_conv1d` to at the top level.

Having defined some useful cursors, we can begin scheduling the program. We'll go about things in a different order than we presented by hand, which was first im2col, tile for the accelerator's register, tile again to run 4 iterations in parallel, then reorder and unroll the 4 iterations. While this was an intuitive way to understand the performance evolution of the program, it's harder to express as transformations on the program when written in this order.

Even though we'll present it here in the order most conducive to Exo, a big benefit of a system like Exo is that the schedule can be written in any order. In fact, we originally wrote this schedule in the same order as our manual optimizations. The upside of having the schedule written out explicitly is that it's quite easy to go back and revise it somewhere in the middle, incrementally improving it with new directives. Writing by hand, every optimization we added was carried out on the result of all those before it, requiring a sort of global reasoning about the program.

With that said, we'll start with all of our tiling:

```python
# Tile outer loops to TILE size for RVM
p, _ = tile_loops(p, [(i_loop, TILE), (j_loop, TILE)], perfect=True)
# Compute 4 registers at a time
p, _ = tile_loops(p, [(i_loop, 4)], perfect=True)
```

We're using the scheduling directive tile_loops, passing the program `p`, along with the cursors we selected and the factor by which we'd like to tile by. In this case, we'd like to tile the `i` and `j` loops by a factor of 4 both, corresponding to our accelerator. Once again, we're doing things out of order, so here we're also going to tile again for the 4x compute optimization we made at the end.

```python
# Exo adds "o" and "i" suffix for outer and inner tiled loops respectively
i_loop_reg = p.find("for ioi in _:_")
p = reorder_loops(p, i_loop_reg)
```
Finally, we also want to reorder this new loop corresponding to the 4 registers so that we can unroll it on the innermost loops.

In Exo, we can `print()` the program at any point to see the result of our scheduling. Printing `p` right now yields:

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                          kernels: i32[16, 4, 4] @ DRAM,
                          out: i32[16, 16] @ DRAM):
    for ioo in seq(0, 1):
        for jo in seq(0, 4):
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out[ii + 4 * ioi + 16 * ioo, ji + 4 * jo] = 0.0
                        for c in seq(0, 4):
                            for r in seq(0, 4):
                                y: i32 @ DRAM
                                if ji + r + 4 * jo < 4:
                                    y = data[c, ji + r + 4 * jo]
                                else:
                                    y = 0
                                out[ii + 4 * ioi + 16 * ioo, ji +
                                    4 * jo] += kernels[ii + 4 * ioi + 16 * ioo,
                                                       c, r] * y
```

What we're going to aim for in the steps that follow is to expose the parts of this computation that can be offloaded to the accelerator. Much like we exposed matrix multiplication by hand with im2col so it was clear to us as programmers, we're going to "stage" the memory accesses into buffers matching the size supported by our accelerator, so we can later tell Exo to use our instructions instead.

We'll start with `out[]`. We want the compute loops to be operating on our staged buffer (which will eventually become registers) so we do that first. `stage_mem` is a directive provided by Exo that replaces all accesses to an array in some loop with a new array, and then inserts code after the loop to copy it back. We use `stage_mem` here to introduce `out_tile`, replacing all accesses in the `c` loop:

```python
# Stage output to out_tile
p, (out_alloc, out_tile, body, _) = auto_stage_mem(
    p, p.find_loop("c").expand(1, 0), "out", "out_tile", rc=True
)
```

Printing `p` gives us

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                          kernels: i32[16, 4, 4] @ DRAM,
                          out: i32[16, 16] @ DRAM):
    for ioo in seq(0, 1):
        for jo in seq(0, 4):
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out_tile: i32 @ DRAM
                        out_tile = 0.0
                        for c in seq(0, 4):
                            for r in seq(0, 4):
                                y: i32 @ DRAM
                                if ji + r + 4 * jo < 16:
                                    y = data[c, ji + r + 4 * jo]
                                else:
                                    y = 0
                                out_tile += kernels[ii + 4 * ioi + 16 * ioo, c,
                                                    r] * y
                        out[ii + 4 * ioi + 16 * ioo, ji + 4 * jo] = out_tile
```

Now all the code at the inside of this loop operates on `out_tile` instead. But we want `out_tile` to correspond to the size of a register. Actually, for each iteration, recall that we're trying to handle 4 output tiles. So we want it to be the size of 4 tiles. For this we should *lift* this scalar allocation out of the loops `ii,jj,ioi` which corresponds to the dimensions of these tiles, and turn it into a buffer of that size. In the end, `out_tile` will be a 3D dimensional array: 4 registers of 4x4 each.

Exo provides a directive in its standard library called `lift_alloc`, which moves the allocation out of a loop, and another called `expand_dim` which adds another dimension to a buffer dependent on some index variable. We can easily repeat these directives for each loop we want to lift out of. But even better, we can take advantage of the fact that Exo directives are just Python functions: we can create new rules composing existing ones. So here, we instead make a new function which repeats the `lift_alloc` - `expand_dim` process until some threshold size is reached:

```python
def autolift_alloc(p, alloc_c, dep_set=None, max_size=0, lift=True):
    """
    for i in seq(0, 10):
        for j in seq(0, 20):
            a : R          <- alloc_c, dep_set = {'i'}
            a[i] = ...
    ---->
    a : R[10]              <- if size is less than max_size
    for i in seq(0, n):
        for j in seq(0, m):
            a[i] = ...
    """
    alloc_c = p.forward(alloc_c)
    loop_c = get_enclosing_loop(p, alloc_c)
    accum_size = 1
    while True:
        try:
            if not isinstance(loop_c, pc.ForCursor):
                break
            if dep_set == None or loop_c.name() in dep_set:
                if (
                    isinstance(loop_c.hi(), LiteralCursor)
                    and accum_size * loop_c.hi().value() <= max_size
                ):
                    p = expand_dim(p, alloc_c, loop_c.hi().value(), loop_c.name())
                    accum_size = accum_size * loop_c.hi().value()
                    if lift:
                        p = lift_alloc(p, alloc_c)
            loop_c = loop_c.parent()
        except:
            break
    return p
```

This way, we can more concisely express the *intent* of our transformation (lifting out `out_tile` until it spans 4 tile registers) in our schedule:

```python
# lift out_tile to span 4 tile (4x4) registers
p = autolift_alloc(p, out_tile, max_size=4 * 4 * 4, dep_set=["ioi","ii","ji"])
```

This yields the following code when we print `p`:

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                          kernels: i32[16, 4, 4] @ DRAM,
                          out: i32[16, 16] @ DRAM):
    for ioo in seq(0, 1):
        for jo in seq(0, 4):
            out_tile: i32[4, 4, 4] @ DRAM
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out_tile[ioi, ii, ji] = 0.0
                        for c in seq(0, 4):
                            for r in seq(0, 4):
                                y: i32 @ DRAM
                                if ji + r + 4 * jo < 16:
                                    y = data[c, ji + r + 4 * jo]
                                else:
                                    y = 0
                                out_tile[ioi, ii,
                                         ji] += kernels[ii + 4 * ioi +
                                                        16 * ioo, c, r] * y
                        out[ii + 4 * ioi + 16 * ioo,
                            ji + 4 * jo] = out_tile[ioi, ii, ji]
```

Next, we want to reorder the loops so that `ioi`, `ii`, and `ji` are on the inside of `c`: for each channel `c`, we are doing 4 (`ioi`) matrix multiplications on each tile (`ii` x `ji`). Currently they are in the opposite order. Exo won't let us reorder nested loops with other statements in the way, however. Indeed, here it would be wrong to simply swap `ji` and `c` because we set `out_tile` based on the index given by `ji`. So first, we should split this statement, as well as the storing into its own loop - *fission*:

```python
# Block the zero initialization and store blocks
p = fission_as_much_as_possible(p, body)
p = fission_as_much_as_possible(p, body[0])
```

Once again, we've used a new helper function, `fission_as_much_as_possible`, which applies `fission` until Exo complains that it is invalid to do so.

Now we're ready to do the reordering:

```python
# Reorder c loop to the top
p = lift_scope_n(p, c_loop, 3)
```

We see that the `c` loop now encloses `ioi`, `ii`, and `ji`:

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                          kernels: i32[16, 4, 4] @ DRAM,
                          out: i32[16, 16] @ DRAM):
    for ioo in seq(0, 1):
        for jo in seq(0, 4):
            out_tile: i32[4, 4, 4] @ DRAM
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out_tile[ioi, ii, ji] = 0.0
            for c in seq(0, 4):
                for ioi in seq(0, 4):
                    for ii in seq(0, 4):
                        for ji in seq(0, 4):
                            for r in seq(0, 4):
                                y: i32 @ DRAM
                                if ji + r + 4 * jo < 4:
                                    y = data[c, ji + r + 4 * jo]
                                else:
                                    y = 0
                                out_tile[ioi, ii,
                                         ji] += kernels[ii + 4 * ioi +
                                                        16 * ioo, c, r] * y
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out[ii + 4 * ioi + 16 * ioo,
                            ji + 4 * jo] = out_tile[ioi, ii, ji]
```

Our next step is to apply the im2col transformation, where we separate out the setting of `y` into its own loop nest, making `y` a large buffer in the process. We can express this through applying fission between setting `y` and doing the multiply-accumulate, then lifting the `y` allocation up the loop nest. Afterwards, we stage the kernel and data matrices into new buffers just like with the output. We used most of these constructs to stage `out`, so we'll just show the whole block of the schedule:

```python
# Stage y
p = autolift_alloc(p, y_alloc, max_size=4 * 4, dep_set=["r","ji"])
p = lift_alloc(p, y_alloc, n_lifts=2)

# Fission the initialization loop and remove redundant loops
p = fission_as_much_as_possible(p, y_assign.parent())
p = remove_redundant_loops(p, y_assign.parent(), num=2)

# Stage kernels to kernel_tile and y to data_tile
ii_loop = p.forward(c_loop).body()[2].body()[0]
p, (kernel_alloc, _, _, _) = auto_stage_mem(
    p, ii_loop, "kernels", "kernel_tile", rc=True
)
p = simplify(expand_dim(p, kernel_alloc, 4, ii_loop.parent().name()))
p = lift_alloc(p, kernel_alloc)
p, (data_alloc, _, _, _) = auto_stage_mem(
p, ii_loop.parent(), "y", "data_tile", rc=True
```

Now im2col and matmul are in their own distinct loop nests, as we'd expect:

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                        kernels: i32[16, 4, 4] @ DRAM,
                        out: i32[16, 16] @ DRAM):
    for ioo in seq(0, 1):
        for jo in seq(0, 4):
            out_tile: i32[4, 4, 4] @ DRAM
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out_tile[ioi, ii, ji] = 0.0
            for c in seq(0, 4):
                y: i32[4, 4] @ DRAM
                for ji in seq(0, 4):
                    for r in seq(0, 4):
                        if ji + r + 4 * jo < 16:
                            y[ji, r] = data[c, ji + r + 4 * jo]
                        else:
                            y[ji, r] = 0
                kernel_tile: i32[4, 4, 4] @ DRAM
                data_tile: i32[4, 4] @ DRAM
                for i0 in seq(0, 4):
                    for i1 in seq(0, 4):
                        data_tile[i0, i1] = y[i0, i1]
                for ioi in seq(0, 4):
                    for i0 in seq(0, 4):
                        for i1 in seq(0, 4):
                            kernel_tile[ioi, i0,
                                        i1] = kernels[i0 + 4 * ioi + 16 * ioo,
                                                        c, i1]
                    for ii in seq(0, 4):
                        for ji in seq(0, 4):
                            for r in seq(0, 4):
                                out_tile[ioi, ii,
                                            ji] += kernel_tile[ioi, ii,
                                                            r] * data_tile[ji,
                                                                            r]
            for ioi in seq(0, 4):
                for ii in seq(0, 4):
                    for ji in seq(0, 4):
                        out[ii + 4 * ioi + 16 * ioo,
                            ji + 4 * jo] = out_tile[ioi, ii, ji]
```

We've just exposed several opportunities to offload work to our accelerator. The loop nests to load, multiply, and store  `data_tile`, `kernel_tile` and `out_tile` correspond nicely with the behavior of the RVM instructions. But how do we express this equivalence to Exo?

When writing by hand in C, we ripped out scalar code and replaced it with our special instructions as inline assembly, trusting that they did the same thing. For example, we got rid of the `out += data * kernels` statements and in the end replaced it with `mmasa.w`, the RVM matmul instruction. So far we've been able to express everything else we wrote in C equivalently, but inline assembly simply doesn't exist in Exo.

We could use some workaround like implementing a compiler backend that detects places in the C code to offload, selecting the appropriate RVM instruction. This could work, but it cripples the utility of the schedule as a "record" for the program's optimizations: the full performance picture is dependent on the behavior of this compiler, and the instructions it selects!

Fortunately, Exo offers a clever, and unique solution to this problem, one that nicely encapsulates all the behavior in the schedule itself. The Exo programmer themselves gives a definition of their hardware instructions as any other generic procedure, with standard scalar operations. For example, here is how we define our `mmasa.w` instruction:

```python
@instr('asm volatile("mmasa.w "{md_int}", "{ms1_int}", "{ms2_int});')
def rvm_mmasa(
    md: [i32][4, 4] @ RVM_TILE, ms1: [i32][4, 4] @ RVM_TILE,
    ms2: [i32][4, 4] @ RVM_TILE
):
    assert stride(md, 1) == 1
    assert stride(ms1, 1) == 1
    assert stride(ms2, 1) == 1
    for i in seq(0, 4):
        for j in seq(0, 4):
            for k in seq(0, 4):
                md[i, j] += ms2[i, k] * ms1[j, k]
```

The body of the procedure is nothing special: it is simply the Exo code for doing matrix multiply with scalar instructions. The key is in the `@instr` decorator we gave it, and the `replace()` directive in Exo.

`replace()` takes as arguments a cursor to some fragment of code inside a procedure, and a procedure whose body the fragment will be matched against. If Exo succeeds in unifying (i.e. pattern matching) the fragment with the body, then it will replace the fragment with a call to that procedure, automatically determining the correct arguments. You may be able to see where this is going: in the case of an instruction like we defined above, the body of the procedure is acting as a sort of *specification* for the instruction, encoding the semantics in a way that allows Exo to reason about when an offload is sound.

This seems like magic! We've managed to express the behavior of our accelerator inside Python. We can take arbitrary pieces of code in Exo and reason about if it's safe to offload some instruction there. No compiler backend needed. But the result is just procedure calls in the Exo language. When we go to compile it to C, how do we actually generate code which is appropriate for the accelerator?

This is where the `@instr` decorator comes in. The string we provided is a snippet of arbitrary C code with holes in it. When Exo goes to compile the call to the corresponding procedure, it pastes this piece of C code, filling in the holes with the names of the arguments in the compiled C code. For example, if the Exo array `out` is passed to the store instruction `rvm_mst`, then the C code will be an inline assembly which is passed the C array `out.`

You may be wondering how we are dealing with the custom memory that our accelerator supports. Indeed, we have completely glossed over this detail, but it's not valid to offload an operator if the array passed to the Exo procedure is in main memory, while the accelerator expects it to reside in some kind of scratchpad or register. We have to manually orchestrate this data movement. Besides, it's not clear how we'd express an accelerator-specific memory in C code in a generic manner.

Once again, Exo has a solution for this specific problem. You may have noticed the `@ RVM_TILE` annotations in the above procedure definition. This is actually a custom class representing our accelerator's tile register memory. Once again, it is defined by us, the programmer:


```python
class RVM_TILE(StaticMemory):
    NUM_RVM_TILES = 8
    StaticMemory.init_state(NUM_RVM_TILES)
    tile_dict = {}

    ...

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not (shape[0].isdecimal() and int(shape[0]) == 4):
            raise MemGenError("Number of tile rows must be 4.")
        if not (shape[1].isdecimal() and int(shape[1]) == 4):
            raise MemGenError("Number of tile columns must be 4.")

        tile_num = cls.find_free_chunk()
        cls.mark(tile_num)
        cls.tile_dict[new_name] = tile_num
        return f'#define {new_name} "m{7-tile_num}"'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        tile_num = cls.tile_dict[new_name]
        del cls.tile_dict[new_name]
        cls.unmark(tile_num)
        return f"#undef {new_name}"
```

The effect of this class is two-fold:

 * The annotation allows programmers to specify where a buffer is located. The class corresponding to the allocation is then used to describe how to lower operations on the memory to C code. Exo does not verify the consistency of these annotations during the scheduling phase, and you can even freely modify which memory is selected for some buffer with the directive `set_memory`. However, since the methods in the class also define *what* operations are allowed on the memory, that restricts what Exo programs will compile successfully. As a result, Exo can check in the backend that the code does not violate the semantics of the memory annotations. For example, `load()`, meaning a scalar load like `A[0]`, is not defined for `RVM_TILE`, because we have no sense of what it means to load a single element from a tile register. With only `alloc()` and `free()` defined, we can only use `RVM_TILE` to invoke the special instruction procedures `rvm_mmasa`, `rvm_mld`, `rvm_mst`, necessitating that we orchestrate the proper data movement.

 * The implementations of these methods offer complete control for the accelerator code generation. For example, what we have in `alloc()` and `free()` is essentially a trivial register allocator: a free list is maintained in the parent class `StaticMemory`, and every `alloc()` takes a new free register, while `free()` puts it back. Spilling is not handled. The return value of these methods is yet another C fragment which Exo pastes when it compiles the program. Our solution here is on the hacky side, but it demonstrates the flexibility we have: we "allocate" a register by `#define`-ing the variable name to the register we selected. This macro gets copy-pasted into the inline assembly we use for our instructions.

Let's take a step back and return to our convolution routine. We had scheduled things to line up nicely with the structure of the RVM instructions. So the next step is to tell Exo to offload, using the constructs we discussed:

```python
# Set adequate memories
p = set_memory(p, y_alloc, DRAM_STATIC)
p = set_memory(p, out_tile, RVM_TILE)
p = set_memory(p, kernel_alloc, RVM_TILE)
p = set_memory(p, data_alloc, RVM_TILE)

# Replace inner loops to calls to RVM instructions
p = replace_all(p, [rvm_mzero, rvm_mst, rvm_mld, rvm_mmasa])
```

The `replace_all` we used here is a wrapper around `replace` which actually finds *all* the fragments matching the provided instructions and replaces them accordingly, rather than requiring it to be passed explicitly.

The resulting Exo code has all offloadable sections replaced by calls to the functions we wrote earlier. Once again, from the Exo perspective, these are just calls to other Exo functions, but they carry a special meaning during code generation which allows them to represent our assembly instructions.

```python
def exo_conv1d_tile_lt_kw(data: i32[4, 16] @ DRAM,
                        kernels: i32[16, 4, 4] @ DRAM,
                        out: i32[16, 16] @ DRAM):
for ioo in seq(0, 1):
    for jo in seq(0, 4):
        out_tile: i32[4, 4, 4] @ RVM_TILE
        for ioi in seq(0, 4):
            rvm_mzero(out_tile[ioi, 0:4, 0:4])
        for c in seq(0, 4):
            y: i32[4, 4] @ DRAM_STATIC
            for ji in seq(0, 4):
                for r in seq(0, 4):
                    if ji + r + 4 * jo < 4:
                        y[ji, r] = data[c, ji + r + 4 * jo]
                    else:
                        y[ji, r] = 0
            kernel_tile: i32[4, 4, 4] @ RVM_TILE
            data_tile: i32[4, 4] @ RVM_TILE
            rvm_mld(data_tile[0:4, 0:4], y[0:4, 0:4])
            for ioi in seq(0, 4):
                rvm_mld(
                    kernel_tile[ioi, 0:4, 0:4],
                    kernels[4 * ioi + 16 * ioo:4 + 4 * ioi + 16 * ioo, c,
                            0:4])
                rvm_mmasa(out_tile[ioi, 0:4, 0:4], data_tile[0:4, 0:4],
                            kernel_tile[ioi, 0:4, 0:4])
        for ioi in seq(0, 4):
            rvm_mst(
                out_tile[ioi, 0:4, 0:4],
                out[4 * ioi + 16 * ioo:4 + 4 * ioi + 16 * ioo,
                    4 * jo:4 + 4 * jo])
```

For our final transformations, we'd like to unroll each of the `ioi` loops, and allocate 4 different `out_tile` s, rather than having only one with an extra dimension. Exo of course provides the `unroll_loop` directive for loops, but also a directive to "unroll" a buffer: replace an allocation for a constant size \\(n\\) on a given dimension with \\(n\\) buffers without that dimension. We utilize these directives below.

```python
# Clean up
p = unroll_loop(p, "ioi")
p = unroll_loop(p, "ioi")
p = unroll_loop(p, "ioi")
p = simplify(p)
p = unroll_buffer(p, kernel_alloc, 0)
p = reuse_buffer(p, "kernel_tile_0: _", "kernel_tile_3: _")
```

We can now compile our full Exo program to C, and see that the result is quite similar to the one we wrote by hand:

```python
// exo_conv1d_tile_lt_kw(
//     data : i32[4, 16] @DRAM,
//     kernels : i32[16, 4, 4] @DRAM,
//     out : i32[16, 16] @DRAM
// )
void exo_conv1d_tile_lt_kw( void *ctxt, const int32_t* data, const int32_t* kernels, int32_t* out ) {
for (int_fast32_t ioo = 0; ioo < 1; ioo++) {
for (int_fast32_t jo = 0; jo < 4; jo++) {
    #define out_tile_0 "m7"
    #define out_tile_1 "m6"
    #define out_tile_2 "m5"
    #define out_tile_3 "m4"
    asm volatile("mzero "out_tile_0);
    asm volatile("mzero "out_tile_1);
    asm volatile("mzero "out_tile_2);
    asm volatile("mzero "out_tile_3);
    for (int_fast32_t c = 0; c < 4; c++) {
        static int32_t y[4 * 4];
        for (int_fast32_t ji = 0; ji < 4; ji++) {
        for (int_fast32_t r = 0; r < 4; r++) {
        if (ji + r + 4 * jo < 16) {
            y[ji * 4 + r] = data[c * 16 + ji + r + 4 * jo];
        } else {
            y[ji * 4 + r] = ((int32_t) 0);
        }
        }
        }
        #define kernel_tile_0 "m3"
        #define kernel_tile_1 "m2"
        #define kernel_tile_2 "m1"
        #define data_tile "m0"
        asm volatile("mld.w "data_tile", (%1), %0" :: "r"(4*(((struct exo_win_2i32c){
        &y[0], { 4, 1 } }).strides[0])), "r"(&y[0]));
        asm volatile("mld.w "kernel_tile_0", (%1), %0" :: "r"(4*(((struct exo_win_2i32c){
        &kernels[(16 * ioo) * (16) + (c) * 4], { 16, 1 } }).strides[0])),
        "r"(&kernels[(16 * ioo) * (16) + (c) * 4]));
        asm volatile("mmasa.w "out_tile_0", "data_tile", "kernel_tile_0);
        asm volatile("mld.w "kernel_tile_1", (%1), %0" :: "r"(4*(((struct exo_win_2i32c){
        &kernels[(4 + 16 * ioo) * (16) + (c) * 4], { 16, 1 } }).strides[0])),
        "r"(&kernels[(4 + 16 * ioo) * (16) + (c) * 4]));
        asm volatile("mmasa.w "out_tile_1", "data_tile", "kernel_tile_1);
        #undef kernel_tile_1
        asm volatile("mld.w "kernel_tile_2", (%1), %0" :: "r"(4*(((struct exo_win_2i32c){
        &kernels[(8 + 16 * ioo) * (16) + (c) * 4], { 16, 1 } }).strides[0])),
        "r"(&kernels[(8 + 16 * ioo) * (16) + (c) * 4]));
        asm volatile("mmasa.w "out_tile_2", "data_tile", "kernel_tile_2);
        #undef kernel_tile_2
        asm volatile("mld.w "kernel_tile_0", (%1), %0" :: "r"(4*(((struct exo_win_2i32c){
        &kernels[(12 + 16 * ioo) * (16) + (c) * 4], { 16, 1 } }).strides[0])),
        "r"(&kernels[(12 + 16 * ioo) * (16) + (c) * 4]));
        asm volatile("mmasa.w "out_tile_3", "data_tile", "kernel_tile_0);
        #undef data_tile
        #undef kernel_tile_0
    }
    asm volatile("mst.w "out_tile_0", (%1), %0" :: "r"(4*(((struct exo_win_2i32){
    &out[(16 * ioo) * (16) + 4 * jo], { 16, 1 } }).strides[0])), "r"(&out[(16 * ioo) * (16) + 4 * jo]));
    #undef out_tile_0
    asm volatile("mst.w "out_tile_1", (%1), %0" :: "r"(4*(((struct exo_win_2i32){
    &out[(4 + 16 * ioo) * (16) + 4 * jo], { 16, 1 } }).strides[0])), "r"(&out[(4 + 16 * ioo) * (16) + 4 * jo]));
    #undef out_tile_1
    asm volatile("mst.w "out_tile_2", (%1), %0" :: "r"(4*(((struct exo_win_2i32){
    &out[(8 + 16 * ioo) * (16) + 4 * jo], { 16, 1 } }).strides[0])), "r"(&out[(8 + 16 * ioo) * (16) + 4 * jo]));
    #undef out_tile_2
    asm volatile("mst.w "out_tile_3", (%1), %0" :: "r"(4*(((struct exo_win_2i32){
    &out[(12 + 16 * ioo) * (16) + 4 * jo], { 16, 1 } }).strides[0])), "r"(&out[(12 + 16 * ioo) * (16) + 4 * jo]));
    #undef out_tile_3
}
}
}
```

## Conclusion

That's it! The full Exo code corresponding to this demo can be found [in the Exo repo.](https://github.com/exo-lang/exo/tree/main/examples/rvm_conv1d). The page also discusses how to build the code using the custom RVM toolchain.

One aspect of Exo which we did not get to see, but is worth noting is its static analysis system. Exo verifies that rewrites applied in a schedule are sound through an 'effect' analysis. As of today, this effect analysis mainly encapsulates when and where locations in memory are accessed, and less about the values at those locations in memory ([although this is an active area of improvement](https://github.com/exo-lang/exo/pull/578)). For example, broadly speaking, swapping two statements in the code is considered valid only when the effects of the two statements are invisible to each other - one does not use the result of the other, or they do not both write to the same location, etc. Similar rules are defined for reordering loops, loop fusion & fission, etc.

In our case, behind the scenes, this system has been verifying that we have not made an unsound step at any point in our schedule. This is a powerful guarantee: Assuming the original program is correct (which is much easier to audit because we've maintained the algorithm itself), then Exo guarantees that our optimized code is correct too.