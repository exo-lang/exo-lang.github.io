---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

# What does Exo do?

Exo is a domain-specific programming language that helps low-level
performance engineers transform very simple programs that specify
what they want to compute into very complex programs that do the
same thing as the specification, only much, much faster.

# Optimizing Code

# Videos

We gave a talk at PLDI 2022. When a video of the talk is available,
we will post it here!

# Getting Started

Working with Exo is easy! Start by creating a virtual environment 
for your project:

```
$ python3 -m venv venv
$ . venv/bin/activate
```

Now, install `exo-lang`:

```
$ python -m pip install -U setuptools wheel
$ python -m pip install exo-lang
```

We'll download an example program and try compiling it.

# Background & Motivation


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

# 🚧 Under Construction! (and Limitations)

Exo is a young research project and under active development.

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