EvoGpuPso
=========
*(I pronounce it "EE-vawg PUP-soe")*

This is me trying to write an efficient (near-realtime, ideally!) implementation
of triangle-based image approximation, as popularized by Roger Alsing's EvoLisa project
(http://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/). The general
idea is to find a relatively small set of overlapping, flat-shaded, semi-transparent triangles
that best approximates a given image. The results are impressive; usually only a few hundred
triangles are necessary. Roger's original code was written in C# (!!!) and performed all
processing in a single CPU thread (!!!) using GDI for rendering (!!!!!). Its results are
impressive, but convergence can take *days*.

I'd like to think that I could do better.

I'm starting from an existing CUDA implementation by Drew Robb and Joy Ding
(http://isites.harvard.edu/fs/docs/icb.topic707165.files/pdfs/Ding_Robb.pdf), which uses
particle swarm optimization instead of Alsing's naive random-walk. Robb & Ding's code
[link? -ed] was designed to be launched from Python from a Linux host, so my first step
has been adapting the code to C++. From there I intend to profile and optimize the kernels.

Here's the Mona Lisa approximated with 500 triangles over 2400 iterations. Each frame
represents 100 iterations of the algorithm. This sequence took about 4 minutes to
evolve on my laptop's 650M (but was effectively stable after 1200 iterations in 100 seconds):

![image](https://raw.githubusercontent.com/cdwfs/evogpupso/master/lisa.gif)

 - 5/13/2014: I took a quick stupid-obvious-optimizations pass over the kernels. I hoisting some loop
   invariants, converted double-precision literals to single-precision, and inserted explicit fused
   multiply-adds where appropriate. Collectively, this made everything run about 4x faster. Along the way
   I found a typo in the triangle-scoring function (present in Ding & Robb's original source!) whose
   correction led to significantly better algorithmic convergence. So, 4x faster *and* better results; score!
   Nsight reports that occupancy is still at 62.5% (limited by register pressure), so there's still work to be
   done there. The instruction issue rate is also sitting at around 20% due to texture stalls, which is alarming.
   And I still haven't even started on some algorithmic improvements I have in mind...
 - 5/9/2014: Fixed some dumb Python->C++ mistakes. I'm now more or less reproducing Robb & Ding's
   results. My next step will be to actually read their paper, to gain a better understanding of the
   Particle Swarm Optimization algorithm and the tweakable parameters they expose. I also profiled
   the kernel using NVVP: the actual frame rendering is negligent compared to the PSO kernel,
   as expected. The PSO kernel is running at about 50% occupancy, and seems to include far more
   int<->float conversion instructions than I would have expected. Finally, the application currently
   isn't launching enough thread blocks to saturate a modern GPU. Any of these could be easy
   low-hanging fruit for optimization.
 - 5/8/2014: First successful run. Results are definitely not matching those described by Robb & Ding
   in their paper; the approximation does converge to a point, but seems to quickly hit a
   brick wall.