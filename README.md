EvoGpuPso
=========
*(I pronounce it "EE-vawg PUP-soe")*

This is me trying to write an efficient (near-realtime, ideally!) implementation
of triangle-based image approximation, as popularized by Roger Alsing's
[EvoLisa](http://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/) project.
The general idea is to find a relatively small set of overlapping, flat-shaded, semi-transparent
triangles that best approximates a given image. The results are impressive; usually only a few hundred
triangles are necessary. Roger's original code was written in C# (!!!) and performed all
processing in a single CPU thread (!!!) using GDI for rendering (!!!!!). Its results are
impressive, but convergence can take *days*.

I'd like to think that I could do better.

I'm starting from an existing CUDA implementation by [Joy Ding and Drew Robb](http://isites.harvard.edu/fs/docs/icb.topic707165.files/pdfs/Ding_Robb.pdf),
which uses particle swarm optimization instead of Alsing's naive random-walk. Ding & Robb's code
[link? -ed] was designed to be launched from Python from a Linux host, so my first step
has been adapting the code to C++. From there I intend to profile and optimize the kernels.

Here's the Mona Lisa approximated with 500 triangles over 2400 iterations. Each frame
represents 100 iterations of the algorithm. This sequence took about 4 minutes to
evolve on my laptop's 650M (but was effectively stable after 1200 iterations in 100 seconds):

![image](https://raw.githubusercontent.com/cdwfs/evogpupso/master/lisa.gif)

Change Log
----------
 - 5/14/2014: Reading Ding & Robb's paper revealed some easy improvements, mostly having to do with parameter tuning. My neighborhood size
   and alpha limit were both too large, and appropriate values can apparently be determined as functions of particle count and
   triangle count respectively. The PSO dampening factor also seems to have two optimal values to choose from, depending on the preference
   for short-term or long-term convergence quality. I'd like to roll all of this into one giant "perf/quality" knob, but for now if I
   tune everything for short-term performance it's about 33% faster than yesterday's build.
   I also located an even better resource than the paper: [Ding](http://joyding.tumblr.com/) & [Robb](http://drewrobb.com/)
   themselves!
 - 5/13/2014: Today I completed a quick optimization pass over the kernels. I hoisted some loop
   invariants, converted double-precision literals to single-precision, and inserted explicit fused
   multiply-adds where appropriate. Collectively, this made everything run about 4x faster. Along the way
   I found a typo in the triangle-scoring function whose correction led to significantly better algorithmic
   convergence. So, 4x faster *and* better results; score!
   Nsight reports that occupancy is still at 62.5% (limited by register pressure), so there's still work to be
   done there. The instruction issue rate is also sitting at around 20% due to texture stalls, which is alarming.
   And I still haven't even started on some algorithmic improvements I have in mind...
 - 5/9/2014: Fixed some dumb Python->C++ conversion errors. I'm now more or less reproducing Ding & Robb's
   results. My next step will be to actually read their paper, to gain a better understanding of their
   Particle Swarm Optimization implementation and the tweakable parameters they expose. I also profiled
   the kernel using NVVP: the actual image rendering is negligent compared to the iterated PSO kernel,
   as expected. The PSO kernel is running at about 50% occupancy, and seems to include far more
   int<->float conversion instructions than I would have expected. Finally, the application currently
   isn't launching enough thread blocks to saturate a modern GPU. Any of these could be easy
   low-hanging fruit for optimization.
 - 5/8/2014: First successful run. Results are definitely not matching those described by Ding & Robb
   in their paper; the approximation does converge to a point, but seems to quickly hit a
   brick wall.

Future Plans
------------
In no particular order; just listed here so I don't forget:
 - Investigate triangle selection method for each iteration. Currently, a triangle is selected randomly from the first iIter/2 entries.
   Would it be preferable to run each triangle at least once before random selection? Would it be helpful to sort the triangles by
   score or area, and favor the iteration of triangles whose current contribution to the final image is negligible?
 - The triangle rasterization loop could probably be improved. It's pretty branchy as-is. Dynamic Parallelism would probably help,
   if I had a GPU that supported it.
 - The PSO kernel currently assigns one particle to each thread block, to process over all covered pixels. I wonder if it would be any
   better to assign each thread block a region of pixels to process for all particles. My gut says no; it would probably turn each PSO
   iteration into a separate kernel launch. But, something needs to be done about those texture stalls...
   
Acknowledgements
----------------
Much thanks to:
 - [Roger Alsing](http://rogeralsing.com/) for the original EvoLisa algorithm.
 - [Joy Ding](http://joyding.tumblr.com/) and [Drew Robb](http://drewrobb.com/) for their CUDA implementation of PSO EvoLisa, and accompanying paper.
 - [Sean Barrett](http://nothings.org/) for stb_image and stb_image_write