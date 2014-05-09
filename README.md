This is me trying to write an efficient (near-realtime, ideally!) implementation
of triangle-based image approximation, as popularized by Roger Alsing's EvoLisa project
(http://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/). Roger's
original code was written in C# (!!!) and performed all processing in a single CPU thread
(!!!) using GDI for rendering (!!!!!). Its results are impressive, but convergence can
take days.

I'd like to think I could do better.

I'm starting from an existing CUDA implementation by Drew Robb and Joy Ding
(http://isites.harvard.edu/fs/docs/icb.topic707165.files/pdfs/Ding_Robb.pdf), which uses
particle swarm optimization instead of Alsing's naive random-walk. Robb & Ding's code
[link? -ed] was designed to be launched from Python from a Linux host, so my first step
has been adapting the code to C++. From there I intend to profile and optimize the kernels.

![image](https://raw.githubusercontent.com/cdwfs/evogpupso/master/lisa.gif)

*5/8/2014:*
 - First successful run. Results are definitely not matching those described by Robb & Ding
   in their paper; the approximation does converge to a point, but seems to quickly hit a
   brick wall.