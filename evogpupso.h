#pragma once

#include <vector_types.h>

#define kEvoIterationCount 2000       // Iteration count of main loop
#define kEvoAlphaLimit 0.1f            // alpha limit for images
#define kEvoAlphaOffset (kEvoAlphaLimit/2) //alpha offset
#define kEvoCheckLimit 150 //number of times we have a non-improving value before we terminate
#define kEvoOutputScale 2 //the factor by which we would like to scale up the image in the final render
#define kEvoPsoParticleCount 64    // number of particles for PSO
#define kEvoBlockDim 16   // grid width/height
#define kEvoPsoIterationCount 1000 // maximum number of iterations at each PSO stage (overridden by checklimit)
#define kEvoMaxTriangleCount 800 // number of triangles to render (max)
#define kEvoPsoNeighborhoodSize 8 // Size of neighbhorhood (PSO topology)
#define kEvoPsoSpringConstant 0.7f // Spring constant for PSO iteration. Anything in the range [0.7..2.0] seems to work equally well.

//triangle class. stores color/alpha values, as well as coordinates of vertices
class triangle {
public:
	float x1;
	float y1;
	float x2;
	float y2;
	float x3;
	float y3;
	float4 c;
};
