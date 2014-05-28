#pragma once

#include <stdint.h>
#include <vector_types.h>

#define kEvoIterationCountDefault 2000       // Iteration count of main loop
#define kEvoAlphaLimitDefault .0625f // alpha limit for images. Should be max(1.0f/255.0f, 50.0f / (float)kEvoMaxTriangleCount)
#define kEvoAlphaOffsetDefault (kEvoAlphaLimitDefault/2) //alpha offset
#define kEvoCheckLimitDefault 150 //number of times we have a non-improving value before we terminate
#define kEvoOutputScaleDefault 2 //the factor by which we would like to scale up the image in the final render
#define kEvoPsoParticleCountDefault 16    // number of particles for PSO. 16 seems fine; larger is slower and doesn't seem to help quality much.
#define kEvoPsoIterationCountDefault 1000 // maximum number of iterations at each PSO stage (overridden by checklimit)
#define kEvoMaxTriangleCountDefault 800 // number of triangles to render (max)
#define kEvoPsoNeighborhoodSizeDefault 4 // Size of neighbhorhood (PSO topology). roughly sqrt(kEvoPsoParticleCount) is a good guess.
#define kEvoPsoSpringConstantDefault 0.7f // Spring constant for PSO iteration. Anything in the range [0.7..2.0] seems to work equally well.
#define kEvoPsoDampeningFactorDefault 0.85f // Dampening factor for PSO iteration. Use 0.85f for faster short-term convergence; 0.999f for higher quality long-term convergence.

#define kEvoOutputFileDefault "out.png"
#define kEvoBlockDim 16   // grid width/height
#define kEvoNumFloatsPerTriangle 10

typedef struct PsoConstants
{
	int32_t iterationCount;
	float   alphaLimit;
	float   alphaOffset; // must be alphaLimit/2
	int32_t checkLimit;
	int32_t outputScale;
	int32_t psoParticleCount;
	int32_t psoIterationCount;
	int32_t maxTriangleCount;
	int32_t psoNeighborhoodSize;
	float   psoSpringConstant;
	float   psoDampeningFactor;
} PsoConstants;

//triangle class. stores color/alpha values, as well as coordinates of vertices
struct triangle {
	float x1;
	float y1;
	float x2;
	float y2;
	float x3;
	float y3;
	float r;
	float g;
	float b;
	float a; // TODO: unused
};

static_assert(sizeof(triangle) / sizeof(float) == kEvoNumFloatsPerTriangle, "sizeof(triangle) does not match expected value!");
