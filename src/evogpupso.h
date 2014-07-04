#pragma once

#include <stdint.h>
#include <vector_types.h>

#pragma warning(push)
#pragma warning(disable:4127) // constant conditional
#include <curand_kernel.h>
#pragma warning(pop)

#define kEvoIterationCountDefault 2000       // Iteration count of main loop
#define kEvoAlphaLimitDefault .0625f // alpha limit for images. Should be max(1.0f/255.0f, 50.0f / (float)kEvoMaxTriangleCount)
#define kEvoCheckLimitDefault 150 //number of times we have a non-improving value before we terminate
#define kEvoOutputScaleDefault 2 //the factor by which we would like to scale up the image in the final render
#define kEvoPsoParticleCountDefault 16    // number of particles for PSO. 16 seems fine; larger is slower and doesn't seem to help quality much.
#define kEvoPsoIterationCountDefault 1000 // maximum number of iterations at each PSO stage (overridden by checklimit)
#define kEvoMaxTriangleCountDefault 800 // number of triangles to render (max)
#define kEvoPsoNeighborhoodSizeDefault 4 // Size of neighbhorhood (PSO topology). roughly sqrt(kEvoPsoParticleCount) is a good guess.
#define kEvoPsoSpringConstantDefault 0.7f // Spring constant for PSO iteration. Anything in the range [0.7..2.0] seems to work equally well.
#define kEvoPsoDampeningFactorDefault 0.85f // Dampening factor for PSO iteration. Use 0.85f for faster short-term convergence; 0.999f for higher quality long-term convergence.

#define kEvoOutputFileDefault "out.png"
#define kEvoBlockDim 16   // block width/height
#define kEvoNumFloatsPerTriangle 9

class PsoConstants
{
public:
	PsoConstants() : iterationCount(kEvoIterationCountDefault)
		, alphaLimit(kEvoAlphaLimitDefault)
		, checkLimit(kEvoCheckLimitDefault)
		, outputScale(kEvoOutputScaleDefault)
		, psoParticleCount(kEvoPsoParticleCountDefault)
		, psoIterationCount(kEvoPsoIterationCountDefault)
		, maxTriangleCount(kEvoMaxTriangleCountDefault)
		, psoNeighborhoodSize(kEvoPsoNeighborhoodSizeDefault)
		, psoSpringConstant(kEvoPsoSpringConstantDefault)
		, psoDampeningFactor(kEvoPsoDampeningFactorDefault)
	{
	}

	int32_t iterationCount;
	float   alphaLimit;
	int32_t checkLimit;
	int32_t outputScale;
	int32_t psoParticleCount;
	int32_t psoIterationCount;
	int32_t maxTriangleCount;
	int32_t psoNeighborhoodSize;
	float   psoSpringConstant;
	float   psoDampeningFactor;
};

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
};

static_assert(sizeof(triangle) / sizeof(float) == kEvoNumFloatsPerTriangle, "sizeof(triangle) does not match expected value!");

class PsoContext
{
public:
	PsoContext();
	~PsoContext();

	int init(int imgWidth, int imgHeight, const float4 *h_originalPixels, const PsoConstants &constants);

	/**
	 * @brief Run one iteration of PSO.
	 */
	void iterate(void);

	float bestScore(void) const { return m_bestScore; }
	float bestPsnr(void) const;
	int32_t width(void) const { return m_imgWidth; }
	int32_t height(void) const { return m_imgHeight; }

	int renderToCudaArray(cudaArray_t dst);
	int renderToFile(const char *imageFileName);

private:
	PsoConstants m_constants;
	int32_t m_currentIteration; ///< Current PSO iteration.
	int32_t m_imgWidth; ///< Width of reference image, in pixels.
	int32_t m_imgHeight; ///< Height of reference image, in pixels.

	cudaArray_t md_originalPixels; ///< Original reference pixels. [device memory]
	size_t m_currentPixelsPitch; ///< Pitch of md_currentPixels, in bytes.
	float4 *md_currentPixels; ///< Current candidate image pixels. Unscaled; same dimensions as original pixels. [device memory]
	float4 *mh_scaledOutputPixels; ///< Used to render final scaled output pixels. [host memory]
	cudaArray_t md_scaledOutputPixels; ///< Used to render final scaled output pixels. [device memory]
	uint32_t *mh_scaledOutputRgba8888; ///< Final scaled output pixels in RGBA8888 format, ready to write to disk. [host memory]
	const textureReference *m_refImg; ///< Reference to reference pixels as a CUDA texture.
	const textureReference *m_currImg; ///< Reference to current candidate pixels as a CUDA texture.
	const surfaceReference *m_scaledImg; ///< Reference to scaled output pixels as a CUDA texture.
	cudaChannelFormatDesc m_refChannelDesc; ///< channel format for m_refImg.
	cudaChannelFormatDesc m_currChannelDesc; ///< channel format for m_currImg (and m_scaledImg)
	triangle *mh_currentTriangles; ///< Current candidate triangles. [host memory]
	triangle *md_currentTriangles; ///< Current candidate triangles. [device memory]
	triangle *mh_bestTriangles; ///< Best candidate triangles found so far. [host memory]
	triangle *md_bestTriangles; ///< Best candidate triangles found so far. [device memory]
	int32_t *md_currentTriangleIndex; ///< Index of the triangle currently being processed. [device memory]
	float m_currentScore; ///< Sum of squared error across all md_currentPixels.
	float *md_currentScore; ///< Sum of squared error across all md_currentPixels. [device memory]
	float m_bestScore; ///< Best squared error across all candidate pixels seen so far.

	curandState_t *md_psoRandStates;        ///< RNG states for each PSO particle. [device memory]
	triangle *md_psoParticlesPos;           ///< Current position of each PSO particle. [device memory]
	triangle *md_psoParticlesVel;           ///< Current velocity of each PSO particle. Range is [-1..1]. [device memory]
	float    *md_psoParticlesFit;           ///< Current summed-squared-error contribution of each particle. [device memory]
	triangle *md_psoParticlesLocalBestPos;  ///< Best position found so far for each particle. [device memory]
	float    *md_psoParticlesLocalBestFit;  ///< Lowest summed-squared-error found so far for each particle so far. [device memory]
	triangle *md_psoParticlesNhoodBestPos;  ///< Best position found to date among each neighborhood of particles. [device memory]
	float    *md_psoParticlesNhoodBestFit;  ///< Lowest summed-squared-error found so far among each neighborhood of particles. [device memory]
	float    *md_psoParticlesGlobalBestFit; ///< Lowest summed-squared-error found so far among all particles. [device memory]

	void setGpuConstants(const PsoConstants *constants);
	void launchSrand();
	void launchInitPsoParticles();
	void launchRender();
	void launchRenderProof();
	void launchRun();
};
