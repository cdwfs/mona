// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"

#include "utils.h"
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <direct.h>
#include <float.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// From evo.cu
extern void getTextureReferences(const textureReference **outRefImg, const textureReference **outCurrImg);
extern void setGpuConstants(const PsoConstants *constants);
extern void launch_render(float4 *d_im, triangle *d_curr, int *d_currentTriangleIndex, float *d_currentScore, int imgWidth, int imgHeight, int imgPitch);
extern void launch_renderproof(float4 * d_im, triangle * d_curr, int imgWidth, int imgHeight, int imgPitch);
extern void launch_run(int32_t particleCount, triangle *d_curr, triangle *d_pos, triangle *d_vel, float *d_fit,
	triangle *d_lbest, float *d_lbval, triangle *d_nbest, float *d_nbval, float *d_gbval,
	int *d_K, int imgWidth, int imgHeight);

// Clamp x to the range [min..max] (inclusive)
static inline float clamp(float x, float min, float max)
{
	return (x<min) ? min : ( (x>max) ? max : x );
}

static inline float randf(void)
{
	return (float)( (float)rand() / (float)RAND_MAX );
}

static void randomizeTrianglePos(triangle *tri)
{
	tri->x1 = randf();
	tri->y1 = randf();
	tri->x2 = randf();
	tri->y2 = randf();
	tri->x3 = randf();
	tri->y3 = randf();
	tri->r = randf();
	tri->g = randf();
	tri->b = randf();
	tri->a = randf();
}

static void randomizeTriangleVel(triangle *tri)
{
	tri->x1 = randf() * 2.0f - 1.0f;
	tri->y1 = randf() * 2.0f - 1.0f;
	tri->x2 = randf() * 2.0f - 1.0f;
	tri->y2 = randf() * 2.0f - 1.0f;
	tri->x3 = randf() * 2.0f - 1.0f;
	tri->y3 = randf() * 2.0f - 1.0f;
	tri->r = randf() * 2.0f - 1.0f;
	tri->g = randf() * 2.0f - 1.0f;
	tri->b = randf() * 2.0f - 1.0f;
	tri->a = randf() * 2.0f - 1.0f;
}

static void usage(void)
{
	printf("\
Usage: evogpupso [OPTIONS] <input image>\n\
OPTIONS:\n\
-out FILE.png      Write final output image to FILE.png [default: %s]\n\
-stats FILE        Write per-generation statistics to FILE [default: none]\n\
-temp DIR          Write intermediate image results to DIR. Images will be\n\
                   written every 100 generations. [default: none]\n\
-scale N           Scale factor for output image. Must be >= 1. [default: %d]\n\
-gens N            Number of full PSO generations to run [default: %d]\n\
-tris N            Number of triangles to render [default: %d]\n\
-psoiter N         Number if PSO iterations within each generation.\n\
                   Supports an early-out limit; see below. [default: %d]\n\
-checklimit N      PSO will early out after this many generations after\n\
		           results stop improving. [default: %d]\n\
-particles N       Number of PSO \"particles\" to simulate for each\n\
		           generation. [default: %d]\n\
-spring N          PSO spring constant. Should be in the range [0.7...2.0].\n\
		           Probably safe to leave as-is. [default: %.2f]\n\
-damp N            PSO dampening factor. Should be in the range [0.7..1.0].\n\
		           Lower values tend to converge better earlier; higher\n\
		           values converge better later. [default: %.2f]\n\
",
		kEvoOutputFileDefault,
		kEvoOutputScaleDefault,
		kEvoIterationCountDefault,
		kEvoMaxTriangleCountDefault,
		kEvoPsoIterationCountDefault,
		kEvoCheckLimitDefault,
		kEvoPsoParticleCountDefault,
		kEvoPsoSpringConstantDefault,
		kEvoPsoDampeningFactorDefault);
}

int main(int argc, char *argv[])
{
	srand( (unsigned int)time(NULL) );
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount != 1)
	{
		printf("ERROR: expected one CUDA device; found %d.\n", deviceCount);
		return -1;
	}
	const int deviceId = 0;
	CUDA_CHECK( cudaSetDevice(deviceId) );

	cudaDeviceProp deviceProps;
	CUDA_CHECK( cudaGetDeviceProperties(&deviceProps, deviceId) );
	const int32_t deviceMpCount = deviceProps.multiProcessorCount;
	const int32_t deviceCudaCoresPerMp = _ConvertSMVer2Cores(deviceProps.major, deviceProps.minor);

	printf("%s (%d MPs, %d cores/MP -> %d CUDA cores\n\n", deviceProps.name, deviceMpCount, deviceCudaCoresPerMp, deviceMpCount*deviceCudaCoresPerMp);

	// Each PSO thread block processes one PSO particle. It would be wasteful to simulate fewer particles than the GPU can support at full occupancy!
	// Compute maximum number of PSO thread blocks per MP
	const int32_t deviceMaxPsoBlocksPerMp = deviceProps.maxThreadsPerMultiProcessor / (kEvoBlockDim*kEvoBlockDim);
	// Launch 4x as many blocks as the GPU can support, to make sure it's fully saturated.
	//  - TODO: Actually use this value.
	//  - TODO: The PSO neighborhood size should be based on this value as well.
	const int32_t kEvoPsoGridDim = 4 * deviceMpCount * deviceMaxPsoBlocksPerMp;
	(void)kEvoPsoGridDim;

	// Set up PSO constants (overridable from command line)
	const char *inImageFileName   = nullptr;
	const char *outImageFileName  = kEvoOutputFileDefault;
	const char *outStatsFileName  = nullptr;
	const char *outTempDirName    = nullptr;
	PsoConstants constants = {};
	constants.iterationCount      = kEvoIterationCountDefault;
	constants.alphaLimit          = kEvoAlphaLimitDefault;
	constants.checkLimit          = kEvoCheckLimitDefault;
	constants.outputScale         = kEvoOutputScaleDefault;
	constants.psoParticleCount    = kEvoPsoParticleCountDefault;
	constants.psoIterationCount   = kEvoPsoIterationCountDefault;
	constants.maxTriangleCount    = kEvoMaxTriangleCountDefault;
	constants.psoNeighborhoodSize = kEvoPsoNeighborhoodSizeDefault;
	constants.psoSpringConstant   = kEvoPsoSpringConstantDefault;
	constants.psoDampeningFactor  = kEvoPsoDampeningFactorDefault;
	if (argc < 2)
	{
		usage();
		return -1;
	}
	for(int iArg=1; iArg<argc; ++iArg)
	{
		if        (strncmp(argv[iArg], "-out", 5) == 0 && iArg+1 < argc) {
			outImageFileName = argv[++iArg];
			continue;
		} else if (strncmp(argv[iArg], "-stats", 7) == 0 && iArg+1 < argc) {
			outStatsFileName = argv[++iArg];
			continue;
		} else if (strncmp(argv[iArg], "-temp", 6) == 0 && iArg+1 < argc) {
			outTempDirName = argv[++iArg];
			continue;
		} else if (strncmp(argv[iArg], "-scale", 7) == 0 && iArg+1 < argc) {
			constants.outputScale = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-gens", 6) == 0 && iArg+1 < argc) {
			constants.iterationCount = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-tris", 6) == 0 && iArg+1 < argc) {
			constants.maxTriangleCount = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-psoiter", 9) == 0 && iArg+1 < argc) {
			constants.psoIterationCount = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-checklimit", 12) == 0 && iArg+1 < argc) {
			constants.checkLimit = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-particles", 11) == 0 && iArg+1 < argc) {
			constants.psoParticleCount = strtol(argv[++iArg], nullptr, 10);
			continue;
		} else if (strncmp(argv[iArg], "-spring", 8) == 0 && iArg+1 < argc) {
			constants.psoSpringConstant = (float)strtod(argv[++iArg], nullptr);
			continue;
		} else if (strncmp(argv[iArg], "-damp", 6) == 0 && iArg+1 < argc) {
			constants.psoDampeningFactor = (float)strtod(argv[++iArg], nullptr);
			continue;
		} else if (strncmp(argv[iArg], "--help", 7) == 0) {
			usage();
			return 0;
		} else if (iArg+1 == argc) {
			// Final argument is the input file
			inImageFileName = argv[iArg];
			continue;
		} else {
			// Unrecognized argument
			usage();
			return -1;
		}
	}
	// Validate arguments
	if (constants.outputScale < 1) {
		fprintf(stderr, "ERROR: output scale (%d) must be >= 1\n", constants.outputScale);
		return -1;
	}
	if (constants.iterationCount < 1) {
		fprintf(stderr, "ERROR: generation count (%d) must be >= 1\n", constants.iterationCount);
		return -1;
	}
	if (constants.maxTriangleCount < 1) {
		fprintf(stderr, "ERROR: triangle count (%d) must be >= 1\n", constants.maxTriangleCount);
		return -1;
	}
	if (constants.psoIterationCount < 1) {
		fprintf(stderr, "ERROR: PSO iteration count (%d) must be >= 1\n", constants.psoIterationCount);
		return -1;
	}
	if (constants.checkLimit < 1) {
		fprintf(stderr, "ERROR: PSO check limit (%d) must be >= 1\n", constants.checkLimit);
		return -1;
	}
	if (constants.psoParticleCount < 1) {
		fprintf(stderr, "ERROR: PSO particle count (%d) must be >= 1\n", constants.psoParticleCount);
		return -1;
	}
	if (constants.psoSpringConstant < 0) {
		fprintf(stderr, "ERROR: PSO spring constant (%.3f) must be >= 0\n", constants.psoSpringConstant);
		return -1;
	}
	if (constants.psoDampeningFactor < 0 || constants.psoDampeningFactor > 1.0f) {
		fprintf(stderr, "ERROR: PSO particle count (%.3f) must be in the range [0..1]\n", constants.psoDampeningFactor);
		return -1;
	}
	if (nullptr == inImageFileName) {
		fprintf(stderr, "ERROR: no input file specified!\n");
		return -1;
	}

	// Clear temp directory
	// rm ./temp_images/*

	// Load input image
	const uint32_t *inputPixels = nullptr;
	int imgWidth = -1, imgHeight = -1, imgNumComp = -1;
	{
		printf("Loading '%s'...\n", inImageFileName);
		inputPixels = (const uint32_t*)stbi_load(inImageFileName, &imgWidth, &imgHeight, &imgNumComp, 4);
		if (nullptr == inputPixels)
		{
			fprintf(stderr, "Error loading input image '%s': %s\n", inImageFileName, stbi_failure_reason());
			return -1;
		}
	}
	// Convert to F32x4, as expected by the CUDA code.
	float4 *h_originalPixels = (float4*)malloc(imgWidth*imgHeight*sizeof(float4));
	for(int32_t iPixel=0; iPixel<imgWidth*imgHeight; ++iPixel)
	{
		h_originalPixels[iPixel].x = (float)((inputPixels[iPixel] >>  0) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].y = (float)((inputPixels[iPixel] >>  8) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].z = (float)((inputPixels[iPixel] >> 16) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].w = (float)((inputPixels[iPixel] >> 24) & 0xFF) / 255.0f;
	}
	// Upload to GPU
	float4 *d_originalPixels = nullptr; // Goim
	size_t srcPitch = (size_t)imgWidth*sizeof(float4);
	size_t originalPixelsPitch = 0;
	CUDA_CHECK( cudaMallocPitch(&d_originalPixels, &originalPixelsPitch,                   srcPitch,           imgHeight) );
	CUDA_CHECK( cudaMemcpy2D(    d_originalPixels,  originalPixelsPitch, h_originalPixels, srcPitch, srcPitch, imgHeight, cudaMemcpyHostToDevice) );

	const textureReference *refImg = nullptr, *currImg = nullptr;
	getTextureReferences(&refImg, &currImg);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	size_t offset = 0;
	CUDA_CHECK( cudaBindTexture2D(&offset, refImg, d_originalPixels, &channelDesc, imgWidth, imgHeight, originalPixelsPitch) );

	// Create array of solution triangles
	triangle *h_currentTriangles = (triangle*)malloc(constants.maxTriangleCount*sizeof(triangle)); // curr
	memset(h_currentTriangles, 0, constants.maxTriangleCount*sizeof(triangle));
	triangle *d_currentTriangles = nullptr; // Gcurr
	CUDA_CHECK( cudaMalloc(&d_currentTriangles,    constants.maxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemset( d_currentTriangles, 0, constants.maxTriangleCount*sizeof(triangle)) );
	triangle *h_bestTriangles = (triangle*)malloc(constants.maxTriangleCount*sizeof(triangle)); // oldCurr
	memcpy(h_bestTriangles, h_currentTriangles, constants.maxTriangleCount*sizeof(triangle));
	triangle *d_bestTriangles = nullptr;
	CUDA_CHECK( cudaMalloc(&d_bestTriangles, constants.maxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemcpy( d_bestTriangles, d_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );


	// Rendered solution on the GPU (and scaled-up version for final output)
	float4 *d_currentPixels = nullptr; // Gim
	CUDA_CHECK( cudaMallocPitch(&d_currentPixels, &originalPixelsPitch,    srcPitch, imgHeight) );
	CUDA_CHECK( cudaMemset2D(    d_currentPixels,  originalPixelsPitch, 0, srcPitch, imgHeight) );
	float4 *d_scaledOutputPixels = nullptr; // Gim3
	size_t scaledPixelsPitch = 0;
	CUDA_CHECK( cudaMallocPitch(&d_scaledOutputPixels, &scaledPixelsPitch,    constants.outputScale*srcPitch, constants.outputScale*imgHeight) );
	CUDA_CHECK( cudaMemset2D(    d_scaledOutputPixels,  scaledPixelsPitch, 0, constants.outputScale*srcPitch, constants.outputScale*imgHeight) );
	float4 *h_scaledOutputPixels   =   (float4*)malloc(constants.outputScale*imgWidth*constants.outputScale*imgHeight*sizeof(float4));
	uint32_t *scaledOutputRgba8888 = (uint32_t*)malloc(constants.outputScale*imgWidth*constants.outputScale*imgHeight*sizeof(uint32_t));

	// Index of triangle currently being updated
	int32_t currentTriangleIndex = 0; // K
	int32_t *d_currentTriangleIndex = nullptr; // GK
	CUDA_CHECK( cudaMalloc(&d_currentTriangleIndex, sizeof(int32_t)) );

	// Current score of this iteration, and best score to date
	float currentScore = FLT_MAX, bestScore = FLT_MAX;
	float *d_currentScore = nullptr; // Gscore
	CUDA_CHECK( cudaMalloc(&d_currentScore, sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(d_currentScore, &currentScore, sizeof(float), cudaMemcpyHostToDevice) );

	// PSO arrays
	triangle *h_psoParticlesPos          = (triangle*)malloc(constants.psoParticleCount*sizeof(triangle)); // pos
	triangle *h_psoParticlesVel          = (triangle*)malloc(constants.psoParticleCount*sizeof(triangle)); // vel
	float *h_psoParticlesFit             =    (float*)malloc(constants.psoParticleCount*sizeof(float));    // fit
	triangle *h_psoParticlesLocalBestPos = (triangle*)malloc(constants.psoParticleCount*sizeof(triangle)); // lbest
	float *h_psoParticlesLocalBestFit    =    (float*)malloc(constants.psoParticleCount*sizeof(float));    // lbval
	triangle *h_psoParticlesNhoodBestPos = (triangle*)malloc(constants.psoParticleCount*sizeof(triangle)); // nbest
	float *h_psoParticlesNhoodBestFit    =    (float*)malloc(constants.psoParticleCount*sizeof(float));    // nbval
	float psoParticlesGlobalBestFit      = FLT_MAX;    // gbval
	triangle *d_psoParticlesPos          = nullptr;    // Gpos
	triangle *d_psoParticlesVel          = nullptr;    // Gvel
	float *d_psoParticlesFit             = nullptr;    // Gfit
	triangle *d_psoParticlesLocalBestPos = nullptr;    // Glbest
	float *d_psoParticlesLocalBestFit    = nullptr;    // Glbval
	triangle *d_psoParticlesNhoodBestPos = nullptr;    // Gnbest
	float *d_psoParticlesNhoodBestFit    = nullptr;    // Gnbval
	float *d_psoParticlesGlobalBestFit   = nullptr;    // Ggbval
	CUDA_CHECK( cudaMalloc(&d_psoParticlesPos,          constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesVel,          constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesFit,          constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesLocalBestPos, constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesLocalBestFit, constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesNhoodBestPos, constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesNhoodBestFit, constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesGlobalBestFit,                     sizeof(float)) );

	// Upload constants to GPU
	setGpuConstants(&constants);

	// Open stats file, if necessary
	FILE *statsFile = nullptr;
	if (nullptr != outStatsFileName)
	{
		fopen_s(&statsFile, outStatsFileName, "w");
		if (nullptr == statsFile)
		{
			fprintf(stderr, "ERROR: Could not open '%s'\n", outStatsFileName);
			return -1;
		}
		printf("Writing stats to '%s'...\n", outStatsFileName);
		fprintf(statsFile, "# Input: %s\n", inImageFileName);
		fprintf(statsFile, "# Date: %s\n", "today");
		fprintf(statsFile, "# Command: TODO\n"); // TODO: write the full command line here
		fprintf(statsFile, "# Iteration:\tTime (sec)\tPSNR (dB):\n");
	}

	// Create directory for temp images, if necessary
	if (nullptr != outTempDirName)
	{
		// TODO: should be much more robust
		// (e.g. abort if directory can't be created, create more than one directory level, etc.)
		_mkdir(outTempDirName);
	}

	CpuTimer cpuTimer;
	cpuTimer.Start();
	for(int32_t iIter=1; iIter<=constants.iterationCount; ++iIter)
	{
		// Choose a new random triangle to update
		currentTriangleIndex = rand() % min((iIter+1)/2, constants.maxTriangleCount);
		CUDA_CHECK( cudaMemcpy(d_currentTriangleIndex, &currentTriangleIndex, sizeof(int32_t), cudaMemcpyHostToDevice) );

		// Render initial solution
		launch_render(d_currentPixels, d_currentTriangles, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, originalPixelsPitch/sizeof(float4));
		CUDA_CHECK( cudaBindTexture2D(&offset, currImg, d_currentPixels, &channelDesc, imgWidth, imgHeight, originalPixelsPitch) );
		CUDA_CHECK( cudaMemcpy(&currentScore, d_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );

		// check that this isn't a huge regression, revert and pick new K if so
		if (currentScore * (1.0f - 2.0f / (float)constants.maxTriangleCount) > bestScore)
		{
			memcpy(h_currentTriangles, h_bestTriangles, constants.maxTriangleCount*sizeof(triangle));
			CUDA_CHECK( cudaMemcpy(d_currentTriangles, h_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
			launch_render(d_currentPixels, d_currentTriangles, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, originalPixelsPitch/sizeof(float4));
			CUDA_CHECK( cudaMemcpy(&currentScore, d_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );
		}

		// Update best score if needed
		if (currentScore < bestScore && currentScore != 0)
		{
			bestScore = currentScore;
			// Update best known solution
			memcpy(h_bestTriangles, h_currentTriangles, constants.maxTriangleCount*sizeof(triangle));
		}

		// Print output
		const float mse = bestScore / (float)(3*imgWidth*imgHeight);
		const float psnr = 10.0f * log10(1.0f * 1.0f / mse);
		cpuTimer.Update();
		printf("%7.2fs (gen %4d)   bestScore: %.4f   PSNR: %.4f dB\n", cpuTimer.GetElapsedSeconds(), iIter, bestScore, psnr);
		if (nullptr != statsFile)
		{
			fprintf(statsFile, "%6d\t\t%7.2f\t\t%12.4f\n", iIter, cpuTimer.GetElapsedSeconds(), psnr);
		}

		// texturize current solution
		CUDA_CHECK( cudaBindTexture2D(&offset, currImg, d_currentPixels, &channelDesc, imgWidth, imgHeight, originalPixelsPitch) );

		// create random data for this PSO iter, and send to device
		for(int32_t iParticle=0; iParticle<constants.psoParticleCount; ++iParticle)
		{
			randomizeTrianglePos(h_psoParticlesPos+iParticle);
			randomizeTriangleVel(h_psoParticlesVel+iParticle);
			h_psoParticlesFit[iParticle] = FLT_MAX;
			randomizeTrianglePos(h_psoParticlesLocalBestPos+iParticle);
			h_psoParticlesLocalBestFit[iParticle] = FLT_MAX;
			randomizeTrianglePos(h_psoParticlesNhoodBestPos+iParticle);
			h_psoParticlesNhoodBestFit[iParticle] = FLT_MAX;
		}
		psoParticlesGlobalBestFit = FLT_MAX;
		CUDA_CHECK( cudaMemcpy(d_psoParticlesPos,          h_psoParticlesPos,          constants.psoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesVel,          h_psoParticlesVel,          constants.psoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesFit,          h_psoParticlesFit,          constants.psoParticleCount*sizeof(float),    cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesLocalBestPos, h_psoParticlesLocalBestPos, constants.psoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesLocalBestFit, h_psoParticlesLocalBestFit, constants.psoParticleCount*sizeof(float),    cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesNhoodBestPos, h_psoParticlesNhoodBestPos, constants.psoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesNhoodBestFit, h_psoParticlesNhoodBestFit, constants.psoParticleCount*sizeof(float),    cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesGlobalBestFit, &psoParticlesGlobalBestFit,                         1*sizeof(float),    cudaMemcpyHostToDevice) );

		// run the pso kernel! the big one!
		launch_run(constants.psoParticleCount, d_currentTriangles, d_psoParticlesPos, d_psoParticlesVel, d_psoParticlesFit,
			d_psoParticlesLocalBestPos, d_psoParticlesLocalBestFit,
			d_psoParticlesNhoodBestPos, d_psoParticlesNhoodBestFit,
			d_psoParticlesGlobalBestFit,
			d_currentTriangleIndex, imgWidth, imgHeight);

		// Copy current solution back to host
		CUDA_CHECK( cudaMemcpy(h_currentTriangles, d_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToHost) );

		// Visual output
		if (nullptr != outTempDirName && (iIter % 100) == 0)
		{
			CUDA_CHECK( cudaMemcpy(d_bestTriangles, d_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );
			launch_renderproof(d_scaledOutputPixels, d_bestTriangles, constants.outputScale*imgWidth, constants.outputScale*imgHeight, scaledPixelsPitch/sizeof(float4));
			CUDA_CHECK( cudaMemcpy2D(h_scaledOutputPixels,  constants.outputScale*srcPitch, d_scaledOutputPixels, scaledPixelsPitch, constants.outputScale*srcPitch, constants.outputScale*imgHeight, cudaMemcpyDeviceToHost) );
			// Convert to RGBA8888 for output
			for(int32_t iPixel=0; iPixel<constants.outputScale*imgWidth*constants.outputScale*imgHeight; ++iPixel)
			{
				scaledOutputRgba8888[iPixel] =
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].x * 255.0f, 0.0f, 255.0f)) <<  0 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].y * 255.0f, 0.0f, 255.0f)) <<  8 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].z * 255.0f, 0.0f, 255.0f)) << 16 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].w * 255.0f, 0.0f, 255.0f)) << 24 );
			}
			// Write output image
			char tempImageFileName[128];
			_snprintf_s(tempImageFileName, 127, "./%s/%05d.png", outTempDirName, iIter);
			tempImageFileName[127] = 0;
			printf("Writing '%s'...\n", tempImageFileName);
			int32_t writeError = stbi_write_png(tempImageFileName, constants.outputScale*imgWidth, constants.outputScale*imgHeight,
				4, scaledOutputRgba8888, constants.outputScale*imgWidth*sizeof(uint32_t));
			if (writeError == 0)
			{
				fprintf(stderr, "Error writing temporary output image '%s'\n", tempImageFileName);
				return -1;
			}
		}
	}

	// Write final output image
	{
		CUDA_CHECK( cudaMemcpy(d_bestTriangles, d_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );
		launch_renderproof(d_scaledOutputPixels, d_bestTriangles, constants.outputScale*imgWidth, constants.outputScale*imgHeight, scaledPixelsPitch/sizeof(float4));
		CUDA_CHECK( cudaMemcpy2D(h_scaledOutputPixels,  constants.outputScale*srcPitch, d_scaledOutputPixels, scaledPixelsPitch, constants.outputScale*srcPitch, constants.outputScale*imgHeight, cudaMemcpyDeviceToHost) );
		// Convert to RGBA8888 for output
		for(int32_t iPixel=0; iPixel<constants.outputScale*imgWidth*constants.outputScale*imgHeight; ++iPixel)
		{
			scaledOutputRgba8888[iPixel] =
				( uint32_t(clamp(h_scaledOutputPixels[iPixel].x * 255.0f, 0.0f, 255.0f)) <<  0 ) |
				( uint32_t(clamp(h_scaledOutputPixels[iPixel].y * 255.0f, 0.0f, 255.0f)) <<  8 ) |
				( uint32_t(clamp(h_scaledOutputPixels[iPixel].z * 255.0f, 0.0f, 255.0f)) << 16 ) |
				( uint32_t(clamp(h_scaledOutputPixels[iPixel].w * 255.0f, 0.0f, 255.0f)) << 24 );
		}
		// Write output image.
		// TODO: select output format from [jpg, png, bmp]?
		printf("Writing '%s'...\n", outImageFileName);
		int32_t writeError = stbi_write_png(outImageFileName, constants.outputScale*imgWidth, constants.outputScale*imgHeight,
			4, scaledOutputRgba8888, constants.outputScale*imgWidth*sizeof(uint32_t));
		if (writeError == 0)
		{
			fprintf(stderr, "Error writing final output image '%s'\n", outImageFileName);
			return -1;
		}
	}

	// cleanup -- lots more to do here
	if (nullptr != statsFile)
	{
		fclose(statsFile);
	}
	free((void*)inputPixels);
	cudaDeviceReset();
}
