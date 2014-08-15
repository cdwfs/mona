// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"

#include "utils.h"
#include "stb_image.h"

#include <direct.h>
#include <float.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static void usage(void)
{
	printf("\
Usage: minimona [OPTIONS] <input image>\n\
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

	// CUDA initialization
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
	CUDA_CHECK( cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2) );

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
	PsoConstants constants;
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

	PsoContext *psoContext = new PsoContext;
	psoContext->init(imgWidth, imgHeight, h_originalPixels, constants);
	free(h_originalPixels);

	if (nullptr != outTempDirName)
	{
		// TODO: Clear temp directory
		// rm ./temp_images/*
	}

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
		psoContext->iterate();

		// Print output
		cpuTimer.Update();
		const float bestPsnr = psoContext->bestPsnr();
		printf("%7.2fs (gen %4d)   bestScore: %.4f   PSNR: %.4f dB\n", cpuTimer.GetElapsedSeconds(),
			iIter, psoContext->bestScore(), bestPsnr);
		if (nullptr != statsFile)
		{
			fprintf(statsFile, "%6d\t\t%7.2f\t\t%12.4f\n", iIter, cpuTimer.GetElapsedSeconds(), bestPsnr);
		}

		// Visual output
		if (nullptr != outTempDirName && (iIter % 100) == 0)
		{
			char tempImageFileName[128];
			_snprintf_s(tempImageFileName, 127, "./%s/%05d.png", outTempDirName, iIter);
			tempImageFileName[127] = 0;
			printf("Writing '%s'...\n", tempImageFileName);
			if (0 != psoContext->renderToFile(tempImageFileName))
			{
				fprintf(stderr, "Error writing temporary output image '%s'\n", tempImageFileName);
				return -1;
			}
		}
	}

	// Write final output image
	printf("Writing '%s'...\n", outImageFileName);
	if (0 != psoContext->renderToFile(outImageFileName) != 0)
	{
		fprintf(stderr, "Error writing final output image '%s'\n", outImageFileName);
		return -1;
	}

	// cleanup -- lots more to do here
	if (nullptr != statsFile)
	{
		fclose(statsFile);
	}
	stbi_image_free((void*)inputPixels);
	delete psoContext;
	cudaDeviceReset();
}
