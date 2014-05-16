// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"

#include "utils.h"
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <float.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>

// From evo.cu
void getTextureReferences(const textureReference **outRefImg, const textureReference **outCurrImg);
void launch_render(float4 *d_im, triangle *d_curr, int *d_currentTriangleIndex, float *d_currentScore, int imgWidth, int imgHeight, int imgPitch);
void launch_renderproof(float4 * d_im, triangle * d_curr, int imgWidth, int imgHeight, int imgPitch);
void launch_run(triangle *d_curr, triangle *d_pos, triangle *d_vel, float *d_fit,
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


int main(int argc, char *argv[])
{
	srand( (unsigned int)time(NULL) );
	int deviceCount = 0;
	CUDA_CHECK( cudaGetDeviceCount(&deviceCount) );
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

	if (argc != 2)
	{
		printf("Usage: %s in.???\n", argv[0]);
		return -1;
	}

	// Clear temp directory
	// rm ./temp_images/*

	// Load input image
	const uint32_t *inputPixels = nullptr;
	int imgWidth = -1, imgHeight = -1, imgNumComp = -1;
	{
		const char *inputImageFileName = argv[1];
		printf("Loading '%s'...\n", inputImageFileName);
		inputPixels = (const uint32_t*)stbi_load(inputImageFileName, &imgWidth, &imgHeight, &imgNumComp, 4);
		if (inputPixels == nullptr)
		{
			printf("Error loading input image '%s': %s\n", inputImageFileName, stbi_failure_reason());
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
	triangle *h_currentTriangles = (triangle*)malloc(kEvoMaxTriangleCount*sizeof(triangle)); // curr
	memset(h_currentTriangles, 0, kEvoMaxTriangleCount*sizeof(triangle));
	triangle *d_currentTriangles = nullptr; // Gcurr
	CUDA_CHECK( cudaMalloc(&d_currentTriangles,    kEvoMaxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemset( d_currentTriangles, 0, kEvoMaxTriangleCount*sizeof(triangle)) );
	triangle *h_bestTriangles = (triangle*)malloc(kEvoMaxTriangleCount*sizeof(triangle)); // oldCurr
	memcpy(h_bestTriangles, h_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle));
	triangle *d_bestTriangles = nullptr;
	CUDA_CHECK( cudaMalloc(&d_bestTriangles, kEvoMaxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemcpy( d_bestTriangles, d_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );


	// Rendered solution on the GPU (and scaled-up version for final output)
	float4 *d_currentPixels = nullptr; // Gim
	CUDA_CHECK( cudaMallocPitch(&d_currentPixels, &originalPixelsPitch,    srcPitch, imgHeight) );
	CUDA_CHECK( cudaMemset2D(    d_currentPixels,  originalPixelsPitch, 0, srcPitch, imgHeight) );
	float4 *d_scaledOutputPixels = nullptr; // Gim3
	size_t scaledPixelsPitch = 0;
	CUDA_CHECK( cudaMallocPitch(&d_scaledOutputPixels, &scaledPixelsPitch,    kEvoOutputScale*srcPitch, kEvoOutputScale*imgHeight) );
	CUDA_CHECK( cudaMemset2D(    d_scaledOutputPixels,  scaledPixelsPitch, 0, kEvoOutputScale*srcPitch, kEvoOutputScale*imgHeight) );
	float4 *h_scaledOutputPixels   =   (float4*)malloc(kEvoOutputScale*imgWidth*kEvoOutputScale*imgHeight*sizeof(float4));
	uint32_t *scaledOutputRgba8888 = (uint32_t*)malloc(kEvoOutputScale*imgWidth*kEvoOutputScale*imgHeight*sizeof(uint32_t));

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
	triangle *h_psoParticlesPos = (triangle*)malloc(kEvoPsoParticleCount*sizeof(triangle)); // pos
	triangle *h_psoParticlesVel = (triangle*)malloc(kEvoPsoParticleCount*sizeof(triangle)); // vel
	float *h_psoParticlesFit    =    (float*)malloc(kEvoPsoParticleCount*sizeof(float));    // fit
	triangle *h_psoParticlesLocalBestPos = (triangle*)malloc(kEvoPsoParticleCount*sizeof(triangle)); // lbest
	float *h_psoParticlesLocalBestFit    =    (float*)malloc(kEvoPsoParticleCount*sizeof(float));    // lbval
	triangle *h_psoParticlesNhoodBestPos = (triangle*)malloc(kEvoPsoParticleCount*sizeof(triangle)); // nbest
	float *h_psoParticlesNhoodBestFit    =    (float*)malloc(kEvoPsoParticleCount*sizeof(float));    // nbval
	float psoParticlesGlobalBestFit      = FLT_MAX; // gbval
	triangle *d_psoParticlesPos          = nullptr;    // Gpos
	triangle *d_psoParticlesVel          = nullptr;    // Gvel
	float *d_psoParticlesFit             = nullptr;    // Gfit
	triangle *d_psoParticlesLocalBestPos = nullptr;    // Glbest
	float *d_psoParticlesLocalBestFit    = nullptr;    // Glbval
	triangle *d_psoParticlesNhoodBestPos = nullptr;    // Gnbest
	float *d_psoParticlesNhoodBestFit    = nullptr;    // Gnbval
	float *d_psoParticlesGlobalBestFit   = nullptr;    // Ggbval
	CUDA_CHECK( cudaMalloc(&d_psoParticlesPos,          kEvoPsoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesVel,          kEvoPsoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesFit,          kEvoPsoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesLocalBestPos, kEvoPsoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesLocalBestFit, kEvoPsoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesNhoodBestPos, kEvoPsoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesNhoodBestFit, kEvoPsoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&d_psoParticlesGlobalBestFit,                     sizeof(float)) );

	CpuTimer cpuTimer;
	cpuTimer.Start();
	for(int32_t iIter=1; iIter<=kEvoIterationCount; ++iIter)
	{
		// Choose a new random triangle to update
		currentTriangleIndex = rand() % min((iIter+1)/2, kEvoMaxTriangleCount);
		CUDA_CHECK( cudaMemcpy(d_currentTriangleIndex, &currentTriangleIndex, sizeof(int32_t), cudaMemcpyHostToDevice) );

		// Render initial solution
		launch_render(d_currentPixels, d_currentTriangles, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, originalPixelsPitch/sizeof(float4));
		CUDA_CHECK( cudaBindTexture2D(&offset, currImg, d_currentPixels, &channelDesc, imgWidth, imgHeight, originalPixelsPitch) );
		CUDA_CHECK( cudaMemcpy(&currentScore, d_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );

		// check that this isn't a huge regression, revert and pick new K if so
		if (currentScore * (1.0f - 2.0f / (float)kEvoMaxTriangleCount) > bestScore)
		{
			memcpy(h_currentTriangles, h_bestTriangles, kEvoMaxTriangleCount*sizeof(triangle));
			CUDA_CHECK( cudaMemcpy(d_currentTriangles, h_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
			launch_render(d_currentPixels, d_currentTriangles, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, originalPixelsPitch/sizeof(float4));
			CUDA_CHECK( cudaMemcpy(&currentScore, d_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );
		}
		// Update best score if needed
		if (currentScore < bestScore && currentScore != 0)
		{
			bestScore = currentScore;
			// Update best known solution
			memcpy(h_bestTriangles, h_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle));
		}
		// Print output
		cpuTimer.Update();
		printf("%7.2fs (gen %4d): score = %.4f [best = %.4f]\n", cpuTimer.GetElapsedSeconds(), iIter, currentScore, bestScore);

		// texturize current solution
		CUDA_CHECK( cudaBindTexture2D(&offset, currImg, d_currentPixels, &channelDesc, imgWidth, imgHeight, originalPixelsPitch) );

		// create random data for this PSO iter, and send to device
		for(int32_t iParticle=0; iParticle<kEvoPsoParticleCount; ++iParticle)
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
		CUDA_CHECK( cudaMemcpy(d_psoParticlesPos, h_psoParticlesPos, kEvoPsoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesVel, h_psoParticlesVel, kEvoPsoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesFit, h_psoParticlesFit, kEvoPsoParticleCount*sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesLocalBestPos, h_psoParticlesLocalBestPos, kEvoPsoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesLocalBestFit, h_psoParticlesLocalBestFit, kEvoPsoParticleCount*sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesNhoodBestPos, h_psoParticlesNhoodBestPos, kEvoPsoParticleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesNhoodBestFit, h_psoParticlesNhoodBestFit, kEvoPsoParticleCount*sizeof(float), cudaMemcpyHostToDevice) );
		CUDA_CHECK( cudaMemcpy(d_psoParticlesGlobalBestFit, &psoParticlesGlobalBestFit, sizeof(float), cudaMemcpyHostToDevice) );

		// run the pso kernel! the big one!
		launch_run(d_currentTriangles, d_psoParticlesPos, d_psoParticlesVel, d_psoParticlesFit,
			d_psoParticlesLocalBestPos, d_psoParticlesLocalBestFit,
			d_psoParticlesNhoodBestPos, d_psoParticlesNhoodBestFit,
			d_psoParticlesGlobalBestFit,
			d_currentTriangleIndex, imgWidth, imgHeight);

		// Copy current solution back to host
		CUDA_CHECK( cudaMemcpy(h_currentTriangles, d_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToHost) );

		// Visual output
		if ((iIter % 100) == 0)
		{
			CUDA_CHECK( cudaMemcpy(d_bestTriangles, d_currentTriangles, kEvoMaxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );
			launch_renderproof(d_scaledOutputPixels, d_bestTriangles, kEvoOutputScale*imgWidth, kEvoOutputScale*imgHeight, scaledPixelsPitch/sizeof(float4));
			CUDA_CHECK( cudaMemcpy2D(h_scaledOutputPixels,  kEvoOutputScale*srcPitch, d_scaledOutputPixels, scaledPixelsPitch, kEvoOutputScale*srcPitch, kEvoOutputScale*imgHeight, cudaMemcpyDeviceToHost) );
			// Convert to RGBA8888 for output
			for(int32_t iPixel=0; iPixel<kEvoOutputScale*imgWidth*kEvoOutputScale*imgHeight; ++iPixel)
			{
				scaledOutputRgba8888[iPixel] =
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].x * 255.0f, 0.0f, 255.0f)) <<  0 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].y * 255.0f, 0.0f, 255.0f)) <<  8 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].z * 255.0f, 0.0f, 255.0f)) << 16 ) |
					( uint32_t(clamp(h_scaledOutputPixels[iPixel].w * 255.0f, 0.0f, 255.0f)) << 24 );
			}
			// Write output image
			char outImageFileName[128];
			_snprintf_s(outImageFileName, 127, "./temp_images/%04d%s", iIter, argv[1]);
			outImageFileName[127] = 0;
			printf("Writing '%s'...\n", outImageFileName);
			int32_t writeError = stbi_write_png(outImageFileName, kEvoOutputScale*imgWidth, kEvoOutputScale*imgHeight,
				4, scaledOutputRgba8888, kEvoOutputScale*imgWidth*sizeof(uint32_t));
			if (writeError == 0)
			{
				printf("Error writing output image '%s'\n", outImageFileName);
				return -1;
			}
		}

	}

	free((void*)inputPixels);
	cudaDeviceReset();
}
