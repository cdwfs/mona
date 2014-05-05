#include "utils.h"
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

int main(int argc, char *argv[])
{
	int deviceCount = 0;
	CUDA_CHECK( cudaGetDeviceCount(&deviceCount) );
	if (deviceCount != 1)
	{
		printf("ERROR: expected one CUDA device; found %d.\n", deviceCount);
		return -1;
	}
	const int deviceId = 0;
	CUDA_CHECK( cudaSetDevice(deviceId) );

	if (argc != 3)
	{
		printf("Usage: %s in.??? out.png\n", argv[0]);
		return -1;
	}

	// Load input image
	const uint32_t *inputPixels = NULL;
	int imgWidth = -1, imgHeight = -1, imgNumComp = -1;
	{
		const char *inputImageFileName = argv[1];
		printf("Loading '%s'...\n", inputImageFileName);
		inputPixels = (const uint32_t*)stbi_load(inputImageFileName, &imgWidth, &imgHeight, &imgNumComp, 4);
		if (inputPixels == NULL)
		{
			printf("Error loading input image '%s': %s\n", inputImageFileName, stbi_failure_reason());
			return -1;
		}
	}
	uint8_t *outputPixels = (uint8_t*)malloc(imgWidth*imgHeight*sizeof(uint8_t));

#if 0
	// Call student's function to convert pixels to greyscale
	{
		printf("Converting to greyscale on the GPU...\n");
		GpuTimer gpuTimer;
		gpuTimer.Start();
		convert_rgbaToGrey(outputPixels, inputPixels, imgWidth, imgHeight);
		gpuTimer.Stop();
		printf("GPU conversion completed in %6.3f msecs.\n", gpuTimer.GetElapsedSeconds() * 1000.0);
		CUDA_CHECK( cudaGetLastError() );
	}
#endif

	// Write output image
	const char *outImageFileName = argv[2];
	printf("Writing '%s'...\n", outImageFileName);
	int32_t writeError = stbi_write_png(outImageFileName, imgWidth, imgHeight, 1, outputPixels, imgWidth*sizeof(uchar1));
	if (writeError == 0)
	{
		printf("Error writing output image '%s'\n", outImageFileName);
		return -1;
	}

	free((void*)inputPixels);
	free((void*)outputPixels);
	cudaDeviceReset();
	return 0;
}
