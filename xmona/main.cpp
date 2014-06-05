#include "xmona.h"

#include <QtWidgets/QApplication>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cassert>

// Use CUDA_CHECK() in host code to validate the result of CUDA functions that return an error.
#define ENABLE_CUDA_CHECK 1 // disable if you want CUDA_CHECK() to compile away
#if ENABLE_CUDA_CHECK
#define CUDA_CHECK(val) \
	do{ \
	cudaError err = (val); \
	if (err != cudaSuccess) { \
	char finalMsg[256]; \
	_snprintf_s(finalMsg, 255, "CUDA error in %s:\n%d:\t%s\n%s\n", __FILE__, __LINE__, #val, cudaGetErrorString(err)); \
	finalMsg[255] = 0; \
	OutputDebugStringA(finalMsg); \
	if (IsDebuggerPresent()) \
	__debugbreak(); \
				else \
				assert(err == cudaSuccess); \
	} \
	__pragma(warning(push)) \
	__pragma(warning(disable:4127)) /* constant conditional */ \
	} while(0) \
	__pragma(warning(pop))
#else
#define CUDA_CHECK(val) (val)
#endif

int main(int argc, char *argv[])
{
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

	QApplication a(argc, argv);
	xmona w;
	w.show();
	int appReturn = a.exec();

	cudaDeviceReset();
	return appReturn;
}
