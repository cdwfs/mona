#include "zombolite.h"
#include "xmona.h"

#include <QtWidgets/QApplication>

#include <cassert>

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
