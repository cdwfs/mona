#include "zombolite.h"
#include "mona.h"

#include <QtWidgets/QApplication>
#include <QMessageBox>

#include <cassert>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);

	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount != 1)
	{
		QMessageBox msgBox;
		msgBox.setText("No CUDA devices found");
		msgBox.setInformativeText("This program requires a CUDA-capable GPU.");
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
		return -1;
	}
	const int deviceId = 0;
	CUDA_CHECK( cudaSetDevice(deviceId) );

	cudaDeviceProp deviceProps;
	CUDA_CHECK( cudaGetDeviceProperties(&deviceProps, deviceId) );

	mona w;
	w.show();
	int appReturn = a.exec();

	cudaDeviceReset();
	return appReturn;
}
