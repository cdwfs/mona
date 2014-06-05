#include "xmona.h"
#include <QtWidgets/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	xmona w;
	w.show();
	return a.exec();
}
