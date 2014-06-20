#include "xmona.h"

#include <QOpenGLContext>

#include <cassert>
#include <cstdint>

xmona::xmona(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

	QObject::connect(ui.actionOpen, &QAction::triggered, this, &xmona::loadRefImage);

	//
	// Can I dynamically change the label image? I can!
	//
	const int32_t refWidth  = ui.label->width();
	const int32_t refHeight = ui.label->height();
	uint32_t *pixels = new uint32_t[refWidth * refHeight];
	for(int iY=0; iY<refHeight; ++iY)
	{
		for(int iX=0; iX<refWidth; ++iX)
		{
			pixels[iY*refWidth+iX] = 0xFF000000 + (iY<<8) + iX;
		}
	}
	QImage refImage( (uchar*)pixels, refWidth, refHeight, QImage::Format_RGBA8888);
	ui.label->setPixmap(QPixmap::fromImage(refImage));

	//
	// Set up the OpenGL view to render the evolved image
	//
	QSurfaceFormat format;
	format.setMajorVersion(3);
	format.setMinorVersion(3);
	format.setProfile(QSurfaceFormat::CoreProfile);
	format.setOption(QSurfaceFormat::DebugContext);

	bool success;
	m_platformContext = new QOpenGLContext;
	m_platformContext->setFormat(format);
	success = m_platformContext->create();
	assert(success);
	success = m_platformContext->isValid();
	assert(success);

	QGLContext *widgetGlContext = QGLContext::fromOpenGLContext(m_platformContext);

	m_glWidget = new MyGLWidget(widgetGlContext, this);
	ui.verticalLayout->addWidget(m_glWidget);
}

xmona::~xmona()
{
	delete m_glWidget;
	delete m_platformContext;
}
