#include "xmona.h"

#include <QOpenGLContext>

#include <cassert>

xmona::xmona(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);

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
