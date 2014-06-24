#include "mona.h"

#include "../stb_image.h"

#include <QMessageBox>
#include <QFileDialog>
#include <QOpenGLContext>

#include <cassert>
#include <cstdint>

mona::mona(QWidget *parent)
	: QMainWindow(parent)
	, m_platformContext(nullptr)
	, m_glWidget(nullptr)
	, m_psoConstants(nullptr)
	, m_psoContext(nullptr)
{
	ui.setupUi(this);

	QObject::connect(ui.actionOpen, &QAction::triggered, this, &mona::loadRefImage);

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
	ui.verticalLayout->setGeometry(QRect(0,0,256,256));
	ui.verticalLayout->addWidget(m_glWidget);

	m_psoConstants = new PsoConstants;
	m_psoContext = new PsoContext;
}

mona::~mona()
{
	delete m_glWidget;
	delete m_platformContext;
	delete m_psoConstants;
	delete m_psoContext;
}

void mona::loadRefImage(void)
{
	bool success = true;

	// Prompt for file name
	QString refImageFileName = QFileDialog::getOpenFileName(this, tr("Open Reference Image"), "", tr("Images (*.png *.jpg *.gif *.bmp *.tga *.psd *.hdr)"));
	if (refImageFileName.size() == 0)
	{
		// user hit cancel; take no action
		return;
	}

	// Load reference image
	const uint32_t *refImagePixels = nullptr;
	int refImageWidth = -1, refImageHeight = -1, refImgNumComp = -1;
	refImagePixels = (const uint32_t*)stbi_load(refImageFileName.toUtf8(), &refImageWidth, &refImageHeight, &refImgNumComp, 4);
	if (nullptr == refImagePixels)
	{
		success = false;
	}
	if (success)
	{
		// Convert to F32x4, as expected by the CUDA code.
		float4 *h_originalPixels = (float4*)malloc(refImageWidth*refImageHeight*sizeof(float4));
		for(int32_t iPixel=0; iPixel<refImageWidth*refImageHeight; ++iPixel)
		{
			h_originalPixels[iPixel].x = (float)((refImagePixels[iPixel] >>  0) & 0xFF) / 255.0f;
			h_originalPixels[iPixel].y = (float)((refImagePixels[iPixel] >>  8) & 0xFF) / 255.0f;
			h_originalPixels[iPixel].z = (float)((refImagePixels[iPixel] >> 16) & 0xFF) / 255.0f;
			h_originalPixels[iPixel].w = (float)((refImagePixels[iPixel] >> 24) & 0xFF) / 255.0f;
		}
		m_psoContext->init(refImageWidth, refImageHeight, h_originalPixels, *m_psoConstants);

		// TODO: resize label to match image aspect ratio. Need to center it within the frame, though.
		// For now, we just stretch to square.
		QSize labelSize = ui.refImageLabel->size();
		//labelSize.setHeight(labelSize.height()/2);
		ui.refImageLabel->resize(labelSize);

		// Update label pixmap
		QImage refImage( (uchar*)refImagePixels, refImageWidth, refImageHeight, QImage::Format_RGBA8888 );
		ui.refImageLabel->setPixmap(QPixmap::fromImage(refImage));

		// Original pixels are no longer needed in host memory
		stbi_image_free(const_cast<uint32_t*>(refImagePixels));
		free(h_originalPixels);
	}

	if (!success)
	{
		QMessageBox msgBox;
		msgBox.setText( tr("Load Failed") );
		msgBox.setInformativeText( tr("The selected image (") + refImageFileName + tr(") could not be loaded. See below for details.") );
		msgBox.setDetailedText( stbi_failure_reason() );
		msgBox.setIcon(QMessageBox::Critical);
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.exec();
	}
}
