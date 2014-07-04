#include "zombolite.h"
#include "mona.h"

#include "../stb_image.h"

#include <QFileDialog>
#include <QMessageBox>
#include <QOpenGLContext>

#include <cassert>
#include <cstdint>

mona::mona(QWidget *parent)
	: QMainWindow(parent)
	, m_platformContext(nullptr)
	, m_glWidget(nullptr)
	, m_psoConstants(nullptr)
	, m_psoContext(nullptr)
	, m_psoTimer(this)
	, m_bestPsnr(0)
{
	ui.setupUi(this);

	// Connect signals to slots
	QObject::connect(ui.actionOpen,  &QAction::triggered, this, &mona::loadRefImage);
	QObject::connect(ui.actionPause, &QAction::toggled, this, &mona::togglePause);

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
	m_psoConstants->outputScale = 1;
	// Initialize PSO context with the default reference image
	m_psoContext = new PsoContext;
	QImage defaultRefImage(":/mona/lisa.jpg");
	defaultRefImage = defaultRefImage.rgbSwapped();
	initPso(defaultRefImage.width(), defaultRefImage.height(), (const uint32_t*)defaultRefImage.bits());

	connect( &m_psoTimer, SIGNAL(timeout()), this, SLOT(iteratePso()) );
	m_psoTimer.start(0);
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
		delete m_psoContext;
		m_psoContext = new PsoContext;
		initPso(refImageWidth, refImageHeight, refImagePixels);

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

		// Recreate GL texture to match new ref image dimensions
		m_glWidget->resizeTexture(refImageWidth, refImageHeight);

		m_bestPsnr = 0;
		ui.iterationCount->display(0);
		if (!ui.actionPause->isChecked())
		{
			ui.actionPause->toggle();
		}
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

void mona::initPso(const int32_t refImageWidth, const int32_t refImageHeight, const uint32_t *refImagePixels)
{
	// Populate PSO context with reference image pixels
	float4 *h_originalPixels = (float4*)malloc(refImageWidth*refImageHeight*sizeof(float4));
	for(int32_t iPixel=0; iPixel<refImageWidth*refImageHeight; ++iPixel)
	{
		// Convert to F32x4, as expected by the CUDA code.
		h_originalPixels[iPixel].x = (float)((refImagePixels[iPixel] >>  0) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].y = (float)((refImagePixels[iPixel] >>  8) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].z = (float)((refImagePixels[iPixel] >> 16) & 0xFF) / 255.0f;
		h_originalPixels[iPixel].w = (float)((refImagePixels[iPixel] >> 24) & 0xFF) / 255.0f;
	}
	m_psoContext->init(refImageWidth, refImageHeight, h_originalPixels, *m_psoConstants);
	free(h_originalPixels); // no longer required
}

void mona::iteratePso(void)
{
	m_psoContext->iterate();
	ui.iterationCount->display(ui.iterationCount->intValue()+1);

	float newPsnr = m_psoContext->bestPsnr();
	if (newPsnr <= m_bestPsnr)
		return;
	// New best score! Update GL widget's display
	m_bestPsnr = newPsnr;
	cudaGraphicsResource_t cudaTexResource = m_glWidget->evoTexResource();
	if (nullptr == cudaTexResource)
	{
		// Make sure the destination texture has been initialized!
		m_glWidget->resizeTexture(m_psoContext->width(), m_psoContext->height());
		cudaTexResource = m_glWidget->evoTexResource();
	}
	CUDA_CHECK( cudaGraphicsMapResources(1, &cudaTexResource) );
	cudaArray_t evoTexArray = nullptr;
	size_t evoTexPixelsSize = m_psoContext->width() * m_psoContext->height() * sizeof(float4);
	CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray(&evoTexArray, cudaTexResource, 0, 0) );
	m_psoContext->renderToCudaArray(evoTexArray);
	CUDA_CHECK( cudaGraphicsUnmapResources(1, &cudaTexResource) );
	m_glWidget->updateGL(); // Force a repaint
}

void mona::togglePause(void)
{
	if (ui.actionPause->isChecked())
	{
		// unpause
		m_psoTimer.start(0);
		ui.pauseButton->setText("Pause");
	}
	else
	{
		// pause
		m_psoTimer.stop();
		ui.pauseButton->setText("Resume");
	}
}
