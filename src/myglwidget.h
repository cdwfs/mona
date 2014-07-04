#pragma once

#include <QGLWidget>
#include <driver_types.h> // for cudaGraphicsResource

class MyGLWidget : public QGLWidget
{
	Q_OBJECT

public:
	MyGLWidget(QGLContext *context, QWidget *parent);
	virtual ~MyGLWidget();

	cudaGraphicsResource_t evoTexResource(void) const
	{
		return m_evoTexResource;
	}

	void resizeTexture(int width, int height);

protected:
	void initializeGL(void);
	void resizeGL(void);
	void paintGL(void);

private:
	GLuint m_evoTex;
	GLuint m_evoTexSampler;
	GLuint m_fullscreenPgm;
	GLuint m_evoVAO;
	cudaGraphicsResource_t m_evoTexResource; ///< Registered to m_evoTex
};
