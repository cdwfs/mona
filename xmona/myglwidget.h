#pragma once

#include <QGLWidget>

class MyGLWidget : public QGLWidget
{
	Q_OBJECT

public:
	MyGLWidget(QGLContext *context, QWidget *parent);

protected:
	void initializeGL(void);
	void resizeGL(void);
	void paintGL(void);

private:
	GLuint m_evoTex;
	GLuint m_evoTexSampler;
	GLuint m_fullscreenPgm;
	GLuint m_evoVAO;
};
