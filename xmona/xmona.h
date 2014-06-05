#ifndef XMONA_H
#define XMONA_H

#include <QtWidgets/QMainWindow>
#include "ui_xmona.h"
#include "myglwidget.h"

class xmona : public QMainWindow, Ui_xmonaClass
{
	Q_OBJECT

public:
	xmona(QWidget *parent = 0);
	~xmona();

protected:
	QOpenGLContext *m_platformContext;
	MyGLWidget *m_glWidget;

private:
	Ui::xmonaClass ui;
};

#endif // XMONA_H
