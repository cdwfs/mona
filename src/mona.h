#pragma once

#include <QtWidgets/QMainWindow>
#include "ui_mona.h"
#include "myglwidget.h"

class mona : public QMainWindow, Ui_monaWindow
{
	Q_OBJECT

public:
	mona(QWidget *parent = 0);
	~mona();

public slots:
	void loadRefImage(void);

protected:
	QOpenGLContext *m_platformContext;
	MyGLWidget *m_glWidget;

private:
	Ui::monaWindow ui;
};