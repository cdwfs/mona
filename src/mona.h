#pragma once

#include "evogpupso.h"
#include "ui_mona.h"
#include "myglwidget.h"

#include <QtWidgets/QMainWindow>

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
	PsoConstants *m_psoConstants;
	PsoContext *m_psoContext;
private:
	Ui::monaWindow ui;
};
