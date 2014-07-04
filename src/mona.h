#pragma once

#include "evogpupso.h"
#include "ui_mona.h"
#include "myglwidget.h"

#include <QtWidgets/QMainWindow>
#include <QTimer>

class mona : public QMainWindow, Ui_monaWindow
{
	Q_OBJECT

public:
	mona(QWidget *parent = 0);
	~mona();

public slots:
	void loadRefImage(void);
	void togglePause(void);
	void iteratePso(void);

protected:
	QOpenGLContext *m_platformContext;
	MyGLWidget *m_glWidget;
	PsoConstants *m_psoConstants;
	PsoContext *m_psoContext;
	QTimer m_psoTimer;
	float m_bestPsnr; // Best PSNR seen to date
private:
	void initPso(int32_t imgWidth, int32_t imgHeight, const uint32_t *refPixels);
	Ui::monaWindow ui;
};
