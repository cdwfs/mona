#include "zombolite.h"
#define GL_ARB_debug_output // prevents multiple definitions in Qt, which still uses this define instead of KHR_debug
#include "myglwidget.h"

#include <cassert>

static void CALLBACK DebugCallbackKHR(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar *message, const GLvoid *userParam)
{
	// Suppress some useless warnings
	switch(id)
	{
	case 131185: // NVIDIA: "static VBO will use VRAM for buffer operations" -- yes, yes it will.
	case 131218: // NVIDIA: "shader will be recompiled due to GL state mismatches" -- apparently NVIDIA wants shaders to be compiled with the same OpenGL state that will be present when they're run.
	case 1008:   // AMD: "program name is 1" (durp?)
		return;
	default:
		break;
	}

	(void)length;
	FILE *outFile = (FILE*)userParam;
	char *finalMsg = (char*)malloc(1024);
	ZomboLite::FormatDebugOutputKHR(finalMsg, 1024, source, type, id, severity, message);
	fprintf(outFile, "%s\n", finalMsg);	
#if defined(_MSC_VER)
	OutputDebugStringA(finalMsg);
	if (IsDebuggerPresent())
	{
		__debugbreak();
	}
#endif
	free(finalMsg);
}


MyGLWidget::MyGLWidget(QGLContext *context, QWidget *parent) : QGLWidget(context, parent)
{
}

void MyGLWidget::initializeGL(void)
{
	// Load OpenGL function pointers
	if (ogl_LoadFunctions() == ogl_LOAD_FAILED)
	{
		printf("Failed to load OpenGL function pointers\n");
		return;
	}
	// Hook up OpenGL debug output
	if (ogl_ext_KHR_debug)
	{
		glDebugMessageCallback(DebugCallbackKHR, stderr);
		glDebugMessageControl(GL_DONT_CARE,GL_DONT_CARE,GL_DONT_CARE,0,0,GL_TRUE);
		glEnable(GL_DEBUG_OUTPUT);
		glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
		printf("GL_KHR_debug is enabled\n");
	}

	// Set up rendering context
	glClearColor(1,0,0,1);
}

void MyGLWidget::resizeGL(void)
{
	// Set up viewport, projection, etc.
}

void MyGLWidget::paintGL(void)
{
	// draw scene
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}
