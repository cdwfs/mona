#include "gl_core_3_3.h"
#define GL_ARB_debug_output // prevents multiple definitions
#include "myglwidget.h"

#include <cassert>

static void FormatDebugOutputKHR(char outStr[], size_t outStrSize, GLenum source, GLenum type, GLuint id, GLenum severity, const char *msg)
{
	char sourceStr[32];
	const char *sourceFmt = "UNDEFINED(0x%04X)";
	switch(source)
	{
	case GL_DEBUG_SOURCE_API:	          sourceFmt = "API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:	  sourceFmt = "WINDOW_SYSTEM"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: sourceFmt = "SHADER_COMPILER"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:	  sourceFmt = "THIRD_PARTY"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     sourceFmt = "APPLICATION"; break;
	case GL_DEBUG_SOURCE_OTHER:	          sourceFmt = "OTHER"; break;
	}
	_snprintf_s(sourceStr, 32, sourceFmt, source);

	char typeStr[32];
	const char *typeFmt = "UNDEFINED(0x%04X)";
	switch(type)
	{
	case GL_DEBUG_TYPE_ERROR:               typeFmt = "ERROR"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: typeFmt = "DEPRECATED_BEHAVIOR"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  typeFmt = "UNDEFINED_BEHAVIOR"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         typeFmt = "PORTABILITY"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         typeFmt = "PERFORMANCE"; break;
	case GL_DEBUG_TYPE_OTHER:               typeFmt = "OTHER"; break;
	case GL_DEBUG_TYPE_MARKER:              typeFmt = "MARKER"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          typeFmt = "PUSH_GROUP"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           typeFmt = "POP_GROUP"; break;
	}
	_snprintf_s(typeStr, 32, typeFmt, type);

	char severityStr[32];
	const char *severityFmt = "UNDEFINED";
	switch(severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         severityFmt = "HIGH";   break;
	case GL_DEBUG_SEVERITY_MEDIUM:       severityFmt = "MEDIUM"; break;
	case GL_DEBUG_SEVERITY_LOW:          severityFmt = "LOW";    break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: severityFmt = "NOTIFICATION"; break;
	}
	_snprintf_s(severityStr, 32, severityFmt, severity);

	_snprintf_s(outStr, outStrSize, outStrSize, "OpenGL: %s [source=%s type=%s severity=%s id=%d]", msg, sourceStr, typeStr, severityStr, id);
}
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
	FormatDebugOutputKHR(finalMsg, 1024, source, type, id, severity, message);
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
