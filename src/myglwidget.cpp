#include "zombolite.h"
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

static const char *fullscreenTriVsCode = "\
#version 330\n\
out vec2 outVS_TexCoord0;\n\
void main(void)\n\
{\n\
	outVS_TexCoord0 = vec2((gl_VertexID<<1) & 2, gl_VertexID & 2);\n\
	gl_Position = vec4(outVS_TexCoord0 * vec2(2,-2) + vec2(-1,1), 0, 1);\n\
}\n";

static const char *fullscreenTriFsCode = "\
#version 330\n\
uniform sampler2DRect colorTex;\n\
in vec2 outVS_TexCoord0;\n\
out vec4 outFS_FragColor0;\n\
void main()\n\
{\n\
	outFS_FragColor0 = texture(colorTex, outVS_TexCoord0 * textureSize(colorTex));\n\
}\n";


MyGLWidget::MyGLWidget(QGLContext *context, QWidget *parent) : QGLWidget(context, parent)
{
}
MyGLWidget::~MyGLWidget()
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

	m_evoTex = 0;
#if 1
	ZomboLite::GenTexture(&m_evoTex, "Evolved Texture");
	glBindTexture(GL_TEXTURE_RECTANGLE, m_evoTex);
	if (ogl_ext_ARB_texture_storage)
	{
		glTexStorage2D(GL_TEXTURE_RECTANGLE, 1, GL_RGBA32F, this->width(), this->height());
		//glTexSubImage2D(GL_TEXTURE_RECTANGLE, 0, 0,0, this->width(), this->height(), GL_RGBA, GL_FLOAT, NULL);
	}
	else
	{
		glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, this->width(), this->height(), 0, GL_RGBA, GL_FLOAT, NULL);
	}
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
#else
	const char *evoTexFileName = "negalisa.dds";
	int32_t loadError = ZomboLite::LoadTextureFromDdsFile(evoTexFileName, &m_evoTex);
	ZOMBOLITE_ASSERT(loadError == 0, "Failed to load texture '%s' (error code: %d)", evoTexFileName, loadError);
#endif
	m_evoTexSampler = 0;
	ZomboLite::GenSampler(&m_evoTexSampler, "Evolved Texture Sampler");
	glSamplerParameteri(m_evoTexSampler, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glSamplerParameteri(m_evoTexSampler, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glSamplerParameteri(m_evoTexSampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(m_evoTexSampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

	m_fullscreenPgm = ZomboLite::CreateGlslProgram(fullscreenTriVsCode, fullscreenTriFsCode, "Fullscreen Triangle Program");
	ZOMBOLITE_ASSERT(m_fullscreenPgm != 0, "Failed to create GLSL program");

	ZomboLite::GenVertexArray(&m_evoVAO, "Evolved VAO");

	glViewport( 0, 0, this->width(), this->height() );

	// CUDA setup
	unsigned int glDeviceCount = 0;
	int cudaDeviceId = -1, glDeviceId = -1;
	CUDA_CHECK( cudaGetDevice(&cudaDeviceId) );
	CUDA_CHECK( cudaGLGetDevices(&glDeviceCount, &glDeviceId, 1, cudaGLDeviceListCurrentFrame) );
	ZOMBOLITE_ASSERT(cudaDeviceId == glDeviceId, "No OpenGL/CUDA devices found");
	cudaGraphicsResource *evoTexResource = NULL;
	CUDA_CHECK( cudaGraphicsGLRegisterImage(&evoTexResource, m_evoTex, GL_TEXTURE_RECTANGLE, cudaGraphicsRegisterFlagsWriteDiscard) );

	CUDA_CHECK( cudaGraphicsMapResources(1, &evoTexResource) );
	cudaArray *evoTexArray = NULL;
	size_t evoTexPixelsSize = this->width() * this->height() * sizeof(float)*4;
	CUDA_CHECK( cudaGraphicsSubResourceGetMappedArray(&evoTexArray, evoTexResource, 0, 0) );
	// Use d_evoTexPixels here
	float *pixels = (float*)malloc(evoTexPixelsSize);
	for(int iY=0; iY<this->height(); ++iY)
	{
		for(int iX=0; iX<this->width(); ++iX)
		{
			pixels[4*(iY*this->width() + iX) + 0] = 0.005f * iX;
			pixels[4*(iY*this->width() + iX) + 1] = 0.005f * iY;
			pixels[4*(iY*this->width() + iX) + 2] = 0;
			pixels[4*(iY*this->width() + iX) + 3] = 1.0f;
		}
	}
	CUDA_CHECK( cudaMemcpyToArray(evoTexArray, 0, 0, pixels, evoTexPixelsSize, cudaMemcpyHostToDevice) );
	free(pixels);
	// Stop using d_evoTexPixels here
	CUDA_CHECK( cudaGraphicsUnmapResources(1, &evoTexResource) );
}

void MyGLWidget::resizeGL(void)
{
	// Set up viewport, projection, etc.
	glViewport( 0, 0, this->width(), this->height() );
}

void MyGLWidget::paintGL(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Clear color buffer to black
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glDepthMask(GL_FALSE);
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);

	// Blit the canvas's pixels to the full-screen texture
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_RECTANGLE, m_evoTex);
	glBindSampler(0, m_evoTexSampler);
	glUseProgram(m_fullscreenPgm);
	glBindVertexArray(m_evoVAO);
	GLint fsTexLoc = glGetUniformLocation(m_fullscreenPgm, "colorTex");
	glUniform1i(fsTexLoc, 0);
	glDrawArrays(GL_TRIANGLES, 0, 3);
	glBindTexture(GL_TEXTURE_RECTANGLE, 0);
	glBindSampler(0, 0);
	glBindVertexArray(0);
	glEnable(GL_DEPTH_TEST);
	glUseProgram(0);
}
