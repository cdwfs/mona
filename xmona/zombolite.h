#pragma once

#include "gl_core_3_3.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>

#include <cassert>
#include <cstdint>
#include <string>

// Use CUDA_CHECK() in host code to validate the result of CUDA functions that return an error.
#define ENABLE_CUDA_CHECK 1 // disable if you want CUDA_CHECK() to compile away
#if ENABLE_CUDA_CHECK
#define CUDA_CHECK(val) \
	do{ \
		cudaError err = (val); \
		if (err != cudaSuccess) { \
			char finalMsg[256]; \
			_snprintf_s(finalMsg, 255, "CUDA error in %s:\n%d:\t%s\n%s\n", __FILE__, __LINE__, #val, cudaGetErrorString(err)); \
			finalMsg[255] = 0; \
			OutputDebugStringA(finalMsg); \
			if (IsDebuggerPresent()) \
				__debugbreak(); \
			else \
				assert(err == cudaSuccess); \
		} \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) /* constant conditional */ \
	} while(0) \
	__pragma(warning(pop))
#else
#define CUDA_CHECK(val) (val)
#endif

// Custom assert macro that prints a formatted error message and breaks immediately from the calling code
// (instead of instead the code to _wassert)
#if defined(NDEBUG)
	#define ZOMBOLITE_ASSERT(cond,msg,...) (void)(1 ? (void)0 : ( (void)(cond) ) )
	#define ZOMBOLITE_ASSERT_RETURN(cond,retval,msg,...) if (!(cond)) { return (retval); }
#elif defined(_MSC_VER)
	#define ZOMBOLITE_ASSERT(cond,msg,...) \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) \
		do { \
			if (!(cond)) { \
				char *buffer = (char*)malloc(1024); \
				_snprintf_s(buffer, 1024, 1023, msg ## "\n", __VA_ARGS__); \
				buffer[1023] = 0; \
				fprintf(stderr, buffer); \
				OutputDebugStringA(buffer); \
				free(buffer); \
				IsDebuggerPresent() ? __debugbreak() : assert(cond); \
			} \
		} while(0) \
		__pragma(warning(pop))
	#define ZOMBOLITE_ASSERT_RETURN(cond,retval,msg,...) \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) \
		do { \
			if (!(cond)) { \
				char *buffer = (char*)malloc(1024); \
				_snprintf_s(buffer, 1024, 1023, msg ## "\n", __VA_ARGS__); \
				buffer[1023] = 0; \
				fprintf(stderr, buffer); \
				OutputDebugStringA(buffer); \
				free(buffer); \
				IsDebuggerPresent() ? __debugbreak() : assert(cond); \
				return (retval); \
			} \
		} while(0) \
		__pragma(warning(pop))
#else
	// Unsupported platform
	#define ZOMBOLITE_ASSERT(cond,msg,...) \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) \
		do { \
			if (!(cond)) { \
				fprintf(stderr, buffer, msg ## "\n", __VA_ARGS__); \
				assert(cond); \
			} \
		} while(0) \
		__pragma(warning(pop))
	#define ZOMBOLITE_ASSERT_RETURN(cond,retval,msg,...) \
		__pragma(warning(push)) \
		__pragma(warning(disable:4127)) \
		do { \
			if (!(cond)) { \
				fprintf(stderr, buffer, msg ## "\n", __VA_ARGS__); \
				assert(cond); \
				return (retval); \
			} \
		} while(0) \
		__pragma(warning(pop))
#endif
#define ZOMBOLITE_ERROR(msg,...) ZOMBOLITE_ASSERT(0, msg, __VA_ARGS__)
#define ZOMBOLITE_ERROR_RETURN(retval,msg,...) ZOMBOLITE_ASSERT_RETURN(0, retval, msg, __VA_ARGS__)

namespace ZomboLite
{
	//
	// Safe string-copying
	//
	template <size_t charCount>
	void strcpy_safe(char (&output)[charCount], const char *pSrc)
	{
		strncpy_s(output, pSrc, charCount);
		output[charCount-1] = 0;
	}
	template <size_t charCount>
	void strcat_safe(char (&output)[charCount], const char *pSrc)
	{
		strncat_s(output, pSrc, charCount);
		output[charCount-1] = 0;
	}

	//
	// String conversion utilities
	//
	bool FromUtf8(const std::string& utf8String, std::wstring *outString);
	bool ToUtf8(const std::wstring& wideString, std::string *outString);
	//
	// Directory path utilities
	//
	bool GetExecutablePath(std::string *outPath);
	bool SplitPathSuffix(const std::string &path, std::string *outFileBase, std::string *outFileSuffix);
	bool SplitPathDirectory(const std::string &path, std::string *outDirectory, std::string *outFilename);
	bool PushDirectory(const std::string &dir);
	void PopDirectory(void);

	//
	// OpenGL utilities
	//
	int32_t InitializeGLEW(void); // Call *after* the OpenGL context has been created!
	void FormatDebugOutputKHR(char outStr[], size_t outStrSize, GLenum source, GLenum type, GLuint id, GLenum severity, const char *msg);
	int32_t LoadTextureFromDdsFile(const std::string &ddsFileName, GLuint *outTex);
	int32_t LoadTextureFromDdsBuffer(const void *ddsBuffer, size_t ddsBufferSize, GLuint *outTex);
	enum VsSemantic
	{
		kVsSemanticPosition  = 0,
		kVsSemanticNormal    = 1,
		kVsSemanticTangent   = 2,
		kVsSemanticBitangent = 3,
		kVsSemanticColor     = 4,
		kVsSemanticTexCoord0 = 5,
		kVsSemanticTexCoord1 = 6,
		kVsSemanticTexCoord2 = 7,
		kVsSemanticTexCoord3 = 8,
		// Extend as needed, up to GL_MAX_VERTEX_ATTRIBS
	};
	enum FsDrawBuffer
	{
		kFsDrawBuffer0  = 0,
		kFsDrawBuffer1  = 1,
		kFsDrawBuffer2  = 2,
		kFsDrawBuffer3  = 3,
		kFsDrawBuffer4  = 4,
		kFsDrawBuffer5  = 5,
		kFsDrawBuffer6  = 6,
		kFsDrawBuffer7  = 7,
		kFsDrawBuffer8  = 8,
		kFsDrawBuffer9  = 9,
		kFsDrawBuffer10 = 10,
		kFsDrawBuffer11 = 11,
		kFsDrawBuffer12 = 12,
		kFsDrawBuffer13 = 13,
		kFsDrawBuffer14 = 14,
		kFsDrawBuffer15 = 15,
	};
	void RegisterVsInput(const std::string &attrName, VsSemantic attrSemantic);
	bool GetVsInputSemantic(const std::string &attrName, VsSemantic *outAttrSemantic);
	void RegisterFsOutput(const std::string &varName, FsDrawBuffer drawBuffer);
	bool GetFsDrawBuffer(const std::string &varName, FsDrawBuffer *outDrawBuffer);

	// Wrappers around various glGen*() functions which also call glObjectLabel() on the
	// newly-created object, if the KHR_debug extension is available.
	void GenVertexArray(GLuint *outVAO, const char *debugLabel); // wraps glGenVertexArrays()
	void GenBuffer(GLuint *outBuf, const char *debugLabel); // wraps glGenBuffers
	void GenTexture(GLuint *outTex, const char *debugLabel); // wraps glGenTextures()
	void GenSampler(GLuint *outSampler, const char *debugLabel); // wraps glGenSamplers()
	GLuint CreateProgram(const char *debugLabel); // wraps glCreateProgram()
	GLuint CreateShader(GLenum shaderType, const char *debugLabel); // wraps glCreateShader()

	// Creates a new GLSL program using the specified shaders.  If an error occurs, the return value
	// will be zero.  Otherwise, it will be the ID of the new program object.
	GLuint CreateGlslProgram(const std::string &vsSource, const std::string &fsSource, const char *debugLabel);
	GLuint CreateGlslProgramFromFiles(const std::string &vsPath, const std::string &fsPath, const char *debugLabel);

	/**
	 * Determines whether it is valid to use the specified GLSL program in the current OpenGL context.
	 * Call this function just before the glDraw*() call to verify that you've set everything up
	 * correctly.
	 * @param program The GLSL program to validate.
	 * @return True if the program is valid; false if not.
	 */
	bool ValidateGlslProgram(GLuint program);
}
