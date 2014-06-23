#include "zombolite.h"

#include <map>
#include <string>
#include <vector>

void ZomboLite::FormatDebugOutputKHR(char outStr[], size_t outStrSize, GLenum source, GLenum type, GLuint id, GLenum severity, const char *msg)
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


typedef std::map<std::string, ZomboLite::VsSemantic> NameSemanticMap;
static NameSemanticMap g_vsAttrToSemanticMap;
typedef std::map<std::string, ZomboLite::FsDrawBuffer> NameDrawBufferMap;
static NameDrawBufferMap g_fsVarToDrawBufferMap;

void ZomboLite::RegisterVsInput(const std::string &attrName, ZomboLite::VsSemantic attrSemantic)
{
	// Validate attrSemantic against GL_MAX_VERTEX_ATTRIBS
	static GLuint maxVertexAttribIndex = 0;
	if (maxVertexAttribIndex == 0)
	{
		glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, (GLint*)&maxVertexAttribIndex);
	}
	ZOMBOLITE_ASSERT((GLuint)attrSemantic < maxVertexAttribIndex, "attrSemantic (%d) must be in the range [0..%d]", attrSemantic, maxVertexAttribIndex-1);
	if ((GLuint)attrSemantic >= maxVertexAttribIndex)
		return;
	// attrName can't start with the reserved prefix "gl_"
	ZOMBOLITE_ASSERT(!(attrName.size() >= 3 && attrName[0] == 'g' && attrName[1] == 'l' && attrName[2] == '_'), "attrName '%s' cannot start with 'gl_'", attrName.c_str());

	g_vsAttrToSemanticMap[attrName] = attrSemantic;
}

bool ZomboLite::GetVsInputSemantic(const std::string &attrName, ZomboLite::VsSemantic *outAttrSemantic)
{
	NameSemanticMap::const_iterator itor = g_vsAttrToSemanticMap.find(attrName);
	if (itor == g_vsAttrToSemanticMap.end())
	{
		return false;
	}
	*outAttrSemantic = (*itor).second;
	return true;
}

static void BindVsAttributeLocations(GLuint program)
{
	NameSemanticMap::const_iterator itor = g_vsAttrToSemanticMap.begin();
	while(itor != g_vsAttrToSemanticMap.end())
	{
		const std::string &name = (*itor).first;
		ZomboLite::VsSemantic semantic = (*itor).second;
		glBindAttribLocation(program, (GLuint)semantic, name.c_str());
		++itor;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////

void ZomboLite::RegisterFsOutput(const std::string &varName, ZomboLite::FsDrawBuffer drawBuffer)
{
	// Validate colorBufferIndex against GL_MAX_DRAW_BUFFERS
	static GLuint maxDrawBuffers = 0;
	if (maxDrawBuffers == 0)
	{
		glGetIntegerv(GL_MAX_DRAW_BUFFERS, (GLint*)&maxDrawBuffers);
	}
	if ((GLuint)drawBuffer >= maxDrawBuffers)
		return;
	// varName can't start with the reserved prefix "gl_"
	ZOMBOLITE_ASSERT(!(varName.size() >= 3 && varName[0] == 'g' && varName[1] == 'l' && varName[2] == '_'), "varName '%s' cannot start with 'gl_'", varName.c_str());

	g_fsVarToDrawBufferMap[varName] = drawBuffer;
}

bool ZomboLite::GetFsDrawBuffer(const std::string &varName, ZomboLite::FsDrawBuffer *outDrawBuffer)
{
	NameDrawBufferMap::const_iterator itor = g_fsVarToDrawBufferMap.find(varName);
	if (itor == g_fsVarToDrawBufferMap.end())
	{
		return false;
	}
	*outDrawBuffer = (*itor).second;
	return true;
}

static void BindFsFragDataLocations(GLuint program)
{
	NameDrawBufferMap::const_iterator itor = g_fsVarToDrawBufferMap.begin();
	while(itor != g_fsVarToDrawBufferMap.end())
	{
		const std::string &name = (*itor).first;
		ZomboLite::FsDrawBuffer dbIndex = (*itor).second;
		glBindFragDataLocation(program, (GLuint)dbIndex, name.c_str());
		++itor;
	}
}

void ZomboLite::GenVertexArray(GLuint *outVAO, const char *debugLabel)
{
	glGenVertexArrays(1, outVAO);
	if (*outVAO != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_VERTEX_ARRAY, *outVAO, -1, debugLabel);
	}
}
void ZomboLite::GenBuffer(GLuint *outBuf, const char *debugLabel)
{
	glGenBuffers(1, outBuf);
	if (*outBuf != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_BUFFER, *outBuf, -1, debugLabel);
	}
}
void ZomboLite::GenTexture(GLuint *outTex, const char *debugLabel)
{
	glGenTextures(1, outTex);
	if (*outTex != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_TEXTURE, *outTex, -1, debugLabel);
	}
}
void ZomboLite::GenSampler(GLuint *outSampler, const char *debugLabel)
{
	glGenSamplers(1, outSampler);
	if (*outSampler != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_SAMPLER, *outSampler, -1, debugLabel);
	}
}
GLuint ZomboLite::CreateProgram(const char *debugLabel)
{
	GLuint outProg = glCreateProgram();
	if (outProg != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_PROGRAM, outProg, -1, debugLabel);
	}
	return outProg;
}
GLuint ZomboLite::CreateShader(GLenum shaderType, const char *debugLabel)
{
	GLuint outShader = glCreateShader(shaderType);
	if (outShader != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_SHADER, outShader, -1, debugLabel);
	}
	return outShader;
}

////////////////////////////////////////////////////////////////////////////////////////

GLuint ZomboLite::CreateGlslProgram(const std::string &vsSource, const std::string &fsSource, const char *debugLabel)
{
	GLuint program = CreateProgram("Currently-loading program");

	// Vertex shader
	GLuint vertexShader = CreateShader(GL_VERTEX_SHADER, "Currently-loading Vertex Shader");
	const char *vsSourceStr = vsSource.c_str();
	glShaderSource(vertexShader, 1, &vsSourceStr, NULL);
	glCompileShader(vertexShader);
	int vsStatus = 0;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &vsStatus);
	if (vsStatus != GL_TRUE)
	{
		char infoLog[8192];
		glGetShaderInfoLog(vertexShader, 8192, NULL, infoLog);
		fprintf(stderr, "vertex shader compile log:\n%s\n", infoLog);
		glDeleteShader(vertexShader);
		glDeleteProgram(program);
		return 0;
	}
	glAttachShader(program, vertexShader);
	glDeleteShader(vertexShader);

	// Fragment shader
	GLuint fragmentShader = CreateShader(GL_FRAGMENT_SHADER, "Currently-loading Fragment Shader");
	const char *fsSourceStr = fsSource.c_str();
	glShaderSource(fragmentShader, 1, &fsSourceStr, NULL);
	glCompileShader(fragmentShader);
	int fsStatus = 0;
	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &fsStatus);
	if (fsStatus != GL_TRUE)
	{
		char infoLog[8192];
		glGetShaderInfoLog(fragmentShader, 8192, NULL, infoLog);
		fprintf(stderr, "fragment shader compile log:\n%s\n", infoLog);
		glDeleteShader(fragmentShader);
		glDeleteShader(vertexShader);
		glDeleteProgram(program);
		return 0;
	}
	glAttachShader(program, fragmentShader);
	glDeleteShader(fragmentShader);

	// Bind input attribute locations
	BindVsAttributeLocations(program);
	// Bind output variable locations
	BindFsFragDataLocations(program);

	// Link the program object
	glLinkProgram(program);
	int status = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &status);
	if (status != 1)
	{
		char infoLog[8192];
		glGetProgramInfoLog(program, 8192, NULL, infoLog);
		fprintf(stderr, "program link log:\n%s\n", infoLog);
		glDeleteProgram(program);
		return 0;
	}

	if (vertexShader != 0 && ogl_ext_KHR_debug)
		glObjectLabel(GL_SHADER, program, -1, (std::string("VS: ")+std::string(debugLabel)).c_str());
	if (fragmentShader != 0 && ogl_ext_KHR_debug)
		glObjectLabel(GL_SHADER, program, -1, (std::string("FS: ")+std::string(debugLabel)).c_str());
	if (program != 0 && ogl_ext_KHR_debug)
		glObjectLabel(GL_PROGRAM, program, -1, debugLabel);

	// The shaders can be safely deleted now; the program will hold a reference to them,
	// and they'll be deleted automatically when the program is deleted.
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return program;
}

GLuint ZomboLite::CreateGlslProgramFromFiles(const std::string &vsPath, const std::string &fsPath, const char *debugLabel)
{
	FILE *vsFile = NULL;
	errno_t vsOpenErr = fopen_s(&vsFile, vsPath.c_str(), "r");
	if (vsOpenErr != 0 || vsFile == NULL)
		return 0;
	FILE *fsFile = NULL;
	errno_t fsOpenErr = fopen_s(&fsFile, fsPath.c_str(), "r");
	if (fsOpenErr != 0 || fsFile == NULL)
	{
		fclose(vsFile);
		return 0;
	}

	// Load vertex shader source
	fseek(vsFile, 0, SEEK_END);
	size_t vsFileSize = ftell(vsFile);
	fseek(vsFile, 0, SEEK_SET);
	char *vsSource = (char*)malloc(vsFileSize+1);
	size_t vsSourceSize = fread(vsSource, 1, vsFileSize, vsFile);
	vsSource[vsSourceSize] = 0;
	fclose(vsFile);

	// Load fragment shader source
	fseek(fsFile, 0, SEEK_END);
	size_t fsFileSize = ftell(fsFile);
	fseek(fsFile, 0, SEEK_SET);
	char *fsSource = (char*)malloc(fsFileSize+1);
	size_t fsSourceSize = fread(fsSource, 1, fsFileSize, fsFile);
	fsSource[fsSourceSize] = 0;
	fclose(fsFile);

	GLuint program = ZomboLite::CreateGlslProgram(vsSource, fsSource, debugLabel);
	if (program == 0)
	{
		fprintf(stderr, "Failed to link shaders:\n\tVS=%s\n\tFS=%s\n", vsPath.c_str(), fsPath.c_str());
	}
	free(vsSource);
	free(fsSource);
	return program;
}

bool ZomboLite::ValidateGlslProgram(GLuint program)
{
	glValidateProgram(program);
	GLint status = 0;
	glGetProgramiv(program, GL_VALIDATE_STATUS, &status);
	if (status != 1)
	{
		char infoLog[8192];
		glGetProgramInfoLog(program, 8192, NULL, infoLog);
		fprintf(stderr, "program validation log:\n%s\n", infoLog);
	}
	return (status == 1);
}
