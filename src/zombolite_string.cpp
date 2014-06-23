#include "zombolite.h"

#include <cstdint>
#include <string>
#include <vector>

#if defined(_MSC_VER)
	#include <windows.h>
	#include <direct.h>
	#define chdir _chdir
	#define getcwd _getcwd 
	#define stat _stat
	#define mkdir _mkdir
#endif

/////////////////////////////////////////
// String conversion utilities
/////////////////////////////////////////

bool ZomboLite::FromUtf8(const std::string& utf8String, std::wstring *outString)
{
	// Determine how long the output string needs to be
	int32_t wideSize = ::MultiByteToWideChar(CP_UTF8, 0, utf8String.c_str(), -1, NULL, 0);
	if (wideSize == ERROR_NO_UNICODE_TRANSLATION)
	{
		return false; // Invalid UTF-8 sequence
	}
	if (wideSize == 0)
	{
		return false;// Error in conversion
	}

	// Acatually perform the conversion
	std::vector<wchar_t> resultString(wideSize);
	int32_t convResult = ::MultiByteToWideChar(CP_UTF8, 0, utf8String.c_str(), -1, &resultString[0], wideSize);
	if (convResult != wideSize)
	{
		return false; // FAIL
	}
	*outString = std::wstring(&resultString[0]);

	return true;
}

bool ZomboLite::ToUtf8(const std::wstring& wideString, std::string *outString)
{
	// Determine how large the output needs to be
	int utf8Size = ::WideCharToMultiByte(CP_UTF8, 0, wideString.c_str(), -1, NULL, 0, NULL, NULL);
	if (utf8Size == 0)
	{
		return false; // Error in conversion
	}

	// Actually perform the conversion
	std::vector<char> resultString(utf8Size);
	int32_t convResult = ::WideCharToMultiByte(CP_UTF8, 0, wideString.c_str(), -1, &resultString[0], utf8Size, NULL, NULL);
	if (convResult != utf8Size)
	{
		return false; // FAIL
	}
	*outString = std::string(&resultString[0]);

	return true;
}

//////////////////////////////////////////////////////////
// Directory path utilities
//////////////////////////////////////////////////////////

bool ZomboLite::GetExecutablePath(std::string *outPath)
{
	if (outPath == NULL)
		return false;

#if _MSC_VER
	wchar_t exeName[1024];
	GetModuleFileName(NULL, exeName, 1023);
	exeName[1023] = 0;
	return ToUtf8(exeName, outPath);
#else
	// Not sure how to do this on POSIX systems
	return false;
#endif
}

bool ZomboLite::SplitPathSuffix(const std::string &path, std::string *outFileBase, std::string *outFileSuffix)
{
	size_t lastDot = path.find_last_of('.');
	if (lastDot == std::string::npos)
		return false;
	if (outFileBase)
		*outFileBase = path.substr(0, lastDot);
	if  (outFileSuffix)
		*outFileSuffix = path.substr(lastDot+1, std::string::npos);
	return true;
}

bool ZomboLite::SplitPathDirectory(const std::string &path, std::string *outDirectory, std::string *outFilename)
{
	size_t lastSep = path.find_last_of("/\\");
	if (lastSep == std::string::npos)
		return false;
	if (outDirectory)
		*outDirectory = path.substr(0, lastSep+1);
	if (outFilename)
		*outFilename = path.substr(lastSep+1, std::string::npos);
	return true;
}

static std::vector<std::string> g_directoryStack;
bool ZomboLite::PushDirectory(const std::string &dir)
{
	char buffer[512];
	getcwd(buffer, 512);
	if (chdir(dir.c_str()) != 0)
		return false;
	g_directoryStack.push_back(buffer);
	return true;
}

void ZomboLite::PopDirectory(void)
{
	if (g_directoryStack.empty())
		return;
	chdir(g_directoryStack.back().c_str());
	g_directoryStack.pop_back();
}

