#include "zombolite.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
using std::max;

#define DDS_PREFIX_MAGIC 0x20534444 // "DDS "

struct DdsPixelFormat
{
	uint32_t structSize;
	uint32_t flags;
	uint32_t code4;
	uint32_t numBitsRGB;
	uint32_t maskR;
	uint32_t maskG;
	uint32_t maskB;
	uint32_t maskA;
};

enum DdsPixelFormatFlags
{
	PF_FLAGS_CODE4     = 0x00000004,  // DDPF_FOURCC
	PF_FLAGS_RGB       = 0x00000040,  // DDPF_RGB
	PF_FLAGS_RGBA      = 0x00000041,  // DDPF_RGB | DDPF_ALPHAPIXELS
	PF_FLAGS_LUMINANCE = 0x00020000,  // DDPF_LUMINANCE
	PF_FLAGS_ALPHA     = 0x00000002,  // DDPF_ALPHA
};

#ifndef MAKE_CODE4
#define MAKE_CODE4(ch0, ch1, ch2, ch3)                              \
	((uint32_t)(uint8_t)(ch0) | ((uint32_t)(uint8_t)(ch1) << 8) | ((uint32_t)(uint8_t)(ch2) << 16) | ((uint32_t)(uint8_t)(ch3) << 24 ))
#endif //defined(MAKE_CODE4)

static const DdsPixelFormat PF_DXT1 = 
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','T','1'), 0, 0, 0, 0, 0 };
static const DdsPixelFormat PF_DXT2 = 
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','T','2'), 0, 0, 0, 0, 0 };
static const DdsPixelFormat PF_DXT3 = 
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','T','3'), 0, 0, 0, 0, 0 };
static const DdsPixelFormat PF_DXT4 = 
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','T','4'), 0, 0, 0, 0, 0 };
static const DdsPixelFormat PF_DXT5 = 
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','T','5'), 0, 0, 0, 0, 0 };

static const DdsPixelFormat PF_A8R8G8B8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGBA, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0xff000000 };

static const DdsPixelFormat PF_A8B8G8R8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGBA, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000 };

static const DdsPixelFormat PF_A1R5G5B5 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGBA, 0, 16, 0x00007c00, 0x000003e0, 0x0000001f, 0x00008000 };

static const DdsPixelFormat PF_A4R4G4B4 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGBA, 0, 16, 0x00000f00, 0x000000f0, 0x0000000f, 0x0000f000 };

static const DdsPixelFormat PF_R8G8B8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGB, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0x00000000 };

static const DdsPixelFormat PF_R8G8B8A8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGBA, 0, 32, 0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000 };

static const DdsPixelFormat PF_B8G8R8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGB, 0, 32, 0x00ff0000, 0x0000ff00, 0x000000ff, 0x00000000 };

static const DdsPixelFormat PF_R5G6B5 =
{ sizeof(DdsPixelFormat), PF_FLAGS_RGB, 0, 16, 0x0000f800, 0x000007e0, 0x0000001f, 0x00000000 };

static const DdsPixelFormat PF_L16 =
{ sizeof(DdsPixelFormat), PF_FLAGS_LUMINANCE, 0, 16, 0x0000ffff, 0x00000000, 0x00000000, 0x00000000 };

static const DdsPixelFormat PF_L8 =
{ sizeof(DdsPixelFormat), PF_FLAGS_LUMINANCE, 0, 8, 0x000000ff, 0x00000000, 0x00000000, 0x00000000 };

static const DdsPixelFormat PF_R32F =
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, 114, 0, 0, 0, 0, 0 };

static const DdsPixelFormat PF_R16FG16FB16FA16F =
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, 36, 0, 0, 0, 0, 0 };

static const DdsPixelFormat PF_R32FG32FB32FA32F =
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, 116, 0, 0, 0, 0, 0 };

// This indicates the DDS_HEADER_DXT10 extension is present (the format is in the extended header's dxFormat field)
static const DdsPixelFormat PF_DX10 =
{ sizeof(DdsPixelFormat), PF_FLAGS_CODE4, MAKE_CODE4('D','X','1','0'), 0, 0, 0, 0, 0 };

enum DdsHeaderFlag
{
	HEADER_FLAGS_CAPS        = 0x00000001,
	HEADER_FLAGS_HEIGHT      = 0x00000002,
	HEADER_FLAGS_WIDTH       = 0x00000004,
	HEADER_FLAGS_PITCH       = 0x00000008,
	HEADER_FLAGS_PIXELFORMAT = 0x00001000,
	HEADER_FLAGS_LINEARSIZE  = 0x00080000,
	HEADER_FLAGS_DEPTH       = 0x00800000,
	HEADER_FLAGS_TEXTURE     = 0x00001007,  // CAPS | HEIGHT | WIDTH | PIXELFORMAT 
	HEADER_FLAGS_MIPMAP      = 0x00020000,  
};

enum DdsSurfaceFlags
{
	SURFACE_FLAGS_TEXTURE = 0x00001000, // HEADER_FLAGS_TEXTURE
	SURFACE_FLAGS_MIPMAP  = 0x00400008, // COMPLEX | MIPMAP
	SURFACE_FLAGS_COMPLEX = 0x00000008, // COMPLEX
};

enum DdsCubemapFlags
{
	CUBEMAP_FLAG_ISCUBEMAP = 0x00000200, // CUBEMAP
	CUBEMAP_FLAG_POSITIVEX = 0x00000600, // CUBEMAP | POSITIVEX
	CUBEMAP_FLAG_NEGATIVEX = 0x00000a00, // CUBEMAP | NEGATIVEX
	CUBEMAP_FLAG_POSITIVEY = 0x00001200, // CUBEMAP | POSITIVEY
	CUBEMAP_FLAG_NEGATIVEY = 0x00002200, // CUBEMAP | NEGATIVEY
	CUBEMAP_FLAG_POSITIVEZ = 0x00004200, // CUBEMAP | POSITIVEZ
	CUBEMAP_FLAG_NEGATIVEZ = 0x00008200, // CUBEMAP | NEGATIVEZ
	CUBEMAP_FLAG_VOLUME    = 0x00200000, // VOLUME
};
#define CUBEMAP_FLAG_ALLFACES ( CUBEMAP_FLAG_ISCUBEMAP | CUBEMAP_FLAG_POSITIVEX | CUBEMAP_FLAG_NEGATIVEX |\
	CUBEMAP_FLAG_POSITIVEY | CUBEMAP_FLAG_NEGATIVEY |\
	CUBEMAP_FLAG_POSITIVEZ | CUBEMAP_FLAG_NEGATIVEZ )

enum DdsDimensions
{
	DIMENSIONS_UNKNOWN   = 0,
	DIMENSIONS_BUFFER    = 1,
	DIMENSIONS_TEXTURE1D = 2,
	DIMENSIONS_TEXTURE2D = 3,
	DIMENSIONS_TEXTURE3D = 4,
};

enum DxFormat
{
	DX_FORMAT_UNKNOWN	                  = 0,      
	DX_FORMAT_R32G32B32A32_TYPELESS       = 1,     
	DX_FORMAT_R32G32B32A32_FLOAT          = 2,     
	DX_FORMAT_R32G32B32A32_UINT           = 3,     
	DX_FORMAT_R32G32B32A32_SINT           = 4,     
	DX_FORMAT_R32G32B32_TYPELESS          = 5,     
	DX_FORMAT_R32G32B32_FLOAT             = 6,     
	DX_FORMAT_R32G32B32_UINT              = 7,     
	DX_FORMAT_R32G32B32_SINT              = 8,     
	DX_FORMAT_R16G16B16A16_TYPELESS       = 9,     
	DX_FORMAT_R16G16B16A16_FLOAT          = 10,
	DX_FORMAT_R16G16B16A16_UNORM          = 11,
	DX_FORMAT_R16G16B16A16_UINT           = 12,
	DX_FORMAT_R16G16B16A16_SNORM          = 13,
	DX_FORMAT_R16G16B16A16_SINT           = 14,
	DX_FORMAT_R32G32_TYPELESS             = 15,
	DX_FORMAT_R32G32_FLOAT                = 16,
	DX_FORMAT_R32G32_UINT                 = 17,
	DX_FORMAT_R32G32_SINT                 = 18,
	DX_FORMAT_R32G8X24_TYPELESS           = 19,
	DX_FORMAT_D32_FLOAT_S8X24_UINT        = 20,
	DX_FORMAT_R32_FLOAT_X8X24_TYPELESS    = 21,
	DX_FORMAT_X32_TYPELESS_G8X24_UINT     = 22,
	DX_FORMAT_R10G10B10A2_TYPELESS        = 23,
	DX_FORMAT_R10G10B10A2_UNORM           = 24,
	DX_FORMAT_R10G10B10A2_UINT            = 25,
	DX_FORMAT_R11G11B10_FLOAT             = 26,
	DX_FORMAT_R8G8B8A8_TYPELESS           = 27,
	DX_FORMAT_R8G8B8A8_UNORM              = 28,
	DX_FORMAT_R8G8B8A8_UNORM_SRGB         = 29,
	DX_FORMAT_R8G8B8A8_UINT               = 30,
	DX_FORMAT_R8G8B8A8_SNORM              = 31,
	DX_FORMAT_R8G8B8A8_SINT               = 32,
	DX_FORMAT_R16G16_TYPELESS             = 33,
	DX_FORMAT_R16G16_FLOAT                = 34,
	DX_FORMAT_R16G16_UNORM                = 35,
	DX_FORMAT_R16G16_UINT                 = 36,
	DX_FORMAT_R16G16_SNORM                = 37,
	DX_FORMAT_R16G16_SINT                 = 38,
	DX_FORMAT_R32_TYPELESS                = 39,
	DX_FORMAT_D32_FLOAT                   = 40,
	DX_FORMAT_R32_FLOAT                   = 41,
	DX_FORMAT_R32_UINT                    = 42,
	DX_FORMAT_R32_SINT                    = 43,
	DX_FORMAT_R24G8_TYPELESS              = 44,
	DX_FORMAT_D24_UNORM_S8_UINT           = 45,
	DX_FORMAT_R24_UNORM_X8_TYPELESS       = 46,
	DX_FORMAT_X24_TYPELESS_G8_UINT        = 47,
	DX_FORMAT_R8G8_TYPELESS               = 48,
	DX_FORMAT_R8G8_UNORM                  = 49,
	DX_FORMAT_R8G8_UINT                   = 50,
	DX_FORMAT_R8G8_SNORM                  = 51,
	DX_FORMAT_R8G8_SINT                   = 52,
	DX_FORMAT_R16_TYPELESS                = 53,
	DX_FORMAT_R16_FLOAT                   = 54,
	DX_FORMAT_D16_UNORM                   = 55,
	DX_FORMAT_R16_UNORM                   = 56,
	DX_FORMAT_R16_UINT                    = 57,
	DX_FORMAT_R16_SNORM                   = 58,
	DX_FORMAT_R16_SINT                    = 59,
	DX_FORMAT_R8_TYPELESS                 = 60,
	DX_FORMAT_R8_UNORM                    = 61,
	DX_FORMAT_R8_UINT                     = 62,
	DX_FORMAT_R8_SNORM                    = 63,
	DX_FORMAT_R8_SINT                     = 64,
	DX_FORMAT_A8_UNORM                    = 65,
	DX_FORMAT_R1_UNORM                    = 66,
	DX_FORMAT_R9G9B9E5_SHAREDEXP          = 67,
	DX_FORMAT_R8G8_B8G8_UNORM             = 68,
	DX_FORMAT_G8R8_G8B8_UNORM             = 69,
	DX_FORMAT_BC1_TYPELESS                = 70,
	DX_FORMAT_BC1_UNORM                   = 71,
	DX_FORMAT_BC1_UNORM_SRGB              = 72,
	DX_FORMAT_BC2_TYPELESS                = 73,
	DX_FORMAT_BC2_UNORM                   = 74,
	DX_FORMAT_BC2_UNORM_SRGB              = 75,
	DX_FORMAT_BC3_TYPELESS                = 76,
	DX_FORMAT_BC3_UNORM                   = 77,
	DX_FORMAT_BC3_UNORM_SRGB              = 78,
	DX_FORMAT_BC4_TYPELESS                = 79,
	DX_FORMAT_BC4_UNORM                   = 80,
	DX_FORMAT_BC4_SNORM                   = 81,
	DX_FORMAT_BC5_TYPELESS                = 82,
	DX_FORMAT_BC5_UNORM                   = 83,
	DX_FORMAT_BC5_SNORM                   = 84,
	DX_FORMAT_B5G6R5_UNORM                = 85,
	DX_FORMAT_B5G5R5A1_UNORM              = 86,
	DX_FORMAT_B8G8R8A8_UNORM              = 87,
	DX_FORMAT_B8G8R8X8_UNORM              = 88,
	DX_FORMAT_R10G10B10_XR_BIAS_A2_UNORM  = 89,
	DX_FORMAT_B8G8R8A8_TYPELESS           = 90,
	DX_FORMAT_B8G8R8A8_UNORM_SRGB         = 91,
	DX_FORMAT_B8G8R8X8_TYPELESS           = 92,
	DX_FORMAT_B8G8R8X8_UNORM_SRGB         = 93,
	DX_FORMAT_BC6H_TYPELESS               = 94,
	DX_FORMAT_BC6H_UF16                   = 95,
	DX_FORMAT_BC6H_SF16                   = 96,
	DX_FORMAT_BC7_TYPELESS                = 97,
	DX_FORMAT_BC7_UNORM                   = 98,
	DX_FORMAT_BC7_UNORM_SRGB              = 99,
};

#define ISBITMASK( r,g,b,a ) ( pf.maskR == r && pf.maskG == g && pf.maskB == b && pf.maskA == a )
static bool DDSPFtoGLFormat( const DdsPixelFormat& pf, uint32_t *outBlockSize, GLenum *outGlFormat, GLenum *outGlInternalFormat, GLenum *outGlType )
{
	if( pf.flags & PF_FLAGS_RGBA )
	{
		switch (pf.numBitsRGB)
		{
		case 32:
			*outBlockSize = 4;
			if( ISBITMASK(0x00ff0000,0x0000ff00,0x000000ff,0xff000000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; return true;
			}
			if( ISBITMASK(0x00ff0000,0x0000ff00,0x000000ff,0x00000000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; return true; // BGRX
			}
			if( ISBITMASK(0x000000ff,0x0000ff00,0x00ff0000,0xff000000) )
			{
				*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; return true;
			}
			if( ISBITMASK(0x000000ff,0x0000ff00,0x00ff0000,0x00000000) )
			{
				*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; return true; // RGBX
			}

			// Note that many common DDS reader/writers swap the
			// the RED/BLUE masks for 10:10:10:2 formats. We assumme
			// below that the 'correct' header mask is being used
			if( ISBITMASK(0x3ff00000,0x000ffc00,0x000003ff,0xc0000000) )
			{
				*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_INT_10_10_10_2; *outGlInternalFormat = GL_RGB10_A2; return true;
			}
			if( ISBITMASK(0x000003ff,0x000ffc00,0x3ff00000,0xc0000000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_INT_10_10_10_2; *outGlInternalFormat = GL_RGB10_A2; return true;
			}

			if( ISBITMASK(0xffffffff,0x00000000,0x00000000,0x00000000) )
			{
				*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_INT; *outGlInternalFormat = GL_RED; return true;
			}
			if( ISBITMASK(0x0000ffff,0xffff0000,0x00000000,0x00000000) )
			{
				*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RG16; return true;
			}
			break;

		case 24:
			*outBlockSize = 3;
			break;

		case 16:
			*outBlockSize = 2;
			if( ISBITMASK(0x0000f800,0x000007e0,0x0000001f,0x00000000) )
			{
				*outGlFormat = GL_RGB;	*outGlType = GL_UNSIGNED_SHORT_5_6_5; *outGlInternalFormat = GL_RGB; return true;
			}
			if( ISBITMASK(0x00007c00,0x000003e0,0x0000001f,0x00008000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_SHORT_5_5_5_1; *outGlInternalFormat = GL_RGB5_A1; return true;
			}
			if( ISBITMASK(0x00007c00,0x000003e0,0x0000001f,0x00000000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_SHORT_5_5_5_1; *outGlInternalFormat = GL_RGB5_A1; return true; // BGRX
			}
			if( ISBITMASK(0x00000f00,0x000000f0,0x0000000f,0x0000f000) )
			{
				*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_SHORT_4_4_4_4; *outGlInternalFormat = GL_RGBA4; return true;
			}
			if( ISBITMASK(0x00000f00,0x000000f0,0x0000000f,0x00000000) )
			{
				*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_SHORT_4_4_4_4; *outGlInternalFormat = GL_RGBA4; return true; // BGRX
			}
			break;

		case 8:
			*outBlockSize = 1;
 			if( ISBITMASK(0x000000ff,0x00000000,0x00000000,0x00000000) )
			{
				*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RED; return true;
			}
			break;
		}
	}
	else if( pf.flags & PF_FLAGS_LUMINANCE )
	{
		if( 8 == pf.numBitsRGB )
		{
			if( ISBITMASK(0x000000ff,0x00000000,0x00000000,0x00000000) ) // L8
			{
				*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RED; *outBlockSize = 1; return true;
			}
		}

		if( 16 == pf.numBitsRGB )
		{
			if( ISBITMASK(0x0000ffff,0x00000000,0x00000000,0x00000000) ) // L16
			{
				*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RED; *outBlockSize = 2; return true;
			}
		}
	}
	else if( pf.flags & PF_FLAGS_ALPHA )
	{
	}
	else if( pf.flags & PF_FLAGS_CODE4 )
	{
		if( MAKE_CODE4( 'D', 'X', 'T', '1' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT; *outBlockSize = 8; return true;
		}
		if( MAKE_CODE4( 'D', 'X', 'T', '2' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; *outBlockSize = 16; return true;
		}
		if( MAKE_CODE4( 'D', 'X', 'T', '3' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; *outBlockSize = 16; return true;
		}
		if( MAKE_CODE4( 'D', 'X', 'T', '4' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; *outBlockSize = 16; return true;
		}
		if( MAKE_CODE4( 'D', 'X', 'T', '5' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; *outBlockSize = 16; return true;
		}
		if( MAKE_CODE4( 'A', 'T', 'I', '1' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RED_RGTC1; *outBlockSize = 8; return true;
		}
		if( MAKE_CODE4( 'A', 'T', 'I', '2' ) == pf.code4 )
		{
			*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RG_RGTC2; *outBlockSize = 16; return true;
		}

		// Certain values are hard-coded into the FourCC field for specific formats
		if ( 110 == pf.code4 )
		{
			*outGlFormat = GL_RGBA;	*outGlType = GL_SHORT; *outGlInternalFormat = GL_RGBA16; *outBlockSize = 8; return true;
		}
		if ( 111 == pf.code4 )
		{
			*outGlFormat = GL_RED;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_R16F; *outBlockSize = 2; return true;
		}
		if ( 112 == pf.code4 )
		{
			*outGlFormat = GL_RG;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_RG16F; *outBlockSize = 4; return true;
		}
		if ( 113 == pf.code4 )
		{
			*outGlFormat = GL_RGBA;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_RGBA16F; *outBlockSize = 8; return true;
		}
		if ( 114 == pf.code4 )
		{
			*outGlFormat = GL_RED;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_R32F; *outBlockSize = 4; return true;
		}
		if ( 115 == pf.code4 )
		{
			*outGlFormat = GL_RG;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_RG32F; *outBlockSize = 8; return true;
		}
		if ( 116 == pf.code4 )
		{
			*outGlFormat = GL_RGBA;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_RGBA32F; *outBlockSize = 16; return true;
		}
		if ( 36 == pf.code4 )
		{
			*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RGBA16; *outBlockSize = 8; return true;
		}
	}

	*outGlFormat = GL_INVALID_ENUM;
	*outGlInternalFormat = GL_INVALID_ENUM;
	*outGlType = GL_INVALID_ENUM;
	*outBlockSize = 0;
	return false; // unknown/unsupported DDSPF format
}

static bool DXtoGLFormat(DxFormat dxFmt, uint32_t *outBlockSize, GLenum *outGlFormat, GLenum *outGlInternalFormat, GLenum *outGlType)
{
	switch(dxFmt)
	{
	case DX_FORMAT_UNKNOWN:
	case DX_FORMAT_R32G32B32A32_TYPELESS:
	case DX_FORMAT_R32G32B32_TYPELESS:
	case DX_FORMAT_R16G16B16A16_TYPELESS:
	case DX_FORMAT_R32G32_TYPELESS:
	case DX_FORMAT_R32G8X24_TYPELESS:
	case DX_FORMAT_R32_FLOAT_X8X24_TYPELESS:
	case DX_FORMAT_X32_TYPELESS_G8X24_UINT:
	case DX_FORMAT_R10G10B10A2_TYPELESS:
	case DX_FORMAT_R8G8B8A8_TYPELESS:
	case DX_FORMAT_R16G16_TYPELESS:
	case DX_FORMAT_R32_TYPELESS:
	case DX_FORMAT_R24G8_TYPELESS:
	case DX_FORMAT_R24_UNORM_X8_TYPELESS:
	case DX_FORMAT_X24_TYPELESS_G8_UINT:
	case DX_FORMAT_R8G8_TYPELESS:
	case DX_FORMAT_R16_TYPELESS:
	case DX_FORMAT_R8_TYPELESS:
	case DX_FORMAT_D32_FLOAT_S8X24_UINT:
	case DX_FORMAT_D24_UNORM_S8_UINT:
	case DX_FORMAT_R9G9B9E5_SHAREDEXP:
	case DX_FORMAT_R8G8_B8G8_UNORM:
	case DX_FORMAT_G8R8_G8B8_UNORM:
	case DX_FORMAT_R10G10B10_XR_BIAS_A2_UNORM:
	case DX_FORMAT_B8G8R8A8_TYPELESS:
	case DX_FORMAT_B8G8R8X8_TYPELESS:
	case DX_FORMAT_R1_UNORM:
	case DX_FORMAT_A8_UNORM:
		break;
	case DX_FORMAT_R32G32B32A32_FLOAT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_RGBA32F; *outBlockSize = 16; return true;
	case DX_FORMAT_R32G32B32A32_UINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_INT; *outGlInternalFormat = GL_RGBA32UI; *outBlockSize = 16; return true;
	case DX_FORMAT_R32G32B32A32_SINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_INT; *outGlInternalFormat = GL_RGBA32I; *outBlockSize = 16; return true;
	case DX_FORMAT_R32G32B32_FLOAT:
		*outGlFormat = GL_RGB;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_RGB32F; *outBlockSize = 12; return true;
	case DX_FORMAT_R32G32B32_UINT:
		*outGlFormat = GL_RGB;	*outGlType = GL_UNSIGNED_INT; *outGlInternalFormat = GL_RGB32UI; *outBlockSize = 12; return true;
	case DX_FORMAT_R32G32B32_SINT:
		*outGlFormat = GL_RGB;	*outGlType = GL_INT; *outGlInternalFormat = GL_RGB32I; *outBlockSize = 12; return true;
	case DX_FORMAT_R16G16B16A16_FLOAT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_RGBA16F; *outBlockSize = 8; return true;
	case DX_FORMAT_R16G16B16A16_UNORM:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RGBA16; *outBlockSize = 8; return true;
	case DX_FORMAT_R16G16B16A16_UINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RGBA16UI; *outBlockSize = 8; return true;
	case DX_FORMAT_R16G16B16A16_SNORM:
		*outGlFormat = GL_RGBA;	*outGlType = GL_SHORT; *outGlInternalFormat = GL_RGBA16; *outBlockSize = 8; return true;
	case DX_FORMAT_R16G16B16A16_SINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_SHORT; *outGlInternalFormat = GL_RGBA16I; *outBlockSize = 8; return true;
	case DX_FORMAT_R32G32_FLOAT:
		*outGlFormat = GL_RG;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_RG32F; *outBlockSize = 8; return true;
	case DX_FORMAT_R32G32_UINT:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_INT; *outGlInternalFormat = GL_RG32UI; *outBlockSize = 8; return true;
	case DX_FORMAT_R32G32_SINT:
		*outGlFormat = GL_RG;	*outGlType = GL_INT; *outGlInternalFormat = GL_RG32I; *outBlockSize = 8; return true;
	case DX_FORMAT_R10G10B10A2_UNORM:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_INT_10_10_10_2; *outGlInternalFormat = GL_RGB10_A2; *outBlockSize = 4; return true;
	case DX_FORMAT_R10G10B10A2_UINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_INT_10_10_10_2; *outGlInternalFormat = GL_RGB10_A2UI; *outBlockSize = 4; return true;
	case DX_FORMAT_R11G11B10_FLOAT:
		*outGlFormat = GL_RGB;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_R11F_G11F_B10F; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8B8A8_UNORM:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8B8A8_UNORM_SRGB:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_SRGB8_ALPHA8; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8B8A8_UINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA8UI; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8B8A8_SNORM:
		*outGlFormat = GL_RGBA;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_RGBA; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8B8A8_SINT:
		*outGlFormat = GL_RGBA;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_RGBA8I; *outBlockSize = 4; return true;
	case DX_FORMAT_R16G16_FLOAT:
		*outGlFormat = GL_RG;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_RG16F; *outBlockSize = 4; return true;
	case DX_FORMAT_R16G16_UNORM:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RG16; *outBlockSize = 4; return true;
	case DX_FORMAT_R16G16_UINT:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RG16UI; *outBlockSize = 4; return true;
	case DX_FORMAT_R16G16_SNORM:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RG; *outBlockSize = 4; return true;
	case DX_FORMAT_R16G16_SINT:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RG16I; *outBlockSize = 4; return true;
	case DX_FORMAT_D32_FLOAT:
		*outGlFormat = GL_DEPTH_COMPONENT; *outGlType = GL_FLOAT; *outGlInternalFormat = GL_R32F; *outBlockSize = 4; return true;
	case DX_FORMAT_R32_FLOAT:
		*outGlFormat = GL_RED;	*outGlType = GL_FLOAT; *outGlInternalFormat = GL_R32F; *outBlockSize = 4; return true;
	case DX_FORMAT_R32_UINT:
		*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_INT; *outGlInternalFormat = GL_R32UI; *outBlockSize = 4; return true;
	case DX_FORMAT_R32_SINT:
		*outGlFormat = GL_RED;	*outGlType = GL_INT; *outGlInternalFormat = GL_R32I; *outBlockSize = 4; return true;
	case DX_FORMAT_R8G8_UNORM:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RG; *outBlockSize = 2; return true;
	case DX_FORMAT_R8G8_UINT:
		*outGlFormat = GL_RG;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RG8UI; *outBlockSize = 2; return true;
	case DX_FORMAT_R8G8_SNORM:
		*outGlFormat = GL_RG;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_RG; *outBlockSize = 2; return true;
	case DX_FORMAT_R8G8_SINT:
		*outGlFormat = GL_RG;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_RG8I; *outBlockSize = 2; return true;
	case DX_FORMAT_R16_FLOAT:
		*outGlFormat = GL_RED;	*outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_R16F; *outBlockSize = 2; return true;
	case DX_FORMAT_D16_UNORM:
		*outGlFormat = GL_DEPTH_COMPONENT; *outGlType = GL_HALF_FLOAT; *outGlInternalFormat = GL_R16F; *outBlockSize = 2; return true;
	case DX_FORMAT_R16_UNORM:
		*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_RED; *outBlockSize = 2; return true;
	case DX_FORMAT_R16_UINT:
		*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_SHORT; *outGlInternalFormat = GL_R16UI; *outBlockSize = 2; return true;
	case DX_FORMAT_R16_SNORM:
		*outGlFormat = GL_RED;	*outGlType = GL_SHORT; *outGlInternalFormat = GL_RED; *outBlockSize = 2; return true;
	case DX_FORMAT_R16_SINT:
		*outGlFormat = GL_RED;	*outGlType = GL_SHORT; *outGlInternalFormat = GL_R16I; *outBlockSize = 2; return true;
	case DX_FORMAT_R8_UNORM:
		*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RED; *outBlockSize = 1; return true;
	case DX_FORMAT_R8_UINT:
		*outGlFormat = GL_RED;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_R8UI; *outBlockSize = 1; return true;
	case DX_FORMAT_R8_SNORM:
		*outGlFormat = GL_RED;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_RED; *outBlockSize = 1; return true;
	case DX_FORMAT_R8_SINT:
		*outGlFormat = GL_RED;	*outGlType = GL_BYTE; *outGlInternalFormat = GL_R8I; *outBlockSize = 1; return true;
	case DX_FORMAT_BC1_TYPELESS:
	case DX_FORMAT_BC1_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT; *outBlockSize = 8; return true;
	case DX_FORMAT_BC1_UNORM_SRGB:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT; *outBlockSize = 8; return true; // loses SRGB
	case DX_FORMAT_BC2_TYPELESS:
	case DX_FORMAT_BC2_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; *outBlockSize = 16; return true;
	case DX_FORMAT_BC2_UNORM_SRGB:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT; *outBlockSize = 16; return true; // loses SRGB
	case DX_FORMAT_BC3_TYPELESS:
	case DX_FORMAT_BC3_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; *outBlockSize = 16; return true;
	case DX_FORMAT_BC3_UNORM_SRGB:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT; *outBlockSize = 16; return true; // loses SRGB
	case DX_FORMAT_BC4_TYPELESS:
	case DX_FORMAT_BC4_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RED_RGTC1; *outBlockSize = 8; return true;
	case DX_FORMAT_BC4_SNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_SIGNED_RED_RGTC1; *outBlockSize = 8; return true;
	case DX_FORMAT_BC5_TYPELESS:
	case DX_FORMAT_BC5_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RG_RGTC2; *outBlockSize = 16; return true;
	case DX_FORMAT_BC5_SNORM:
		ZOMBOLITE_ASSERT(ogl_ext_EXT_texture_compression_s3tc, "S3TC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_SIGNED_RG_RGTC2; *outBlockSize = 16; return true;
	case DX_FORMAT_BC6H_UF16:
		ZOMBOLITE_ASSERT(ogl_ext_ARB_texture_compression_bptc, "BPTC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_ARB; *outBlockSize = 16; return true;
	case DX_FORMAT_BC6H_SF16:
		ZOMBOLITE_ASSERT(ogl_ext_ARB_texture_compression_bptc, "BPTC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT_ARB; *outBlockSize = 16; return true;
	case DX_FORMAT_BC7_UNORM:
		ZOMBOLITE_ASSERT(ogl_ext_ARB_texture_compression_bptc, "BPTC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_RGBA_BPTC_UNORM_ARB; *outBlockSize = 16; return true;
	case DX_FORMAT_BC7_UNORM_SRGB:
		ZOMBOLITE_ASSERT(ogl_ext_ARB_texture_compression_bptc, "BPTC texture compression is not supported by this system! The texture will be created correctly, but it cannot be used.");
		*outGlFormat = GL_INVALID_ENUM;	*outGlType = GL_INVALID_ENUM; *outGlInternalFormat = GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM_ARB; *outBlockSize = 16; return true;
	case DX_FORMAT_B5G6R5_UNORM:
		*outGlFormat = GL_BGR;	*outGlType = GL_UNSIGNED_SHORT_5_6_5; *outGlInternalFormat = GL_RGB; *outBlockSize = 2; return true;
	case DX_FORMAT_B5G5R5A1_UNORM:
		*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_SHORT_5_5_5_1; *outGlInternalFormat = GL_RGBA; *outBlockSize = 2; return true;
	case DX_FORMAT_B8G8R8A8_UNORM:
	case DX_FORMAT_B8G8R8X8_UNORM:
		*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_RGBA; *outBlockSize = 4; return true;
	case DX_FORMAT_B8G8R8A8_UNORM_SRGB:
	case DX_FORMAT_B8G8R8X8_UNORM_SRGB:
		*outGlFormat = GL_BGRA;	*outGlType = GL_UNSIGNED_BYTE; *outGlInternalFormat = GL_SRGB8_ALPHA8; *outBlockSize = 4; return true;
	default:
		break;
	}

	*outGlFormat = GL_INVALID_ENUM;
	*outGlInternalFormat = GL_INVALID_ENUM;
	*outGlType = GL_INVALID_ENUM;
	*outBlockSize = 4;
	return false; // unknown/unsupported format
}

struct DdsHeader
{
	uint32_t structSize;
	DdsHeaderFlag flags;
	uint32_t height;
	uint32_t width;
	uint32_t pitchOrLinearSize;
	uint32_t depth; // only if HEADER_FLAGS_VOLUME is set in flags
	uint32_t mipCount;
	uint32_t unused1[11];
	DdsPixelFormat pixelFormat;
	uint32_t caps;
	uint32_t caps2;
	uint32_t unused2[3];
};

struct DdsHeader10
{
	DxFormat dxgiFormat;
	DdsDimensions resourceDimension;
	uint32_t flag;
	uint32_t arraySize;
	uint32_t unused;
};

static bool ContainsCompressedTexture(const DdsHeader *header, const DdsHeader10 *header10)
{
	if (header10 != NULL)
	{
		switch(header10->dxgiFormat)
		{
		case DX_FORMAT_BC1_TYPELESS:
		case DX_FORMAT_BC1_UNORM:
		case DX_FORMAT_BC1_UNORM_SRGB:
		case DX_FORMAT_BC2_TYPELESS:
		case DX_FORMAT_BC2_UNORM:
		case DX_FORMAT_BC2_UNORM_SRGB:
		case DX_FORMAT_BC3_TYPELESS:
		case DX_FORMAT_BC3_UNORM:
		case DX_FORMAT_BC3_UNORM_SRGB:
		case DX_FORMAT_BC4_TYPELESS:
		case DX_FORMAT_BC4_UNORM:
		case DX_FORMAT_BC4_SNORM:
		case DX_FORMAT_BC5_TYPELESS:
		case DX_FORMAT_BC5_UNORM:
		case DX_FORMAT_BC5_SNORM:
		case DX_FORMAT_BC6H_UF16:
		case DX_FORMAT_BC6H_SF16:
		case DX_FORMAT_BC7_UNORM:
		case DX_FORMAT_BC7_UNORM_SRGB:
			return true;
		default:
			return false;
		}
	}
	else if( header->pixelFormat.flags & PF_FLAGS_CODE4 )
	{
		return
			MAKE_CODE4( 'D', 'X', 'T', '1' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'D', 'X', 'T', '2' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'D', 'X', 'T', '3' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'D', 'X', 'T', '4' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'D', 'X', 'T', '5' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'A', 'T', 'I', '1' ) == header->pixelFormat.code4 ||
			MAKE_CODE4( 'A', 'T', 'I', '2' ) == header->pixelFormat.code4;
	}
	return false;
}

int32_t ZomboLite::LoadTextureFromDdsFile(const std::string &ddsFileName, GLuint *outTex)
{
	FILE *ddsFile = NULL;
	errno_t fopenErr = fopen_s(&ddsFile, ddsFileName.c_str(), "rb");
	if (fopenErr != 0 || ddsFile == NULL)
		return -1; // File load error
	fseek(ddsFile, 0, SEEK_END);
	size_t ddsFileSize = ftell(ddsFile);
	fseek(ddsFile, 0, SEEK_SET);
	if (ddsFileSize < sizeof(DdsHeader)+sizeof(uint32_t))
	{
		fclose(ddsFile);
		return -2; // File too small to contain a valid DDS
	}
	uint8_t *ddsFileData = (uint8_t*)malloc(ddsFileSize);
	if (fread(ddsFileData, ddsFileSize, 1, ddsFile) != 1)
	{
		fclose(ddsFile);
		free(ddsFileData);
		return -3; // fread size mismatch
	}
	fclose(ddsFile);

	int32_t retval = ZomboLite::LoadTextureFromDdsBuffer(ddsFileData, ddsFileSize, outTex);
	free(ddsFileData);
	if (*outTex != 0 && ogl_ext_KHR_debug)
	{
		glObjectLabel(GL_TEXTURE, *outTex, -1, ddsFileName.c_str());
	}
	return retval;
}

int32_t ZomboLite::LoadTextureFromDdsBuffer(const void *ddsBuffer, size_t ddsBufferSize, GLuint *outTex)
{
	const uint8_t *ddsBytes = (const uint8_t*)ddsBuffer;

	// Check magic number and header validity
	const uint32_t *magic = (const uint32_t*)ddsBytes;
	if (*magic != DDS_PREFIX_MAGIC)
	{
		return -4; // Incorrect magic number
	}
	const DdsHeader *header = (const DdsHeader*)(ddsBytes + sizeof(uint32_t));
	if (header->structSize != sizeof(DdsHeader) || header->pixelFormat.structSize != sizeof(DdsPixelFormat))
	{
		return -5; // Incorrect header size
	}
	if ((header->flags & (HEADER_FLAGS_WIDTH | HEADER_FLAGS_HEIGHT)) != (HEADER_FLAGS_WIDTH | HEADER_FLAGS_HEIGHT))
	{
		// technically DDSD_CAPS and DDSD_PIXELFORMAT are required as well, but their absence is so widespread that they can't be relied upon.
		return -6; // Required flag is missing from header
	}

	// Note according to msdn:  when you read a .dds file, you should not rely on the DDSCAPS_TEXTURE 
	//	and DDSCAPS_COMPLEX flags being set because some writers of such a file might not set these flags.
	//if ((header->caps & SURFACE_FLAGS_TEXTURE) == 0)
	//{
	//	free(ddsFileData);
	//	return -7; // Required flag is missing from header
	//}
	uint32_t pixelOffset = sizeof(uint32_t) + sizeof(DdsHeader);

	// Check for DX10 header
	const DdsHeader10 *header10 = NULL;
	if ( (header->pixelFormat.flags & PF_FLAGS_CODE4) && (MAKE_CODE4( 'D', 'X', '1', '0' ) == header->pixelFormat.code4) )
	{
		// Must be long enough for both headers and magic value
		if( ddsBufferSize < (sizeof(DdsHeader)+sizeof(uint32_t)+sizeof(DdsHeader10)) )
		{
			return -8; // File too small to contain a valid DX10 DDS
		}
		header10 = (const DdsHeader10*)(ddsBytes + sizeof(uint32_t) + sizeof(DdsHeader));
		pixelOffset += sizeof(DdsHeader10);
	}
	
	// Check if the contents are a cubemap.  If so, all six faces must be present.
	bool isCubeMap = false;
	if ((header->caps & SURFACE_FLAGS_COMPLEX) && (header->caps2 & CUBEMAP_FLAG_ISCUBEMAP))
	{
		if ((header->caps2 & CUBEMAP_FLAG_ALLFACES) != CUBEMAP_FLAG_ALLFACES)
		{
			return -9; // The cubemap is missing one or more faces.
		}
		isCubeMap = true;
	}

	// Check if the contents are a volume texture.
	bool isVolumeTexture = false;
	if ((header->flags & HEADER_FLAGS_DEPTH) && (header->caps2 & CUBEMAP_FLAG_VOLUME)) // (header->dwCaps & SURFACE_FLAGS_COMPLEX) -- doesn't always seem to be set?
	{
		if (header->depth == 0)
		{
			return -10; // The file is marked as a volume texture, but depth is <1
		}
		isVolumeTexture = true;
	}

	bool isCompressed = ContainsCompressedTexture(header, header10);

	uint32_t mipMapCount = 1;
	if ((header->flags & HEADER_FLAGS_MIPMAP) == HEADER_FLAGS_MIPMAP)
	{
		mipMapCount = header->mipCount;
	}

	// Begin GL-specific code!
	uint32_t blockSize;
	GLenum glFormat = GL_INVALID_ENUM;
	GLenum glInternalFormat = GL_INVALID_ENUM;
	GLenum glType = GL_INVALID_ENUM;
	if (header10 != NULL)
	{
		DXtoGLFormat(header10->dxgiFormat, &blockSize, &glFormat, &glInternalFormat, &glType);
	}
	else
	{
		DDSPFtoGLFormat(header->pixelFormat, &blockSize, &glFormat, &glInternalFormat, &glType);
	}
	if (glInternalFormat == GL_INVALID_ENUM)
	{
		return -11; // It is either unknown or unsupported format
	}
	GLenum glTarget = GL_INVALID_ENUM;
	const uint8_t *nextSrcSurface = ddsBytes+pixelOffset;
	if (isCubeMap)
	{
		glTarget = GL_TEXTURE_CUBE_MAP;
		// TODO
	}
	else if (isVolumeTexture)
	{
		glTarget = GL_TEXTURE_3D;
		// TODO
	}
	else
	{
		glTarget = GL_TEXTURE_2D;
		ZomboLite::GenTexture(outTex, "Currently-loading Texture");
		glBindTexture(glTarget, *outTex);
		glTexParameteri(glTarget, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(glTarget, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(glTarget, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(glTarget, GL_TEXTURE_MIN_FILTER, mipMapCount > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR );
		if (ogl_ext_ARB_texture_storage)
		{
			glTexStorage2D(glTarget, mipMapCount, glInternalFormat, header->width, header->height);
		}
		for(uint32_t iMip=0; iMip<mipMapCount; ++iMip)
		{
			uint32_t mipWidth  = max(header->width >> iMip, 1U);
			uint32_t mipHeight = max(header->height >> iMip, 1U);
			uint32_t mipPitch  = isCompressed ? ((mipWidth+3)/4)*blockSize : mipWidth*blockSize;
			uint32_t numRows = isCompressed ? ((mipHeight+3)/4) : mipHeight;
			uint32_t surfaceSize = mipPitch*numRows;
			ZOMBOLITE_ASSERT(nextSrcSurface + surfaceSize <= ddsBytes + ddsBufferSize, "mip %d surface is out of range in DDS data.", iMip);
			if (isCompressed)
			{
				if (ogl_ext_ARB_texture_storage)
					glCompressedTexSubImage2D(glTarget, iMip, 0,0, mipWidth,mipHeight, glInternalFormat, surfaceSize, nextSrcSurface);
				else
					glCompressedTexImage2D(glTarget, iMip, glInternalFormat, mipWidth, mipHeight, 0, surfaceSize, nextSrcSurface);
#if defined(_DEBUG)
				GLint param = 0;
				glGetTexLevelParameteriv(glTarget, iMip, GL_TEXTURE_COMPRESSED, &param);
				ZOMBOLITE_ASSERT(param != 0, "OpenGL doesn't think compressed mip is compressed!");
#endif
			}
			else
			{
				if (ogl_ext_ARB_texture_storage)
					glTexSubImage2D(glTarget, iMip, 0,0, mipWidth,mipHeight, glFormat, glType, nextSrcSurface);
				else
					glTexImage2D(glTarget, iMip, glInternalFormat, mipWidth, mipHeight, 0, glFormat, glType, nextSrcSurface);

			}
			nextSrcSurface += surfaceSize;
		}
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	return 0;
}
