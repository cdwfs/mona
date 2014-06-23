#include "utils.h"

#include <cassert>

static const double kOriginSeconds = 4294967296.0;

double CpuTimer::GetRawSeconds(void)
{
	static double kTicksPerSecond = 0;
	if (kTicksPerSecond == 0)
	{
		LARGE_INTEGER freq;
		QueryPerformanceFrequency(&freq);
		kTicksPerSecond = (double)( (uint64_t)(freq.HighPart) << 32ULL | (uint64_t)(freq.LowPart) );
	}
	LARGE_INTEGER pc;
	QueryPerformanceCounter(&pc);
	uint64_t tickCount = (uint64_t)(pc.HighPart) << 32ULL | (uint64_t)(pc.LowPart);
	return (double)tickCount / kTicksPerSecond;
}

CpuTimer::CpuTimer(void) :
m_elapsedSeconds(kOriginSeconds),
	m_lastRawSeconds(0),
	m_isRunning(false)
{
}

void CpuTimer::Reset(void)
{
	m_elapsedSeconds = kOriginSeconds;
	m_lastRawSeconds = 0;
	m_isRunning = false;
}

void CpuTimer::Start(void)
{
	assert(!m_isRunning);
	m_isRunning = true;
	m_lastRawSeconds = GetRawSeconds();
}

void CpuTimer::Stop(void)
{
	assert(m_isRunning);
	m_isRunning = false;
}

double CpuTimer::UpdateWithScale(double scale)
{
	if (!m_isRunning)
		return 0;
	double newRawSeconds = GetRawSeconds();
	double delta = (newRawSeconds - m_lastRawSeconds) * scale;
	m_elapsedSeconds += delta;
	m_lastRawSeconds = newRawSeconds;
	return delta;
}

double CpuTimer::GetElapsedSeconds(void) const
{
	return m_elapsedSeconds - kOriginSeconds;
}

// Copied from helper_cuda.h in the NVIDIA deviceQuery sample
int _ConvertSMVer2Cores(int major, int minor)
{
	// Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
	typedef struct
	{
		int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] =
	{
		{ 0x10,  8 }, // Tesla Generation (SM 1.0) G80 class
		{ 0x11,  8 }, // Tesla Generation (SM 1.1) G8x class
		{ 0x12,  8 }, // Tesla Generation (SM 1.2) G9x class
		{ 0x13,  8 }, // Tesla Generation (SM 1.3) GT200 class
		{ 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
		{ 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
		{ 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
		{ 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
		{ 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
		{ 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
		{   -1, -1 }
	};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1)
	{
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
		{
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one to run properly
	printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[7].Cores);
	return nGpuArchCoresPerSM[7].Cores;
}
