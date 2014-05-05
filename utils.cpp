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
