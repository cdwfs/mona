// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"
#include "utils.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <float.h>
#include <stdlib.h>

typedef unsigned int uint;

// poor man's random number generator
// this is sketchy, but for purposes of PSO, acceptable (and fast!)
#define my_randf() (r = r * 1103515245 + 12345, (((float) (r & 65535)) * .000015259f))

#define max(a,b) (a > b ? a : b)
#define min(a,b) (a > b ? b : a)
#define clip(a,l,h) (max(min(a,h),l))

#define dot3(a,b) ( fmaf((a).x, (b).x, fmaf((a).y, (b).y, (a).z*(b).z)) )

//module for creating a float on our device
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

texture<float4, cudaTextureType2D, cudaReadModeElementType> currimg;  // current image rendered in triangles, minus the triangle being evolved.
surface<void, cudaSurfaceType2D> scaledimg; // current image rendered in triangles. element type is float4
texture<float4, cudaTextureType2D, cudaReadModeElementType> refimg; // original reference image

__constant__ float dkEvoAlphaLimit;
__constant__ int dkEvoCheckLimit;
__constant__ int dkEvoPsoParticleCount;
__constant__ int dkEvoPsoIterationCount;
__constant__ int dkEvoMaxTriangleCount;
__constant__ int dkEvoPsoNeighborhoodSize;
__constant__ float dkEvoPsoSpringConstant;
__constant__ float dkEvoPsoDampeningFactor;

__global__ void cudaSrand(const uint seed, const uint stateCount, curandState_t *outStates)
{
	uint iState = blockDim.x*blockIdx.x + threadIdx.x;
	while(iState < stateCount)
	{
		curand_init(seed, iState, 0, &outStates[iState]);
		iState += blockDim.x * gridDim.x;
	}
}

inline __device__ void randomizeTrianglePos(triangle *tri, curandState_t &state)
{
	tri->x1 = curand_uniform(&state);
	tri->y1 = curand_uniform(&state);
	tri->x2 = curand_uniform(&state);
	tri->y2 = curand_uniform(&state);
	tri->x3 = curand_uniform(&state);
	tri->y3 = curand_uniform(&state);
	tri->r  = curand_uniform(&state);
	tri->g  = curand_uniform(&state);
	tri->b  = curand_uniform(&state);
}

inline __device__ void randomizeTriangleVel(triangle *tri, curandState_t &state)
{
	tri->x1 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->y1 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->x2 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->y2 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->x3 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->y3 = curand_uniform(&state) * 2.0f - 1.0f;
	tri->r  = curand_uniform(&state) * 2.0f - 1.0f;
	tri->g  = curand_uniform(&state) * 2.0f - 1.0f;
	tri->b  = curand_uniform(&state) * 2.0f - 1.0f;
}

__global__ void initPsoParticles(const uint particleCount, curandState_t *states,
	triangle *positions,
	triangle *velocities,
	float *fits,
	triangle *localBestPositions,
	float *localBestFits,
	triangle *nhoodBestPositions,
	float *nhoodBestFits,
	float *globalBestFit)
{
	if (threadIdx.x >= particleCount)
	{
		return;
	}
	curandState_t state = states[threadIdx.x];
	randomizeTrianglePos(positions+threadIdx.x, state);
	randomizeTriangleVel(velocities+threadIdx.x, state);
	fits[threadIdx.x] = FLT_MAX;
	randomizeTrianglePos(localBestPositions+threadIdx.x, state);
	localBestFits[threadIdx.x] = FLT_MAX;
	randomizeTrianglePos(nhoodBestPositions+threadIdx.x, state);
	nhoodBestFits[threadIdx.x] = FLT_MAX;
	states[threadIdx.x] = state;
	if (threadIdx.x == 0)
	{
		*globalBestFit = FLT_MAX;
	}
}


// adds a triangle to the working image, or subtracts it if add==0
inline   __device__  void addtriangle(float4 * img, triangle * tri, bool add, float imgWidth, float imgHeight, int imgPitch)
{
	//intializes shared memory
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	//copies over image
	float3 triColor = make_float3(tri->r, tri->g, tri->b);

	// negate the color channels if we're subtracing this triangle
	if(add == 0) {
		triColor.x *= -1;
		triColor.y *= -1;
		triColor.z *= -1;
	}
	
	//set to alpha values that we are using 
	triColor.x = dkEvoAlphaLimit * triColor.x;
	triColor.y = dkEvoAlphaLimit * triColor.y;
	triColor.z = dkEvoAlphaLimit * triColor.z;
	
	if(threadIdx.y+threadIdx.x == 0) {
		// sort points by y value so that we can render triangles properly
		bad = 0;
		if     (tri->y1 < tri->y2 && tri->y2 < tri->y3) {
			x1 = tri->x1; y1 = tri->y1; x2 = tri->x2; y2 = tri->y2; x3 = tri->x3; y3 = tri->y3;
		} 
		else if(tri->y1 < tri->y3 && tri->y3 < tri->y2) {
			x1 = tri->x1; y1 = tri->y1; x2 = tri->x3; y2 = tri->y3; x3 = tri->x2; y3 = tri->y2;
		}
		else if(tri->y2 < tri->y1 && tri->y1 < tri->y3) {
			x1 = tri->x2; y1 = tri->y2; x2 = tri->x1; y2 = tri->y1; x3 = tri->x3; y3 = tri->y3;
		}
		else if(tri->y2 < tri->y3 && tri->y3 < tri->y1) {
			x1 = tri->x2; y1 = tri->y2; x2 = tri->x3; y2 = tri->y3; x3 = tri->x1; y3 = tri->y1;
		}
		else if(tri->y3 < tri->y1 && tri->y1 < tri->y2) {
			x1 = tri->x3; y1 = tri->y3; x2 = tri->x1; y2 = tri->y1; x3 = tri->x2; y3 = tri->y2;
		}
		else if(tri->y3 < tri->y2 && tri->y2 < tri->y1) {
			x1 = tri->x3; y1 = tri->y3; x2 = tri->x2; y2 = tri->y2; x3 = tri->x1; y3 = tri->y1;
		}
		// flag if something isn't right...
		else bad = 1;

		// calculate slopes
		m1 = (imgWidth/imgHeight)*(x2 - x1) / (y2 - y1);
		m2 = (imgWidth/imgHeight)*(x3 - x1) / (y3 - y1);
		m3 = (imgWidth/imgHeight)*(x3 - x2) / (y3 - y2);

		swap = 0;
		// enforce that m2 > m1
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = imgWidth * x1;
		xt = imgWidth * x1;

		// high limits of rows
		h1 = clip(imgHeight * y1, 0.0f, imgHeight);
		h2 = clip(imgHeight * y2, 0.0f, imgHeight);
		h3 = clip(imgHeight * y3, 0.0f, imgHeight);
	}
	__syncthreads();
	if(bad) {return;}

	// shade first half of triangle
	for(int yy = h1 + threadIdx.y; yy < h2; yy += kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y1), 0.0f, imgWidth);
		int pixelIndex = imgPitch * yy + xStart;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			img[pixelIndex].x += triColor.x;
			img[pixelIndex].y += triColor.y;
			img[pixelIndex].z += triColor.z;
			pixelIndex += kEvoBlockDim;
		}

	}

	// update slopes, row end points for second half of triangle
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * (imgHeight * (y2 - y1));
		xt += m2 * (imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	// shade second half of triangle
	for(int yy = h2 + threadIdx.y; yy < h3; yy += kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		int pixelIndex = imgPitch * yy + xStart;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			img[pixelIndex].x += triColor.x;
			img[pixelIndex].y += triColor.y;
			img[pixelIndex].z += triColor.z;
			pixelIndex += kEvoBlockDim;
		}
	}
}


// calculates the net effect on the score for a a given triangle T
// similar to addtriangle
inline   __device__  void scoretriangle(float * sum, triangle * tri, float imgWidth, float imgHeight)
{

	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	__shared__ float3 triColor;

	if(threadIdx.y+threadIdx.x == 0) {
		// sort points by y value, yes, this is retarded
		bad = 0;
		if     (tri->y1 < tri->y2 && tri->y2 < tri->y3) {
			x1 = tri->x1; y1 = tri->y1; x2 = tri->x2; y2 = tri->y2; x3 = tri->x3; y3 = tri->y3;
		} 
		else if(tri->y1 < tri->y3 && tri->y3 < tri->y2) {
			x1 = tri->x1; y1 = tri->y1; x2 = tri->x3; y2 = tri->y3; x3 = tri->x2; y3 = tri->y2;
		}
		else if(tri->y2 < tri->y1 && tri->y1 < tri->y3) {
			x1 = tri->x2; y1 = tri->y2; x2 = tri->x1; y2 = tri->y1; x3 = tri->x3; y3 = tri->y3;
		}
		else if(tri->y2 < tri->y3 && tri->y3 < tri->y1) {
			x1 = tri->x2; y1 = tri->y2; x2 = tri->x3; y2 = tri->y3; x3 = tri->x1; y3 = tri->y1;
		}
		else if(tri->y3 < tri->y1 && tri->y1 < tri->y2) {
			x1 = tri->x3; y1 = tri->y3; x2 = tri->x1; y2 = tri->y1; x3 = tri->x2; y3 = tri->y2;
		}
		else if(tri->y3 < tri->y2 && tri->y2 < tri->y1) {
			x1 = tri->x3; y1 = tri->y3; x2 = tri->x2; y2 = tri->y2; x3 = tri->x1; y3 = tri->y1;
		}
		// flag if something isn't right...
		else bad = 1;

		// calculate slopes
		m1 = clip((imgWidth/imgHeight)*(x2 - x1) / (y2 - y1), -imgHeight, imgHeight);
		m2 = clip((imgWidth/imgHeight)*(x3 - x1) / (y3 - y1), -imgHeight, imgHeight);
		m3 = clip((imgWidth/imgHeight)*(x3 - x2) / (y3 - y2), -imgHeight, imgHeight);
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = imgWidth * x1;
		xt = imgWidth * x1;

		// high limits of rows
		h1 = clip(imgHeight * y1, 0.0f, imgHeight);
		h2 = clip(imgHeight * y2, 0.0f, imgHeight);
		h3 = clip(imgHeight * y3, 0.0f, imgHeight);

		triColor = make_float3(tri->r, tri->g, tri->b);
		triColor.x = dkEvoAlphaLimit * triColor.x;
		triColor.y = dkEvoAlphaLimit * triColor.y;
		triColor.z = dkEvoAlphaLimit * triColor.z;
	}
	__syncthreads();
	if(bad) {*sum = FLT_MAX; return;}

	// score first half of triangle. This substract the score prior to the last triangle
	float localsum = 0.0f;
	for(int yy = threadIdx.y+h1; yy < h2; yy+=kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y1), 0.0f, imgWidth);
		float4 currentPixel, refPixel;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
#if 0
			// This version is fewer ALU ops, but more registers
			currentPixel = tex2D(currimg, xx, yy);
			refPixel     = tex2D(refimg, xx, yy);
			float pixelScoreR = fmaf(-2.0f, refPixel.x, triColor.x);
			float pixelScoreG = fmaf(-2.0f, refPixel.y, triColor.y);
			float pixelScoreB = fmaf(-2.0f, refPixel.z, triColor.z);
			pixelScoreR       = fmaf(2.0f, currentPixel.x, pixelScoreR);
			pixelScoreG       = fmaf(2.0f, currentPixel.y, pixelScoreG);
			pixelScoreB       = fmaf(2.0f, currentPixel.z, pixelScoreB);
			localsum          = fmaf(pixelScoreR, triColor.x, localsum);
			localsum          = fmaf(pixelScoreG, triColor.y, localsum);
			localsum          = fmaf(pixelScoreB, triColor.z, localsum);
#else
			currentPixel = tex2D(currimg, xx, yy);
			refPixel     = tex2D(refimg,  xx, yy);
			float4 pixelDiff = currentPixel - refPixel;
			localsum -= dot3(pixelDiff, pixelDiff);
			pixelDiff.x += triColor.x; pixelDiff.y += triColor.y; pixelDiff.z += triColor.z;
			localsum += dot3(pixelDiff, pixelDiff);
#endif
		}
	}
	
	// update slopes and limits to score second half of triangle
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * (imgHeight * (y2 - y1));
		xt += m2 * (imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();
		
	// score second half
	for(int yy = threadIdx.y+h2; yy < h3; yy+=kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		float4 currentPixel, refPixel;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
#if 0
			// This version is fewer ALU ops, but more registers
			currentPixel = tex2D(currimg, xx, yy);
			refPixel     = tex2D(refimg, xx, yy);
			float pixelScoreR = fmaf(-2.0f, refPixel.x, triColor.x);
			float pixelScoreG = fmaf(-2.0f, refPixel.y, triColor.y);
			float pixelScoreB = fmaf(-2.0f, refPixel.z, triColor.z);
			pixelScoreR       = fmaf(2.0f, currentPixel.x, pixelScoreR);
			pixelScoreG       = fmaf(2.0f, currentPixel.y, pixelScoreG);
			pixelScoreB       = fmaf(2.0f, currentPixel.z, pixelScoreB);
			localsum          = fmaf(pixelScoreR, triColor.x, localsum);
			localsum          = fmaf(pixelScoreG, triColor.y, localsum);
			localsum          = fmaf(pixelScoreB, triColor.z, localsum);
#else
			currentPixel = tex2D(currimg, xx, yy);
			refPixel     = tex2D(refimg,  xx, yy);
			float4 pixelDiff = currentPixel - refPixel;
			localsum -= dot3(pixelDiff, pixelDiff);
			pixelDiff.x += triColor.x; pixelDiff.y += triColor.y; pixelDiff.z += triColor.z;
			localsum += dot3(pixelDiff, pixelDiff);
#endif
		}
	}

	atomicAdd(sum, localsum); // Could do a more clever reduction than this, but it doesn't seem to be a bottleneck
}



// optimizes the Mth triangle using PSO
__global__ void run(triangle * curr,   //D (triangles)
					triangle * pos,    //S (particles)
					triangle * vel,    //S (particles)
					float * fit,       //S (particles)
					triangle * lbest,  //S (particles)
					float * lbval,     
					triangle * nbest,
					float * nbval,
					float * gbval,
					int * M,
					float imgWidth,
					float imgHeight) {
	uint r = pos[0].x1 * (float)(100 + threadIdx.x * 666 + blockIdx.x * 94324 + threadIdx.y * 348);
	__shared__ int check; check = 0;


	// loop over pso updates
	for(int q = 0; q < dkEvoPsoIterationCount; q++) {

		// integrate position
		if(q > 0 && threadIdx.y==0 && threadIdx.x < kEvoNumFloatsPerTriangle) {
			float vmax =  .2f * my_randf() + 0.05f;
			float vmin = -.2f * my_randf() - 0.05f;
			float * v = (((float *) &vel[blockIdx.x]) + threadIdx.x);
			float * p = (((float *) &pos[blockIdx.x]) + threadIdx.x);
			float * l = (((float *) &lbest[blockIdx.x]) + threadIdx.x);
			float * n = (((float *) &nbest[blockIdx.x]) + threadIdx.x);
			*v *= dkEvoPsoDampeningFactor;
			*v += dkEvoPsoSpringConstant * my_randf() * (*n - *p);
			*v += dkEvoPsoSpringConstant * my_randf() * (*l - *p);
			*v = min( max(*v, vmin), vmax );
			*p += *v;
			if(fit[blockIdx.x] > 0 && my_randf() < 0.01f)
				*p = my_randf();
			if (threadIdx.x >= 6)
			{
				*p = clip(*p, 0.0f, 1.0f);
			}
		}
		__syncthreads();

		// eval fitness
		fit[blockIdx.x] = 0;
		scoretriangle(&fit[blockIdx.x], &pos[blockIdx.x], imgWidth, imgHeight);

		if(threadIdx.x+threadIdx.y == 0) {
			// local max find
			if(fit[blockIdx.x] < lbval[blockIdx.x]) {
				lbest[blockIdx.x] = pos[blockIdx.x];
				lbval[blockIdx.x] = fit[blockIdx.x];
			}
			// hack to improve early PSO convergence (is this D&R's "local best reduction"?)
			else if(lbval[blockIdx.x] > 0) { 
				lbval[blockIdx.x] *= 1.1f;
			}

			// global max find
			// This is a race condition! Update must happen atomically!
			if (fit[blockIdx.x] < *gbval) {
				*gbval = fit[blockIdx.x];
				curr[*M] = pos[blockIdx.x];
				check = 0;
			}
			else check++;

			// neighbor max find (next k topology)
			float v;
			int b;
			b = blockIdx.x;
			v = nbval[b % dkEvoPsoParticleCount];
			for(int j = 0; j < dkEvoPsoNeighborhoodSize; j++) {
				if(lbval[(blockIdx.x + j) % dkEvoPsoParticleCount] < v) {
					v = lbval[(blockIdx.x + j) % dkEvoPsoParticleCount];
					b = blockIdx.x + j;
				}
			}
			if(v < nbval[blockIdx.x]) {
				nbval[blockIdx.x] = v;
				nbest[blockIdx.x] = lbest[b % dkEvoPsoParticleCount];
			}	
			// hack to improve early PSO convergence (is this D&R's "local best reduction"?)
			else if(lbval[blockIdx.x] > 0) 
				nbval[blockIdx.x] *= 1.1f;

		}
		// exit if PSO stagnates
		if(check > dkEvoCheckLimit) return;
		__syncthreads();
	}

}

// renders and scores an image
__global__ void render(float4 *img,
					   triangle * curr,
					   int * K,
					   float * score,
					   int imgWidth,
					   int imgHeight,
					   int imgPitch) {
	// clear image
	for(int y = threadIdx.y; y < imgHeight; y += kEvoBlockDim) {
		for(int i = threadIdx.x; i < imgWidth; i += kEvoBlockDim) {
			int g = y * imgPitch + i;
			img[g].x = 0.0f;
			img[g].y = 0.0f;
			img[g].z = 0.0f;
		}
	}
	__syncthreads();
	// render all triangles
	for(int k = 0; k < dkEvoMaxTriangleCount; k++)
		addtriangle(img, &curr[k], 1, (float)imgWidth, (float)imgHeight, imgPitch);
	__syncthreads();
	// score the image
	float localsum = 0.0f;
	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		int g = yy*imgPitch + threadIdx.x;
		for(int xx = threadIdx.x; xx < imgWidth; xx += kEvoBlockDim) {
			float4 o = tex2D(refimg, xx, yy);
			o.x -= img[g].x; o.y -= img[g].y; o.z -= img[g].z;
			localsum += dot3(o, o);
			g += kEvoBlockDim;
		}
	}
	atomicAdd(score, localsum);

	// remove triangles we are modifying
	__syncthreads();
	addtriangle(img, &curr[*K], 0, (float)imgWidth, (float)imgHeight, imgPitch);
}




// similar to addtriangle function, but for output. Not worth looking at...
inline   __device__  void addtriangleproof(triangle * T, float imgWidth, float imgHeight)
{
	//sort points by y value, yes, this is retarded
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	if(threadIdx.x+threadIdx.y==0) {
		T->r = clip(T->r, 0.0f, 1.0f);
		T->g = clip(T->g, 0.0f, 1.0f);
		T->b = clip(T->b, 0.0f, 1.0f);
		bad = 0;
		if     (T->y1 < T->y2 && T->y2 < T->y3) {
			x1 = T->x1; y1 = T->y1; x2 = T->x2; y2 = T->y2; x3 = T->x3; y3 = T->y3;
		} 
		else if(T->y1 < T->y3 && T->y3 < T->y2) {
			x1 = T->x1; y1 = T->y1; x2 = T->x3; y2 = T->y3; x3 = T->x2; y3 = T->y2;
		}
		else if(T->y2 < T->y1 && T->y1 < T->y3) {
			x1 = T->x2; y1 = T->y2; x2 = T->x1; y2 = T->y1; x3 = T->x3; y3 = T->y3;
		}
		else if(T->y2 < T->y3 && T->y3 < T->y1) {
			x1 = T->x2; y1 = T->y2; x2 = T->x3; y2 = T->y3; x3 = T->x1; y3 = T->y1;
		}
		else if(T->y3 < T->y1 && T->y1 < T->y2) {
			x1 = T->x3; y1 = T->y3; x2 = T->x1; y2 = T->y1; x3 = T->x2; y3 = T->y2;
		}
		else if(T->y3 < T->y2 && T->y2 < T->y1) {
			x1 = T->x3; y1 = T->y3; x2 = T->x2; y2 = T->y2; x3 = T->x1; y3 = T->y1;
		}
		else bad = 1;

		m1 = clip(((imgWidth)/(imgHeight))*(x2 - x1) / (y2 - y1), -imgHeight, imgHeight);
		m2 = clip(((imgWidth)/(imgHeight))*(x3 - x1) / (y3 - y1), -imgHeight, imgHeight);
		m3 = clip(((imgWidth)/(imgHeight))*(x3 - x2) / (y3 - y2), -imgHeight, imgHeight);
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}
		xs = imgWidth * x1;
		xt = imgWidth * x1;
		h1 = clip(imgHeight * y1, 0.0f, imgHeight);
		h2 = clip(imgHeight * y2, 0.0f, imgHeight);
		h3 = clip(imgHeight * y3, 0.0f, imgHeight);
	}
	__syncthreads();

	if(bad) return;
	for(int yy = h1 + threadIdx.y; yy < h2; yy += kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y1), 0.0f, imgWidth);
		float4 currentPixel;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			surf2Dread(&currentPixel, scaledimg, xx*sizeof(float4), yy);
			currentPixel.x += dkEvoAlphaLimit * T->r;
			currentPixel.y += dkEvoAlphaLimit * T->g;
			currentPixel.z += dkEvoAlphaLimit * T->b;
			surf2Dwrite(currentPixel, scaledimg, xx*sizeof(float4), yy);
		}

	}
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * (imgHeight * (y2 - y1));
		xt += m2 * (imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	for(int yy = h2 + threadIdx.y; yy < h3; yy += kEvoBlockDim) {
		int xStart = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		int xMax   =               clip(xt + m2 * (yy - imgHeight * y2 + 1), 0.0f, imgWidth);
		float4 currentPixel;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			surf2Dread(&currentPixel, scaledimg, xx*sizeof(float4), yy);
			currentPixel.x += dkEvoAlphaLimit * T->r;
			currentPixel.y += dkEvoAlphaLimit * T->g;
			currentPixel.z += dkEvoAlphaLimit * T->b;
			surf2Dwrite(currentPixel, scaledimg, xx*sizeof(float4), yy);
		}
	}
}

// similar to render, but for output. Also not worth looking at.
__global__ void renderproof(triangle * curr, int imgWidth, int imgHeight) {
	// clear image
	for(int yy = threadIdx.y; yy < imgHeight; yy += kEvoBlockDim) {
		const float4 zero4 = make_float4(0,0,0,0);
		for (int xx = threadIdx.x; xx < imgWidth; xx += kEvoBlockDim) {
			surf2Dwrite(zero4, scaledimg, xx*sizeof(float4), yy);
		}
	}
	__syncthreads();
	// Add triangles
	for(int k = 0; k < dkEvoMaxTriangleCount; k++)
		addtriangleproof(&curr[k], (float)imgWidth, (float)imgHeight);
	__syncthreads();
	// Clamp output to [0..1]
	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		float4 currentPixel;
		for(int xx = threadIdx.x; xx < imgWidth; xx += kEvoBlockDim) {
			surf2Dread(&currentPixel, scaledimg, xx*sizeof(float4), yy);
			currentPixel.x = clip(currentPixel.x, 0.0f, 1.0f);
			currentPixel.y = clip(currentPixel.y, 0.0f, 1.0f);
			currentPixel.z = clip(currentPixel.z, 0.0f, 1.0f);
			currentPixel.w = 1.0;
			surf2Dwrite(currentPixel, scaledimg, xx*sizeof(float4), yy);
		}
	}
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

// Clamp x to the range [min..max] (inclusive)
static inline float clamp(float x, float min, float max)
{
	return (x<min) ? min : ( (x>max) ? max : x );
}

PsoContext::PsoContext(void)
{
	m_currentIteration = 0;
	m_imgWidth = 0;
	m_imgHeight = 0;
	md_originalPixels = nullptr;
	m_currentPixelsPitch = 0;
	md_currentPixels = nullptr;
	mh_scaledOutputPixels = nullptr;
	mh_scaledOutputRgba8888 = nullptr;
	md_scaledOutputPixels = nullptr;
	m_refImg = nullptr;
	m_currImg = nullptr;
	m_scaledImg = nullptr;
	mh_currentTriangles = nullptr;
	md_currentTriangles = nullptr;
	mh_bestTriangles = nullptr;
	md_bestTriangles = nullptr;
	md_currentTriangleIndex = nullptr;
	m_currentScore = 0;
	md_currentScore = nullptr;
	m_bestScore = 0;
	md_psoRandStates = nullptr;
	md_psoParticlesPos = nullptr;
	md_psoParticlesVel = nullptr;
	md_psoParticlesFit = nullptr;
	md_psoParticlesLocalBestPos = nullptr;
	md_psoParticlesLocalBestFit = nullptr;
	md_psoParticlesNhoodBestPos = nullptr;
	md_psoParticlesNhoodBestFit = nullptr;
	md_psoParticlesGlobalBestFit = nullptr;
}
PsoContext::~PsoContext(void)
{
	CUDA_CHECK( cudaFreeArray(md_originalPixels) );
	CUDA_CHECK( cudaFree(md_currentPixels) );
	free(mh_scaledOutputPixels);
	free(mh_scaledOutputRgba8888);
	CUDA_CHECK( cudaFreeArray(md_scaledOutputPixels) );
	free(mh_currentTriangles);
	CUDA_CHECK( cudaFree(md_currentTriangles) );
	free(mh_bestTriangles);
	CUDA_CHECK( cudaFree(md_bestTriangles) );
	CUDA_CHECK( cudaFree(md_currentTriangleIndex) );
	CUDA_CHECK( cudaFree( md_currentScore ) );
	CUDA_CHECK( cudaFree(md_psoRandStates) );
	CUDA_CHECK( cudaFree(md_psoParticlesPos) );
	CUDA_CHECK( cudaFree(md_psoParticlesVel) );
	CUDA_CHECK( cudaFree(md_psoParticlesFit) );
	CUDA_CHECK( cudaFree(md_psoParticlesLocalBestPos) );
	CUDA_CHECK( cudaFree(md_psoParticlesLocalBestFit) );
	CUDA_CHECK( cudaFree(md_psoParticlesNhoodBestPos) );
	CUDA_CHECK( cudaFree(md_psoParticlesNhoodBestFit) );
	CUDA_CHECK( cudaFree(md_psoParticlesGlobalBestFit) );
}

int PsoContext::init(const int imgWidth, const int imgHeight, const float4 *h_originalPixels, const PsoConstants &constants)
{
	m_currentIteration = 0;
	m_imgWidth  = imgWidth;
	m_imgHeight = imgHeight;

	// Upload original pixels to device memory
	const size_t srcPitch = (size_t)imgWidth * sizeof(float4);
	m_refChannelDesc = cudaCreateChannelDesc<float4>();
	CUDA_CHECK( cudaMallocArray(&md_originalPixels, &m_refChannelDesc, imgWidth, imgHeight, cudaArrayDefault) );
	CUDA_CHECK( cudaMemcpy2DToArray(md_originalPixels, 0,0, h_originalPixels, srcPitch, srcPitch, imgHeight, cudaMemcpyHostToDevice) );

	// Bind reference image texture to original pixels
	CUDA_CHECK( cudaGetTextureReference(&m_refImg, &refimg) );
	CUDA_CHECK( cudaBindTextureToArray(m_refImg, md_originalPixels, &m_refChannelDesc) );

	// Create array of solution triangles
	mh_currentTriangles = (triangle*)malloc(constants.maxTriangleCount*sizeof(triangle));
	memset(mh_currentTriangles, 0, constants.maxTriangleCount*sizeof(triangle));
	md_currentTriangles = nullptr;
	CUDA_CHECK( cudaMalloc(&md_currentTriangles,    constants.maxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemset( md_currentTriangles, 0, constants.maxTriangleCount*sizeof(triangle)) );
	mh_bestTriangles = (triangle*)malloc(constants.maxTriangleCount*sizeof(triangle));
	memcpy(mh_bestTriangles, mh_currentTriangles, constants.maxTriangleCount*sizeof(triangle));
	md_bestTriangles = nullptr;
	CUDA_CHECK( cudaMalloc(&md_bestTriangles, constants.maxTriangleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMemcpy( md_bestTriangles, md_currentTriangles, constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );

	// Rendered solution on the GPU
	md_currentPixels = nullptr;
	m_currentPixelsPitch = 0;
	m_currChannelDesc = cudaCreateChannelDesc<float4>();
	CUDA_CHECK( cudaMallocPitch(&md_currentPixels, &m_currentPixelsPitch,    imgWidth*sizeof(float4), imgHeight) );
	size_t textureOffset = 0; // unused
	CUDA_CHECK( cudaGetTextureReference(&m_currImg, &currimg) );
	CUDA_CHECK( cudaBindTexture2D(&textureOffset, m_currImg, md_currentPixels, &m_currChannelDesc, m_imgWidth, m_imgHeight, m_currentPixelsPitch) );

	// Scaled-up pixels for final output to image file
	md_scaledOutputPixels = nullptr;
	CUDA_CHECK( cudaMallocArray(&md_scaledOutputPixels, &m_currChannelDesc, constants.outputScale*imgWidth, constants.outputScale*imgHeight, cudaArraySurfaceLoadStore) );
	CUDA_CHECK( cudaGetSurfaceReference(&m_scaledImg, &scaledimg) );
	mh_scaledOutputPixels   =   (float4*)malloc(constants.outputScale*imgWidth*constants.outputScale*imgHeight*sizeof(float4));
	mh_scaledOutputRgba8888 = (uint32_t*)malloc(constants.outputScale*imgWidth*constants.outputScale*imgHeight*sizeof(uint32_t));

	// Index of triangle currently being updated
	md_currentTriangleIndex = nullptr;
	CUDA_CHECK( cudaMalloc(&md_currentTriangleIndex, sizeof(int32_t)) );

	// Current score of this iteration, and best score to date
	m_currentScore = FLT_MAX;
	CUDA_CHECK( cudaMalloc(&md_currentScore, sizeof(float)) );
	CUDA_CHECK( cudaMemcpy(md_currentScore, &m_currentScore, sizeof(float), cudaMemcpyHostToDevice) );
	m_bestScore = FLT_MAX;

	// PSO arrays
	md_psoRandStates             = nullptr;
	md_psoParticlesPos           = nullptr;
	md_psoParticlesVel           = nullptr;
	md_psoParticlesFit           = nullptr;
	md_psoParticlesLocalBestPos  = nullptr;
	md_psoParticlesLocalBestFit  = nullptr;
	md_psoParticlesNhoodBestPos	 = nullptr;
	md_psoParticlesNhoodBestFit  = nullptr;
	md_psoParticlesGlobalBestFit = nullptr;
	CUDA_CHECK( cudaMalloc(&md_psoRandStates,            constants.psoParticleCount*sizeof(curandState_t)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesPos,          constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesVel,          constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesFit,          constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesLocalBestPos, constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesLocalBestFit, constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesNhoodBestPos, constants.psoParticleCount*sizeof(triangle)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesNhoodBestFit, constants.psoParticleCount*sizeof(float)) );
	CUDA_CHECK( cudaMalloc(&md_psoParticlesGlobalBestFit,                           sizeof(float)) );
	launchSrand();

	// Upload constants to GPU
	m_constants = constants;
	setGpuConstants(&constants);

	return 0;
}

void PsoContext::iterate(void)
{
	m_currentIteration += 1;
	// Choose a new random triangle to update
	const int32_t currentTriangleIndex = rand() % min((m_currentIteration+1)/2, m_constants.maxTriangleCount);
	CUDA_CHECK( cudaMemcpy(md_currentTriangleIndex, &currentTriangleIndex, sizeof(int32_t), cudaMemcpyHostToDevice) );

	// Generate random particles for this PSO iteration
	launchInitPsoParticles();

	// Render initial solution
	launchRender();
	CUDA_CHECK( cudaMemcpy(&m_currentScore, md_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );

	// check that this isn't a huge regression, revert and pick new K if so
	if (m_currentScore * (1.0f - 2.0f / (float)m_constants.maxTriangleCount) > m_bestScore)
	{
		memcpy(mh_currentTriangles, mh_bestTriangles, m_constants.maxTriangleCount*sizeof(triangle));
		CUDA_CHECK( cudaMemcpy(md_currentTriangles, mh_currentTriangles, m_constants.maxTriangleCount*sizeof(triangle), cudaMemcpyHostToDevice) );
		launchRender();
		CUDA_CHECK( cudaMemcpy(&m_currentScore, md_currentScore, sizeof(float), cudaMemcpyDeviceToHost) );
	}

	// Update best score if needed
	if (m_currentScore < m_bestScore && m_currentScore != 0)
	{
		m_bestScore = m_currentScore;
		// Update best known solution
		memcpy(mh_bestTriangles, mh_currentTriangles, m_constants.maxTriangleCount*sizeof(triangle));
		CUDA_CHECK( cudaMemcpy(md_bestTriangles, md_currentTriangles, m_constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToDevice) );
	}

	// Launch the PSO grid
	launchRun();

	// Copy current solution back to host
	CUDA_CHECK( cudaMemcpy(mh_currentTriangles, md_currentTriangles, m_constants.maxTriangleCount*sizeof(triangle), cudaMemcpyDeviceToHost) );
}

float PsoContext::bestPsnr(void) const
{
	const float mse = m_bestScore / (float)(3*m_imgWidth*m_imgHeight);
	return 10.0f * log10(1.0f * 1.0f / mse);
}

int PsoContext::renderToCudaArray(cudaArray_t dst)
{
	CUDA_CHECK( cudaBindSurfaceToArray(m_scaledImg, dst, &m_currChannelDesc) );
	launchRenderProof();
	return 0;
}
int PsoContext::renderToFile(const char *imageFileName)
{
	renderToCudaArray(md_scaledOutputPixels);

	const size_t srcPitch = (size_t)m_imgWidth * sizeof(float4);
	CUDA_CHECK( cudaMemcpy2DFromArray(mh_scaledOutputPixels, m_constants.outputScale*srcPitch, md_scaledOutputPixels,
		0,0, m_constants.outputScale*srcPitch, m_constants.outputScale*m_imgHeight, cudaMemcpyDeviceToHost) );
	// Convert to RGBA8888 for output
	for(int32_t iPixel=0; iPixel<m_constants.outputScale*m_imgWidth * m_constants.outputScale*m_imgHeight; ++iPixel)
	{
		mh_scaledOutputRgba8888[iPixel] =
			( uint32_t(clamp(mh_scaledOutputPixels[iPixel].x * 255.0f, 0.0f, 255.0f)) <<  0 ) |
			( uint32_t(clamp(mh_scaledOutputPixels[iPixel].y * 255.0f, 0.0f, 255.0f)) <<  8 ) |
			( uint32_t(clamp(mh_scaledOutputPixels[iPixel].z * 255.0f, 0.0f, 255.0f)) << 16 ) |
			(                                                                   0xFFU << 24 );
	}
	// Write output image.
	// TODO: select output format from [jpg, png, bmp]?
	//printf("Writing '%s'...\n", imageFileName);
	int32_t writeError = stbi_write_png(imageFileName, m_constants.outputScale*m_imgWidth, m_constants.outputScale*m_imgHeight,
		4, mh_scaledOutputRgba8888, m_constants.outputScale*m_imgWidth*sizeof(uint32_t));
	if (writeError == 0)
	{
		//fprintf(stderr, "Error writing final output image '%s'\n", imageFileName);
		return -1;
	}
	return 0;
}


void PsoContext::setGpuConstants(const PsoConstants *constants)
{
	size_t destSize = 0;

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoAlphaLimit) );
	assert( destSize == sizeof(float) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoAlphaLimit, &(constants->alphaLimit), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoCheckLimit) );
	assert( destSize == sizeof(int32_t) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoCheckLimit, &(constants->checkLimit), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoPsoParticleCount) );
	assert( destSize == sizeof(int32_t) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoPsoParticleCount, &(constants->psoParticleCount), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoPsoIterationCount) );
	assert( destSize == sizeof(int32_t) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoPsoIterationCount, &(constants->psoIterationCount), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoMaxTriangleCount) );
	assert( destSize == sizeof(int32_t) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoMaxTriangleCount, &(constants->maxTriangleCount), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoPsoNeighborhoodSize) );
	assert( destSize == sizeof(int32_t) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoPsoNeighborhoodSize, &(constants->psoNeighborhoodSize), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoPsoSpringConstant) );
	assert( destSize == sizeof(float) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoPsoSpringConstant, &(constants->psoSpringConstant), destSize) );

	CUDA_CHECK( cudaGetSymbolSize(&destSize, dkEvoPsoDampeningFactor) );
	assert( destSize == sizeof(float) );
	CUDA_CHECK( cudaMemcpyToSymbol(dkEvoPsoDampeningFactor, &(constants->psoDampeningFactor), destSize) );
}

void PsoContext::launchSrand(void)
{
	dim3 gridDim(1,1);
	dim3 blockDim(m_constants.psoParticleCount,1);
	cudaSrand<<<gridDim, blockDim>>>(rand(), m_constants.psoParticleCount, md_psoRandStates);
	CUDA_CHECK( cudaGetLastError() );
}

void PsoContext::launchInitPsoParticles(void)
{
	dim3 gridDim(1,1);
	dim3 blockDim(m_constants.psoParticleCount,1);
	initPsoParticles<<<gridDim, blockDim>>>(m_constants.psoParticleCount, md_psoRandStates,
		md_psoParticlesPos, md_psoParticlesVel, md_psoParticlesFit, md_psoParticlesLocalBestPos, md_psoParticlesLocalBestFit,
		md_psoParticlesNhoodBestPos, md_psoParticlesNhoodBestFit, md_psoParticlesGlobalBestFit);
	CUDA_CHECK( cudaGetLastError() );
}

void PsoContext::launchRender(void)
{
	CUDA_CHECK( cudaMemset(md_currentScore, 0, sizeof(float)) );
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	render<<<gridDim, blockDim>>>(md_currentPixels, md_currentTriangles, md_currentTriangleIndex, md_currentScore, m_imgWidth, m_imgHeight, m_currentPixelsPitch/sizeof(float4));
	CUDA_CHECK( cudaGetLastError() );
}

void PsoContext::launchRenderProof(void)
{
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	renderproof<<<gridDim, blockDim>>>(md_bestTriangles, m_constants.outputScale*m_imgWidth, m_constants.outputScale*m_imgHeight);
	CUDA_CHECK( cudaGetLastError() );
}

void PsoContext::launchRun(void)
{
	dim3 gridDim(m_constants.psoParticleCount, 1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim, 1);
	//CUDA_CHECK( cudaFuncSetCacheConfig(run, cudaFuncCachePreferL1) ); // No significant difference observed, but this kernel certainly doesn't use smem
	run<<<gridDim, blockDim>>>(md_currentTriangles, md_psoParticlesPos, md_psoParticlesVel, md_psoParticlesFit,
		md_psoParticlesLocalBestPos, md_psoParticlesLocalBestFit,
		md_psoParticlesNhoodBestPos, md_psoParticlesNhoodBestFit,
		md_psoParticlesGlobalBestFit,
		md_currentTriangleIndex, (float)m_imgWidth, (float)m_imgHeight);
	CUDA_CHECK( cudaGetLastError() );
}
