// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"
#include "utils.h"

#include <float.h>

typedef unsigned int uint;

// poor man's random number generator
#define rand() (r = r * 1103515245 + 12345, (((float) (r & 65535)) * .000015259f))
// this is sketchy, but for purposes of PSO, acceptable (and fast!)

#define max(a,b) (a > b ? a : b)
#define min(a,b) (a > b ? b : a)
#define clip(a,l,h) (max(min(a,h),l))

#define dot3(a,b) ( fmaf((a).x, (b).x, fmaf((a).y, (b).y, (a).z*(b).z)) )

//module for creating a float on our device
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
     
texture<float4, cudaTextureType2D, cudaReadModeElementType> currimg;  // current image rendered in triangles
texture<float4, cudaTextureType2D, cudaReadModeElementType> refimg; // original reference image

__constant__ float dkEvoAlphaLimit;
__constant__ int dkEvoCheckLimit;
__constant__ int dkEvoPsoParticleCount;
__constant__ int dkEvoPsoIterationCount;
__constant__ int dkEvoMaxTriangleCount;
__constant__ int dkEvoPsoNeighborhoodSize;
__constant__ float dkEvoPsoSpringConstant;
__constant__ float dkEvoPsoDampeningFactor;


// adds a triangle to the working image, or subtracts it if add==0
inline   __device__  void addtriangle(float4 * img, triangle * tri, bool add, float imgWidth, float imgHeight, int imgPitch)
{
	//intializes shared memory
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	//copies over image
	float3 triColor = make_float3(tri->r, tri->g, tri->b);

	//clip color values to valid range
	triColor.x = clip(triColor.x, 0.0f, 1.0f);
	triColor.y = clip(triColor.y, 0.0f, 1.0f);
	triColor.z = clip(triColor.z, 0.0f, 1.0f);

	//if we are subtracting the triangle, set values to -1
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
		triColor.x = clip(triColor.x, 0.0f, 1.0f);
		triColor.y = clip(triColor.y, 0.0f, 1.0f);
		triColor.z = clip(triColor.z, 0.0f, 1.0f);
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
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
#if 0
			// This version is fewer ALU ops, but more registers
			const float4 currentPixel = tex2D(currimg, xx, yy);
			const float4 refPixel = tex2D(refimg, xx, yy);
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
			float4 pixelDiff = tex2D(currimg, xx, yy) - tex2D(refimg, xx, yy);
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
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
#if 0
			// This version is fewer ALU ops, but more registers
			const float4 currentPixel = tex2D(currimg, xx, yy);
			const float4 refPixel = tex2D(refimg, xx, yy);
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
			float4 pixelDiff = tex2D(currimg, xx, yy) - tex2D(refimg, xx, yy);
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
			float vmax = .2f * rand() + 0.05f;
			float vmin = -.2f * rand() - 0.05f;	
			float * v = (((float *) &vel[blockIdx.x]) + threadIdx.x);
			float * p = (((float *) &pos[blockIdx.x]) + threadIdx.x);
			float * l = (((float *) &lbest[blockIdx.x]) + threadIdx.x);
			float * n = (((float *) &nbest[blockIdx.x]) + threadIdx.x);
			*v *= dkEvoPsoDampeningFactor;
			*v += dkEvoPsoSpringConstant * rand() * (*n - *p);
			*v += dkEvoPsoSpringConstant * rand() * (*l - *p);
			*v = min( max(*v, vmin), vmax );
			*p += *v;
			if(fit[blockIdx.x] > 0 && rand() < 0.01f)
				*p = rand();
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
__global__ void render(float4 * im,
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
			im[g].x = 0.0f;
			im[g].y = 0.0f;
			im[g].z = 0.0f;
		}
	}
	// render all triangles
	for(int k = 0; k < dkEvoMaxTriangleCount; k++)
		addtriangle(im, &curr[k], 1, (float)imgWidth, (float)imgHeight, imgPitch);

	// score the image
	float localsum = 0.0f;
	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		int g = yy*imgPitch + threadIdx.x;
		for(int xx = threadIdx.x; xx < imgWidth; xx += kEvoBlockDim) {
			float4 o = tex2D(refimg, xx, yy);
			o.x -= im[g].x; o.y -= im[g].y; o.z -= im[g].z;
			localsum += dot3(o, o);
			g += kEvoBlockDim;
		}
	}
	atomicAdd(score, localsum);

	// remove triangles we are modifying
	__syncthreads();
	addtriangle(im, &curr[*K], 0, (float)imgWidth, (float)imgHeight, imgPitch);
}




// similar to addtriangle function, but for output. Not worth looking at...
inline   __device__  void addtriangleproof(float4 * im, triangle * T, float imgWidth, float imgHeight, int imgPitch)
{
	//sort points by y value, yes, this is retarded
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	if(threadIdx.x+threadIdx.y==0) {
		T->a = clip(T->a, 0.0f, 1.0f);
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
		int g = imgPitch * yy + xStart;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			im[g].x += dkEvoAlphaLimit * T->r;
			im[g].y += dkEvoAlphaLimit * T->g;
			im[g].z += dkEvoAlphaLimit * T->b;
			g += kEvoBlockDim;
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
		int g = imgPitch * yy + xStart;
		for(int xx = xStart; xx < xMax; xx += kEvoBlockDim) {
			im[g].x += dkEvoAlphaLimit * T->r;
			im[g].y += dkEvoAlphaLimit * T->g;
			im[g].z += dkEvoAlphaLimit * T->b;
			g += kEvoBlockDim;
		}
	}
}

// similar to render, but for output. Also not worth looking at.
__global__ void renderproof(float4 * im, triangle * curr, int imgWidth, int imgHeight, int imgPitch) {
	for(int y = threadIdx.y; y < imgHeight; y += kEvoBlockDim) {
		int g = y * imgPitch + threadIdx.x;
		for (int x = threadIdx.x; x < imgWidth; x += kEvoBlockDim) {
			im[g].x = 0.0f;
			im[g].y = 0.0f;
			im[g].z = 0.0f;
			im[g].w = 1.0f;
			g += kEvoBlockDim;
		}
	}
	for(int k = 0; k < dkEvoMaxTriangleCount; k++)
		addtriangleproof(im, &curr[k], (float)imgWidth, (float)imgHeight, imgPitch);

	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		int g = yy * imgPitch + threadIdx.x;
		for(int xx = threadIdx.x; xx < imgWidth; xx += kEvoBlockDim) {
			im[g].x = clip(im[g].x,0.0f,1.0f);
			im[g].y = clip(im[g].y,0.0f,1.0f);
			im[g].z = clip(im[g].z,0.0f,1.0f);
			im[g].w = 1.0;
			g += kEvoBlockDim;
		}
	}
}

void getTextureReferences(const textureReference **outRefImg, const textureReference **outCurrImg)
{
	CUDA_CHECK( cudaGetTextureReference(outRefImg, &refimg) );
	CUDA_CHECK( cudaGetTextureReference(outCurrImg, &currimg) );
}

void setGpuConstants(const PsoConstants *constants)
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

void launch_render(float4 *d_im, triangle *d_curr, int *d_currentTriangleIndex, float *d_currentScore, int imgWidth, int imgHeight, int imgPitch)
{
	CUDA_CHECK( cudaMemset(d_currentScore, 0, sizeof(float)) );
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	render<<<gridDim, blockDim>>>(d_im, d_curr, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, imgPitch);
	CUDA_CHECK( cudaGetLastError() );
}

void launch_renderproof(float4 * d_im, triangle * d_curr, int imgWidth, int imgHeight, int imgPitch)
{
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	renderproof<<<gridDim, blockDim>>>(d_im, d_curr, imgWidth, imgHeight, imgPitch);
	CUDA_CHECK( cudaGetLastError() );
}

void launch_run(int32_t particleCount, triangle *d_curr, triangle *d_pos, triangle *d_vel, float *d_fit,
	triangle *d_lbest, float *d_lbval, triangle *d_nbest, float *d_nbval, float *d_gbval,
	int *d_K, int imgWidth, int imgHeight)
{
	dim3 gridDim(particleCount, 1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim, 1);
	//CUDA_CHECK( cudaFuncSetCacheConfig(run, cudaFuncCachePreferL1) ); // No significant difference observed, but this kernel certainly doesn't use smem
	run<<<gridDim, blockDim>>>(d_curr, d_pos, d_vel, d_fit, d_lbest, d_lbval, d_nbest, d_nbval, d_gbval, d_K, (float)imgWidth, (float)imgHeight);
	CUDA_CHECK( cudaGetLastError() );
}
