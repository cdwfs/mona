// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

#include "evogpupso.h"
#include "utils.h"

typedef unsigned int uint;

// poor man's random number generator
#define rand() (r = r * 1103515245 + 12345, (((float) (r & 65535)) * .000015259))
// this is sketchy, but for purposes of PSO, acceptable (and fast!)

#define max(a,b) (a > b ? a : b)
#define min(a,b) (a > b ? b : a)
#define clip(a,l,h) (max(min(a,h),l))

//luminance =  0.3 R + 0.59 G + 0.11 B
#define RL 0.3
#define GL 0.59
#define BL 0.11

//swap float module for swapping values in evaluation
inline __host__ __device__ void swap(float& a, float& b) {
	float temp = a;
	a = b;
	b = temp;
}

//module for creating a float on our device
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
     
texture<float4, cudaTextureType2D, cudaReadModeElementType> currimg;  // current image rendered in triangles
texture<float4, cudaTextureType2D, cudaReadModeElementType> refimg; // original reference image


// adds a triangle to the working image, or subtracts it if add==0
inline   __device__  void addtriangle(rgba * im, triangle * T, bool add, int imgWidth, int imgHeight, int imgPitch)
{
	//intializes shared memory
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	//copies over image
	triangle TT = *T;

	//clip color values to valid range
	TT.c.r = clip(TT.c.r, 0.0, 1.0);
	TT.c.g = clip(TT.c.g, 0.0, 1.0);
	TT.c.b = clip(TT.c.b, 0.0, 1.0);

	//if we are subtracting the triangle, set values to -1
	if(add == 0) {
		TT.c.r *= -1;
		TT.c.g *= -1;
		TT.c.b *= -1;
	}
	
	//set to alpha values that we are using 
	TT.c.r = kEvoAlphaLimit * TT.c.r - kEvoAlphaOffset;
	TT.c.g = kEvoAlphaLimit * TT.c.g - kEvoAlphaOffset;
	TT.c.b = kEvoAlphaLimit * TT.c.b - kEvoAlphaOffset;
	
	if(threadIdx.y+threadIdx.x == 0) {
		// sort points by y value so that we can render triangles properly
		bad = 0;
		if     (TT.y1 < TT.y2 && TT.y2 < TT.y3) {
			x1 = TT.x1; y1 = TT.y1; x2 = TT.x2; y2 = TT.y2; x3 = TT.x3; y3 = TT.y3;
		} 
		else if(TT.y1 < TT.y3 && TT.y3 < TT.y2) {
			x1 = TT.x1; y1 = TT.y1; x2 = TT.x3; y2 = TT.y3; x3 = TT.x2; y3 = TT.y2;
		}
		else if(TT.y2 < TT.y1 && TT.y1 < TT.y3) {
			x1 = TT.x2; y1 = TT.y2; x2 = TT.x1; y2 = TT.y1; x3 = TT.x3; y3 = TT.y3;
		}
		else if(TT.y2 < TT.y3 && TT.y3 < TT.y1) {
			x1 = TT.x2; y1 = TT.y2; x2 = TT.x3; y2 = TT.y3; x3 = TT.x1; y3 = TT.y1;
		}
		else if(TT.y3 < TT.y1 && TT.y1 < TT.y2) {
			x1 = TT.x3; y1 = TT.y3; x2 = TT.x1; y2 = TT.y1; x3 = TT.x2; y3 = TT.y2;
		}
		else if(TT.y3 < TT.y2 && TT.y2 < TT.y1) {
			x1 = TT.x3; y1 = TT.y3; x2 = TT.x2; y2 = TT.y2; x3 = TT.x1; y3 = TT.y1;
		}
		// flag if something isn't right...
		else bad = 1;

		// calculate slopes
		m1 = ((float)imgWidth/(float)imgHeight)*(x2 - x1) / (y2 - y1);
		m2 = ((float)imgWidth/(float)imgHeight)*(x3 - x1) / (y3 - y1);
		m3 = ((float)imgWidth/(float)imgHeight)*(x3 - x2) / (y3 - y2);

		swap = 0;
		// enforce that m2 > m1
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = (float)imgWidth * x1;
		xt = (float)imgWidth * x1;

		// high limits of rows
		h1 = clip((float)imgHeight * y1, 0.0, (float)imgHeight);
		h2 = clip((float)imgHeight * y2, 0.0, (float)imgHeight);
		h3 = clip((float)imgHeight * y3, 0.0, (float)imgHeight);
	}
	__syncthreads();
	if(bad) {return;}

	// shade first half of triangle
	for(int yy = h1 + threadIdx.y; yy < h2; yy += kEvoBlockDim) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0, (float)imgWidth); 
			i < clip(xt + m2 * (yy - imgHeight * y1), 0.0, (float)imgWidth); i += kEvoBlockDim) {
			int g = imgPitch * yy + i;
			im[g].r += TT.c.r;
			im[g].g += TT.c.g;
			im[g].b += TT.c.b;
		}

	}

	// update slopes, row end points for second half of triangle
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * ((float)imgHeight * (y2 - y1));
		xt += m2 * ((float)imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	// shade second half of triangle
	for(int yy = h2 + threadIdx.y; yy < h3; yy += kEvoBlockDim) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y2 + 1), 0, (float)imgHeight); 
			i < clip(xt + m2 * (yy - imgHeight * y2 + 1), 0, (float)imgWidth); i += kEvoBlockDim) {
			int g = imgPitch * yy + i;
			im[g].r += TT.c.r;
			im[g].g += TT.c.g;
			im[g].b += TT.c.b;
		}
	}
}


// calculates the net effect on the score for a a given triangle T
// similar to addtriangle
inline   __device__  void scoretriangle(float * sum, triangle * T, int imgWidth, int imgHeight)
{

	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	triangle TT = *T;
	TT.c.r = clip(TT.c.r, 0.0, 1.0);
	TT.c.g = clip(TT.c.g, 0.0, 1.0);
	TT.c.b = clip(TT.c.b, 0.0, 1.0);
	TT.c.r = kEvoAlphaLimit * TT.c.r - kEvoAlphaOffset;
	TT.c.g = kEvoAlphaLimit * TT.c.g - kEvoAlphaOffset;
	TT.c.b = kEvoAlphaLimit * TT.c.b - kEvoAlphaOffset;
	
	if(threadIdx.y+threadIdx.x == 0) {
		// sort points by y value, yes, this is retarded
		bad = 0;
		if     (TT.y1 < TT.y2 && TT.y2 < TT.y3) {
			x1 = TT.x1; y1 = TT.y1; x2 = TT.x2; y2 = TT.y2; x3 = TT.x3; y3 = TT.y3;
		} 
		else if(TT.y1 < TT.y3 && TT.y3 < TT.y2) {
			x1 = TT.x1; y1 = TT.y1; x2 = TT.x3; y2 = TT.y3; x3 = TT.x2; y3 = TT.y2;
		}
		else if(TT.y2 < TT.y1 && TT.y1 < TT.y3) {
			x1 = TT.x2; y1 = TT.y2; x2 = TT.x1; y2 = TT.y1; x3 = TT.x3; y3 = TT.y3;
		}
		else if(TT.y2 < TT.y3 && TT.y3 < TT.y1) {
			x1 = TT.x2; y1 = TT.y2; x2 = TT.x3; y2 = TT.y3; x3 = TT.x1; y3 = TT.y1;
		}
		else if(TT.y3 < TT.y1 && TT.y1 < TT.y2) {
			x1 = TT.x3; y1 = TT.y3; x2 = TT.x1; y2 = TT.y1; x3 = TT.x2; y3 = TT.y2;
		}
		else if(TT.y3 < TT.y2 && TT.y2 < TT.y1) {
			x1 = TT.x3; y1 = TT.y3; x2 = TT.x2; y2 = TT.y2; x3 = TT.x1; y3 = TT.y1;
		}
		// flag if something isn't right...
		else bad = 1;

		// calculate slopes
		m1 = clip(((float)imgWidth/(float)imgHeight)*(x2 - x1) / (y2 - y1), -(float)imgHeight, (float)imgHeight);
		m2 = clip(((float)imgWidth/(float)imgHeight)*(x3 - x1) / (y3 - y1), -(float)imgHeight, (float)imgHeight);
		m3 = clip(((float)imgWidth/(float)imgHeight)*(x3 - x2) / (y3 - y2), -(float)imgHeight, (float)imgHeight);
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = (float)imgWidth * x1;
		xt = (float)imgWidth * x1;

		// high limits of rows
		h1 = clip((float)imgHeight * y1, 0.0, (float)imgHeight);
		h2 = clip((float)imgHeight * y2, 0.0, (float)imgHeight);
		h3 = clip((float)imgHeight * y3, 0.0, (float)imgHeight);
	}
	__syncthreads();
	if(bad) {*sum = 0.0; return;}

	// score first half of triangle. This substract the score prior to the last triangle
	float localsum = 0.0;
	for(int yy = threadIdx.y+h1; yy < h2; yy+=kEvoBlockDim) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0, (float)imgWidth); 
			i < clip(xt + m2 * (yy - imgHeight * y1 ), 0.0, (float)imgWidth); i += kEvoBlockDim) {

			rgba o = tex2D(currimg, i, yy) - tex2D(refimg, i, yy);
			localsum -= o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);
			o.r += TT.c.r; o.g += TT.c.g; o.b += TT.c.b;
			localsum += o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);
		}
	}
	
	// update slopes and limits to score second half of triangle
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * ((float)imgHeight * (y2 - y1));
		xt += m2 * ((float)imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();
		
	// score second half
	for(int yy = threadIdx.y+h2; yy < h3; yy+=kEvoBlockDim) {
		for(int i = threadIdx.x + clip(threadIdx.x + xs + m1 * (yy - imgHeight * y2 + 1), 0, (float)imgWidth);
			i < clip(xt + m2 * (yy - imgHeight * y2 + 1), 0, (float)imgWidth); i += kEvoBlockDim) {

			rgba o = tex2D(currimg, i, yy) - tex2D(refimg, i, yy);
			localsum -= o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);
			o.r += TT.c.r; o.g += TT.c.g; o.b += TT.c.b;
			localsum += o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);	
		}
	}
	__shared__ float sums[kEvoBlockDim];
	if(threadIdx.x == 0) sums[threadIdx.y] = 0.0;
	for(int i = 0; i < kEvoBlockDim; i++)
		if(threadIdx.x ==i) sums[threadIdx.y] += localsum;
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		for(int i = 0; i < kEvoBlockDim; i++) {
			*sum += sums[i];
		}
	}
	__syncthreads();
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
					int imgWidth,
					int imgHeight) {
	uint r = pos[0].x1 * 100 + threadIdx.x * 666 + blockIdx.x * 94324 + threadIdx.y * 348;
	__shared__ int check; check = 0;


	// loop over pso updates
	for(int q = 0; q < kEvoPsoIterationCount; q++) {

		// integrate position
		if(q > 0 && threadIdx.y==0 && threadIdx.x < 10) {
			float vmax = .2 * rand() + 0.05;
			float vmin = -.2 * rand() - 0.05;	
			float * v = (((float *) &vel[blockIdx.x]) + threadIdx.x);
			float * p = (((float *) &pos[blockIdx.x]) + threadIdx.x);
			float * l = (((float *) &lbest[blockIdx.x]) + threadIdx.x);
			float * n = (((float *) &nbest[blockIdx.x]) + threadIdx.x);
			*v *= .85;
			*v += 0.70 * rand() * (*n - *p);
			*v += 0.70 * rand() * (*l - *p);
			*v = max(*v, vmin);
			*v = min(*v, vmax);
			*p = *p + *v;
			if(fit[blockIdx.x] > 0 && rand() < 0.01)
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
			// hack to improve early PSO convergence
			else if(lbval[blockIdx.x] > 0) { 
				lbval[blockIdx.x] *= 1.1;
			}

			// global max find
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
			v = nbval[b % kEvoPsoParticleCount];
			for(int j = 0; j < kEvoPsoNeighborhoodSize; j++) {
				if(lbval[(blockIdx.x + j) % kEvoPsoParticleCount] < v) {
					v = lbval[(blockIdx.x + j) % kEvoPsoParticleCount];
					b = blockIdx.x + j;
				}
			}
			if(v < nbval[blockIdx.x]) {
				nbval[blockIdx.x] = v;
				nbest[blockIdx.x] = lbest[b % kEvoPsoParticleCount];
			}	
			// hack to improve early PSO convergence
			else if(lbval[blockIdx.x] > 0) 
				nbval[blockIdx.x] *= 1.1;

		}
		// exit if PSO stagnates
		if(check > kEvoCheckLimit) return;
		__syncthreads();
	}

}




// renders and scores an image
__global__ void render(rgba * im,
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
			im[g].r = 0.0;
			im[g].g = 0.0;
			im[g].b = 0.0;
		}
	}
	// render all triangles
	for(int k = 0; k < kEvoMaxTriangleCount; k++)
		addtriangle(im, &curr[k], 1, imgWidth, imgHeight, imgPitch);

	// score the image
	__shared__ float sums[kEvoBlockDim*kEvoBlockDim];
	sums[kEvoBlockDim*threadIdx.y + threadIdx.x] = 0.0;
	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		for(int i = threadIdx.x; i < imgWidth; i += kEvoBlockDim) {
			int g = yy * imgPitch + i;
			rgba o = tex2D(refimg, i, yy);
			o.r -= im[g].r; o.g -= im[g].g; o.b -= im[g].b;
			sums[kEvoBlockDim*threadIdx.y+threadIdx.x] += o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * 
				(RL * o.r + GL * o.g + BL * o.b);
		}
	}
	__syncthreads();
	*score = 0;
	if(threadIdx.x+threadIdx.y == 0) {
		for(int i = 0; i < kEvoBlockDim*kEvoBlockDim; i++) {
			*score += sums[i];
		}
	}

	// remove triangles we are modifying
	addtriangle(im, &curr[*K], 0, imgWidth, imgHeight, imgPitch);
}




// similar to addtriangle function, but for output. Not worth looking at...
inline   __device__  void addtriangleproof(rgba * im, triangle * T, int imgWidth, int imgHeight, int imgPitch)
{
	//sort points by y value, yes, this is retarded
	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	if(threadIdx.x+threadIdx.y==0) {
		T->c.a = clip(T->c.a, 0.0, 1.0);
		T->c.r = clip(T->c.r, 0.0, 1.0);
		T->c.g = clip(T->c.g, 0.0, 1.0);
		T->c.b = clip(T->c.b, 0.0, 1.0);
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

		m1 = clip((((float)imgWidth)/((float)imgHeight))*(x2 - x1) / (y2 - y1), -(float)imgHeight, (float)imgHeight);
		m2 = clip((((float)imgWidth)/((float)imgHeight))*(x3 - x1) / (y3 - y1), -(float)imgHeight, (float)imgHeight);
		m3 = clip((((float)imgWidth)/((float)imgHeight))*(x3 - x2) / (y3 - y2), -(float)imgHeight, (float)imgHeight);
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}
		xs = (float)imgWidth * x1;
		xt = (float)imgWidth * x1;
		h1 = clip((float)imgHeight * y1, 0.0, (float)imgHeight);
		h2 = clip((float)imgHeight * y2, 0.0, (float)imgHeight);
		h3 = clip((float)imgHeight * y3, 0.0, (float)imgHeight);
	}
	__syncthreads();

	if(bad) return;
	for(int yy = h1 + threadIdx.y; yy < h2; yy += kEvoBlockDim) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y1), 0.0, (float)imgWidth); 
			i < clip(xt + m2 * (yy - imgHeight * y1), 0.0, (float)imgWidth); i += kEvoBlockDim) {
			if(i > imgWidth || i < 0) continue;
			int g = imgPitch * yy + i;

			im[g].r += (kEvoAlphaLimit * T->c.r - kEvoAlphaOffset);
			im[g].g += (kEvoAlphaLimit * T->c.g - kEvoAlphaOffset);
			im[g].b += (kEvoAlphaLimit * T->c.b - kEvoAlphaOffset);

		}

	}
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * ((float)imgHeight * (y2 - y1));
		xt += m2 * ((float)imgHeight * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	for(int yy = h2 + threadIdx.y; yy < h3; yy += kEvoBlockDim) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - imgHeight * y2 + 1), 0, imgWidth); 
			i < clip(xt + m2 * (yy - imgHeight * y2 + 1), 0, imgWidth); i += kEvoBlockDim) {
			if(i > imgWidth || i < 0) continue;
			int g = imgPitch * yy + i;
			im[g].r += (kEvoAlphaLimit * T->c.r - kEvoAlphaOffset);
			im[g].g += (kEvoAlphaLimit * T->c.g - kEvoAlphaOffset);
			im[g].b += (kEvoAlphaLimit * T->c.b - kEvoAlphaOffset);

		}
	}
	__syncthreads();
}

// similar to render, but for output. Also not worth looking at.
__global__ void renderproof(rgba * im,
					   triangle * curr, int imgWidth, int imgHeight, int imgPitch) {
#if 1
	for(int y = threadIdx.y; y < imgHeight; y += kEvoBlockDim) {
		for(int i = threadIdx.x; i < imgWidth; i += kEvoBlockDim) {
			int g = y * imgPitch + i;
			im[g].r = 0.0;
			im[g].g = 0.0;
			im[g].b = 0.0;
			im[g].a = 1.0;
		}
	}
	for(int k = 0; k < kEvoMaxTriangleCount; k++)
		addtriangleproof(im, &curr[k], imgWidth, imgHeight, imgPitch);

	for(int yy = threadIdx.y; yy < imgHeight; yy+=kEvoBlockDim) {
		for(int i = threadIdx.x; i < imgWidth; i += kEvoBlockDim) {
			int g = yy * imgPitch + i;
			im[g].r = clip(im[g].r,0.0,1.0);
			im[g].g = clip(im[g].g,0.0,1.0);
			im[g].b = clip(im[g].b,0.0,1.0);
			im[g].a = 1.0;
		}
	}
#else
	for(int y = threadIdx.y; y < imgHeight; y += kEvoBlockDim) {
		for(int i = threadIdx.x; i < imgWidth; i += kEvoBlockDim) {
			int g = y * imgPitch + i;
			rgba o = tex2D(refimg, i, y);
			im[g].r = o.r;
			im[g].g = o.g;
			im[g].b = o.b;
			im[g].a = o.a;
		}
	}
#endif
}

void getTextureReferences(const textureReference **outRefImg, const textureReference **outCurrImg)
{
	CUDA_CHECK( cudaGetTextureReference(outRefImg, &refimg) );
	CUDA_CHECK( cudaGetTextureReference(outCurrImg, &currimg) );
}

void launch_render(rgba *d_im, triangle *d_curr, int *d_currentTriangleIndex, float *d_currentScore, int imgWidth, int imgHeight, int imgPitch)
{
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	render<<<gridDim, blockDim>>>(d_im, d_curr, d_currentTriangleIndex, d_currentScore, imgWidth, imgHeight, imgPitch);
	CUDA_CHECK( cudaGetLastError() );
}

void launch_renderproof(rgba * d_im, triangle * d_curr, int imgWidth, int imgHeight, int imgPitch)
{
	dim3 gridDim(1,1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim);
	renderproof<<<gridDim, blockDim>>>(d_im, d_curr, imgWidth, imgHeight, imgPitch);
	CUDA_CHECK( cudaGetLastError() );
}

void launch_run(triangle *d_curr, triangle *d_pos, triangle *d_vel, float *d_fit,
	triangle *d_lbest, float *d_lbval, triangle *d_nbest, float *d_nbval, float *d_gbval,
	int *d_K, int imgWidth, int imgHeight)
{
	dim3 gridDim(kEvoPsoParticleCount, 1);
	dim3 blockDim(kEvoBlockDim, kEvoBlockDim, 1);
	run<<<gridDim, blockDim>>>(d_curr, d_pos, d_vel, d_fit, d_lbest, d_lbval, d_nbest, d_nbval, d_gbval, d_K, imgWidth, imgHeight);
	CUDA_CHECK( cudaGetLastError() );
}