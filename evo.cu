// Adapted from evo.py
// CS264 Final Project 2009
// by Drew Robb & Joy Ding

typedef unsigned int uint;

// poor man's random number generator
#define rand() (r = r * 1103515245 + 12345, (((float) (r & 65535)) * .000015259))
// this is sketchy, but for purposes of PSO, acceptable (and fast!)

#define max(a,b) (a > b ? a : b)
#define min(a,b) (a > b ? b : a)
#define clip(a,l,h) (max(min(a,h),l))


#define ALPHALIM 0.2            // alpha limit for images
#define ALPHAOFFSET (ALPHALIM/2) //alpha offset
#define CHECKLIM 150 //number of times we have a non-improving value before we terminate
#define SCALE 2 //the factor by which we would like to scale up the image in the final render
#define W ((float) 300) //width of lisa.jpg
#define H ((float) 391) //height of lisa.jpg
#define SIZE 30    // number of particles for PSO
#define KSIZE 16   // grid width/height
#define PSOITERS 1000 // maximum number of iterations at each PSO stage (overridden by checklimit)
#define DIM 800 // number of triangles to render (max)
#define NHOODSIZE 12 // Size of neighbhorhood (PSO topology)



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

//rgb class! initialized with float4's from our texture memory
class rgba {
public:
	float r;
	float g;
	float b;
	float a;
	__device__ rgba() {
	}
	inline __device__ rgba(float4 f) {
		r = f.x;
		g = f.y;
		b = f.z;
		a = f.w;
	}
	
};

//triangle class. stores color/alpha values, as well as coordinates of vertices
class triangle {
public:
	float x1;
	float y1;
	float x2;
	float y2;
	float x3;
	float y3;
	rgba c;
};

//module for creating a float on our device
inline __host__ __device__ float4 operator-(float4 a, float4 b)
{
    return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}
     
texture<float4, 2> currimg;  // current image rendered in triangles
texture<float4, 2> refimg; // original reference image


// adds a triangle to the working image, or subtracts it if add==0
inline   __device__  void addtriangle(rgba * im, triangle * T, bool add)
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
	TT.c.r = ALPHALIM * TT.c.r - ALPHAOFFSET;
	TT.c.g = ALPHALIM * TT.c.g - ALPHAOFFSET;
	TT.c.b = ALPHALIM * TT.c.b - ALPHAOFFSET;
	
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
		m1 = (W/H)*(x2 - x1) / (y2 - y1);
		m2 = (W/H)*(x3 - x1) / (y3 - y1);
		m3 = (W/H)*(x3 - x2) / (y3 - y2);

		swap = 0;
		// enforce that m2 > m1
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = W * x1;
		xt = W * x1;

		// high limits of rows
		h1 = clip(H * y1, 0.0, H);
		h2 = clip(H * y2, 0.0, H);
		h3 = clip(H * y3, 0.0, H);
	}
	__syncthreads();
	if(bad) {return;}

	// shade first half of triangle
	for(int yy = h1 + threadIdx.y; yy < h2; yy += KSIZE) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - H * y1), 0.0, W); 
			i < clip(xt + m2 * (yy - H * y1), 0.0, W); i += KSIZE) {
			int g = W * yy + i;
			im[g].r += TT.c.r;
			im[g].g += TT.c.g;
			im[g].b += TT.c.b;
		}

	}

	// update slopes, row end points for second half of triangle
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * (H * (y2 - y1));
		xt += m2 * (H * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	// shade second half of triangle
	for(int yy = h2 + threadIdx.y; yy < h3; yy += KSIZE) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - H * y2 + 1), 0, W); 
			i < clip(xt + m2 * (yy - H * y2 + 1), 0, W); i += KSIZE) {
			int g = W * yy + i;
			im[g].r += TT.c.r;
			im[g].g += TT.c.g;
			im[g].b += TT.c.b;
		}
	}
}


// calculates the net effect on the score for a a given triangle T
// similar to addtriangle
inline   __device__  void scoretriangle(float * sum, triangle * T)
{

	__shared__ float x1,y1,x2,y2,x3,y3,m1,m2,m3,xs,xt;
	__shared__ int h1,h2,h3,swap,bad;
	triangle TT = *T;
	TT.c.r = clip(TT.c.r, 0.0, 1.0);
	TT.c.g = clip(TT.c.g, 0.0, 1.0);
	TT.c.b = clip(TT.c.b, 0.0, 1.0);
	TT.c.r = ALPHALIM * TT.c.r - ALPHAOFFSET;
	TT.c.g = ALPHALIM * TT.c.g - ALPHAOFFSET;
	TT.c.b = ALPHALIM * TT.c.b - ALPHAOFFSET;
	
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
		m1 = clip((W/H)*(x2 - x1) / (y2 - y1), -H, H);
		m2 = clip((W/H)*(x3 - x1) / (y3 - y1), -H, H);
		m3 = clip((W/H)*(x3 - x2) / (y3 - y2), -H, H);
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}

		// stop and end pixel in first line of triangle
		xs = W * x1;
		xt = W * x1;

		// high limits of rows
		h1 = clip(H * y1, 0.0, H);
		h2 = clip(H * y2, 0.0, H);
		h3 = clip(H * y3, 0.0, H);
	}
	__syncthreads();
	if(bad) {*sum = 0.0; return;}

	// score first half of triangle. This substract the score prior to the last triangle
	float localsum = 0.0;
	for(int yy = threadIdx.y+h1; yy < h2; yy+=KSIZE) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - H * y1), 0.0, W); 
			i < clip(xt + m2 * (yy - H * y1 ), 0.0, W); i += KSIZE) {

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
		xs += m1 * (H * (y2 - y1));
		xt += m2 * (H * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();
		
	// score second half
	for(int yy = threadIdx.y+h2; yy < h3; yy+=KSIZE) {
		for(int i = threadIdx.x + clip(threadIdx.x + xs + m1 * (yy - H * y2 + 1), 0, W);
			i < clip(xt + m2 * (yy - H * y2 + 1), 0, W); i += KSIZE) {

			rgba o = tex2D(currimg, i, yy) - tex2D(refimg, i, yy);
			localsum -= o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);
			o.r += TT.c.r; o.g += TT.c.g; o.b += TT.c.b;
			localsum += o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * (RL * o.r + GL * o.g + BL * o.b);	
		}
	}
	__shared__ float sums[KSIZE];
	if(threadIdx.x == 0) sums[threadIdx.y] = 0.0;
	for(int i = 0; i < KSIZE; i++)
		if(threadIdx.x ==i) sums[threadIdx.y] += localsum;
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		for(int i = 0; i < KSIZE; i++) {
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
					int * M) {
	uint r = pos[0].x1 * 100 + threadIdx.x * 666 + blockIdx.x * 94324 + threadIdx.y * 348;
	__shared__ int check; check = 0;


	// loop over pso updates
	for(int q = 0; q < PSOITERS; q++) {

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
		scoretriangle(&fit[blockIdx.x], &pos[blockIdx.x]);

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
			v = nbval[b % SIZE];
			for(int j = 0; j < NHOODSIZE; j++) {
				if(lbval[(blockIdx.x + j) % SIZE] < v) {
					v = lbval[(blockIdx.x + j) % SIZE];
					b = blockIdx.x + j;
				}
			}
			if(v < nbval[blockIdx.x]) {
				nbval[blockIdx.x] = v;
				nbest[blockIdx.x] = lbest[b % SIZE];
			}	
			// hack to improve early PSO convergence
			else if(lbval[blockIdx.x] > 0) 
				nbval[blockIdx.x] *= 1.1;

		}
		// exit if PSO stagnates
		if(check > CHECKLIM) return;
		__syncthreads();
	}

}




// renders and scores an image
__global__ void render(rgba * im,
					   triangle * curr,
					   int * K,
					   float * score) {

	// clear image
	for(int y = threadIdx.y; y < H; y += KSIZE) {
		for(int i = threadIdx.x; i < W; i += KSIZE) {
			int g = y * W + i;
			im[g].r = 0.0;
			im[g].g = 0.0;
			im[g].b = 0.0;
		}
	}
	// render all triangles
	for(int k = 0; k < DIM; k++)
		addtriangle(im, &curr[k], 1);

	// score the image
	__shared__ float sums[KSIZE*KSIZE];
	sums[KSIZE*threadIdx.y + threadIdx.x] = 0.0;
	for(int yy = threadIdx.y; yy < H; yy+=KSIZE) {
		for(int i = threadIdx.x; i < W; i += KSIZE) {
			int g = yy * W + i;
			rgba o = tex2D(refimg, i, yy);
			o.r -= im[g].r; o.g -= im[g].g; o.b -= im[g].b;
			sums[KSIZE*threadIdx.y+threadIdx.x] += o.r * o.r + o.g * o.g + o.b * o.b + 
				(RL * o.r + GL * o.g + BL * o.b) * 
				(RL * o.r + GL * o.g + BL * o.b);
		}
	}
	__syncthreads();
	*score = 0;
	if(threadIdx.x+threadIdx.y == 0) {
		for(int i = 0; i < KSIZE*KSIZE; i++) {
			*score += sums[i];
		}
	}

	// remove triangles we are modifying
	addtriangle(im, &curr[*K], 0);
}




// similar to addtriangle function, but for output. Not worth looking at...
inline   __device__  void addtriangleproof(rgba * im, triangle * T)
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

		m1 = clip(((SCALE*W)/(SCALE*H))*(x2 - x1) / (y2 - y1), -(SCALE*H), (SCALE*H));
		m2 = clip(((SCALE*W)/(SCALE*H))*(x3 - x1) / (y3 - y1), -(SCALE*H), (SCALE*H));
		m3 = clip(((SCALE*W)/(SCALE*H))*(x3 - x2) / (y3 - y2), -(SCALE*H), (SCALE*H));
		swap = 0;
		if(m1 > m2) {swap = 1; float temp = m1; m1 = m2; m2 = temp;}
		xs = (SCALE*W) * x1;
		xt = (SCALE*W) * x1;
		h1 = clip((SCALE*H) * y1, 0.0, (SCALE*H));
		h2 = clip((SCALE*H) * y2, 0.0, (SCALE*H));
		h3 = clip((SCALE*H) * y3, 0.0, (SCALE*H));
	}
	__syncthreads();

	if(bad) return;
	for(int yy = h1 + threadIdx.y; yy < h2; yy += KSIZE) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - (SCALE*H) * y1), 0.0, (SCALE*W)); 
			i < clip(xt + m2 * (yy - (SCALE*H) * y1), 0.0, (SCALE*W)); i += KSIZE) {
			if(i > (SCALE*W) || i < 0) continue;
			int g = (SCALE*W) * yy + i;

			im[g].r += (ALPHALIM * T->c.r - ALPHAOFFSET);
			im[g].g += (ALPHALIM * T->c.g - ALPHAOFFSET);
			im[g].b += (ALPHALIM * T->c.b - ALPHAOFFSET);

		}

	}
	__syncthreads();
	if(threadIdx.x+threadIdx.y == 0) {
		xs += m1 * ((SCALE*H) * (y2 - y1));
		xt += m2 * ((SCALE*H) * (y2 - y1));
		if(swap) m2 = m3;
		else m1 = m3;
	}
	__syncthreads();

	for(int yy = h2 + threadIdx.y; yy < h3; yy += KSIZE) {
		for(int i = threadIdx.x + clip(xs + m1 * (yy - (SCALE*H) * y2 + 1), 0, (SCALE*W)); 
			i < clip(xt + m2 * (yy - (SCALE*H) * y2 + 1), 0, (SCALE*W)); i += KSIZE) {
			if(i > (SCALE*W) || i < 0) continue;
			int g = (SCALE*W) * yy + i;
			im[g].r += (ALPHALIM * T->c.r - ALPHAOFFSET);
			im[g].g += (ALPHALIM * T->c.g - ALPHAOFFSET);
			im[g].b += (ALPHALIM * T->c.b - ALPHAOFFSET);

		}
	}
	__syncthreads();
}

// similar to render, but for output. Also not worth looking at.
__global__ void renderproof(rgba * im,
					   triangle * curr,
					   float * score) {

	for(int y = threadIdx.y; y < SCALE*H; y += KSIZE) {
		for(int i = threadIdx.x; i < SCALE*W; i += KSIZE) {
			int g = y * SCALE*W + i;
			im[g].r = 0.0;
			im[g].g = 0.0;
			im[g].b = 0.0;
			im[g].a = 1.0;
		}
	}
	for(int k = 0; k < DIM; k++)
		addtriangleproof(im, &curr[k]);

	for(int yy = threadIdx.y; yy < SCALE*H; yy+=KSIZE) {
		for(int i = threadIdx.x; i < SCALE*W; i += KSIZE) {
			int g = yy * SCALE*W + i;
			im[g].r = clip(im[g].r,0.0,1.0);
			im[g].g = clip(im[g].g,0.0,1.0);
			im[g].b = clip(im[g].b,0.0,1.0);
			im[g].a = 1.0;
		}
	}


}
