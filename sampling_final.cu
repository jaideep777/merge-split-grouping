#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "utils/simple_timer.h"
#include "utils/simple_math.h"
#include "utils/cuda_device.h"
#include "utils/cuda_vector_math.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
using namespace std;


/*------------------------------------------------------------------------------

	periodic pairwise distances on CPU and GPU
	returns f(x) where x is pairwise distance

------------------------------------------------------------------------------*/
void calc_pairwise(float2 * pos, float * out, int n, float ki, float L, float dL){
	for (int i=0; i<n; ++i){
		for (int j=0; j<n; ++j){
			//float2 x = periodic_displacement()
			float x = length(periodicDisplacement(pos[i], pos[j], L, L));
			out[i*n+j] = expf(-x*x/2/ki/ki);
		}
	}
}



__global__ void pairwise_distance_kernel(float2 * pos, float * pd, int nc, float ki, float L, float dL){
	int tx = threadIdx.x, ty = threadIdx.y;
	int bx = blockIdx.x, by = blockIdx.y;
	
	int o = by*16*nc + ty*nc + bx*16 + tx;
	int i = 16*by + ty;
	int j = 16*bx + tx;
	float x = length(periodicDisplacement(pos[i], pos[j], L, L));

	//pd[o] = x;	
	pd[o] = expf(-x*x/2/ki/ki);
}


/*------------------------------------------------------------------------------

	Roulette sampling on GPU

------------------------------------------------------------------------------*/
__global__ void sample_roulette_kernel(int nc, float* prob, float* cumm_prob_all, int * ids, curandState* p_rngStates){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nc) return;
	
	// create ranges array 
	float * cumm_prob = &cumm_prob_all[tid*(nc+1)];	// cumm_prob_all row #tid
	cumm_prob[0] = 0;
	for (int i=0; i<nc; ++i) cumm_prob[i+1] = cumm_prob[i] + prob[tid*nc + i]*float(i!=tid);

	// generate range selector
	float a = 1-curand_uniform(&p_rngStates[tid]);  // a is range selector. Must be in [0,1)
	a *= cumm_prob[nc];				// transform a into [0, sum(weights) )

	// get least upper bound
	int r = bin_search_lub(a, cumm_prob, nc+1); 

	// we want r-1 because upper bound is the right edge of the range for the desired element
	ids[tid] = r-1;
	
}


/*------------------------------------------------------------------------------

	Rejection sampling on GPU

------------------------------------------------------------------------------*/
__global__ void sample_reject_kernel(int nc, float kisd, float* pd, int * ids, int *iters, curandState* p_rngStates, float2*pos, float L){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nc) return;
	
	bool accept = false;
	int r = -1;
	int niter=0;
	while(!accept){
		// choose a random individual other than self
		int chosen_one = (1-curand_uniform(&p_rngStates[tid]))*(nc-1);
		int self = (chosen_one == tid);
		chosen_one = self*(nc-1) + (1-self)*chosen_one;

		// get probability of choosing for imitation from imitation kernel
		float prob = pd[tid*nc + chosen_one];
		
		// if rnd is < prob, accept.
		if (curand_uniform(&p_rngStates[tid]) < prob){
			r = chosen_one;
			accept=true;
		}
		
		// if iters exceed limit, just choose self.. i.e. no imitation
		++niter;
		if (niter > 5000){
			r = 0;
			break;
		}
	}
	ids[tid] = r;	// the id of the individual that consumer tid chooses to imitate
	iters[tid] = niter; // number of iterations required for choosing
			
}






// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// KERNEL to set up RANDOM GENERATOR STATES
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void rngStateSetup_kernel(int * rng_Seeds, curandState * rngStates){
	//int tid = threadIdx.x;							// each block produces exactly the same random numbers
	int tid_u = threadIdx.x + blockIdx.x*blockDim.x;	// each block produces different random numbers
	curand_init (rng_Seeds[tid_u], 0, 0, &rngStates[tid_u]);
}

// Wrapper for state setup kernel 
void launch_rngStateSetup_kernel(int * rng_blockSeeds, curandState * rngStates, int nstates){
	unsigned int blocksize = 256; 
	unsigned int nblocks = (nstates-1)/blocksize + 1;
	rngStateSetup_kernel  <<<nblocks, blocksize >>> (rng_blockSeeds, rngStates);
}



int main(){
	
	int nc=512;
	cout << "Enter Ki: ";
	float kI = 20;
	cin >> kI;
	float L = 225;

	SimpleTimer T;
	ofstream fout;

	// set up RNG on GPU
	curandState * dev_XWstates;
	int *seeds_h, *seeds_dev; 	

	seeds_h = new int[nc];
	cudaMalloc((void**)&seeds_dev,    nc*sizeof(int)); // rng seeds
	cudaMalloc((void**)&dev_XWstates, nc*sizeof(curandState));	// rng states
	
	for (int i=0; i<nc; ++i) seeds_h[i] = rand(); 
	cudaMemcpy( seeds_dev, seeds_h, sizeof(int)*nc, cudaMemcpyHostToDevice);
	launch_rngStateSetup_kernel(seeds_dev, dev_XWstates, nc);
	getLastCudaError("RNG_kernel_launch");
	
	

	// create pos vector (random positions)
	float2 pos[nc];
	for (int i=0; i<nc; ++i) pos[i] = make_float2(runif()*L, runif()*L);
	float out[nc*nc];

	float2 * pos_dev;
	float * out_dev;
	float * out_h = new float[nc*nc];
	cudaMalloc((void**)&pos_dev, nc*sizeof(float2));
	cudaMalloc((void**)&out_dev, nc*nc*sizeof(float));
	cudaMemcpy(pos_dev, pos, nc*sizeof(float2), cudaMemcpyHostToDevice);
	
	//----------------------------------------------------------------------//

	// pairwise distances on GPU
	T.reset(); T.start();
	cout << "pairwise distances gpu...";
	dim3 nt(16,16);
	dim3 nb((nc-1)/16+1, (nc-1)/16+1); 
	pairwise_distance_kernel <<<nb, nt >>> (pos_dev, out_dev, nc, kI, L, 0);
	getLastCudaError("pd_kernel");
	cudaMemcpy(out_h, out_dev, nc*nc*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	T.stop();
	T.printTime();

	// pairwise distances on CPU
	T.reset(); T.start();
	cout << "pairwise distances cpu...";
	calc_pairwise(pos, out, nc, kI, L, 0);
	T.stop();
	T.printTime();
	
	cout << "compare...";
	fout.open("pd.txt");
	int sum=0;
	for (int i=0; i<nc*nc; ++i){
		if (i%nc == 0) fout << endl;
		if (fabs(out[i]-out_h[i]) > 0.0001)  sum++;
		//cout << out[i] << " " << out_h[i] << endl;
		fout << out[i] << "\t";
	}
	fout.close();
	cout << sum << " discrepancies." << endl;

	//----------------------------------------------------------------------//

	// sample_reject on GPU
	int ns = nc;
	int * ids = new int[ns];
	int *iters = new int[ns];
	int * ids_dev, *iters_dev;
	cudaMalloc((void**)&ids_dev, ns*sizeof(int));
	cudaMalloc((void**)&iters_dev, ns*sizeof(int));
	T.reset(); T.start();
	cout << "reject gpu...";
	fout.open("reject_out_gpu.txt");
	for (int t=0; t<1; ++t){
		sample_reject_kernel <<<(ns-1)/256+1, 256 >>> (nc, kI, out_dev, ids_dev, iters_dev, dev_XWstates, pos_dev, L);
		getLastCudaError("sample_reject kernel");
		cudaMemcpy(ids, ids_dev, ns*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(iters, iters_dev, ns*sizeof(int), cudaMemcpyDeviceToHost);

		for (int i=0; i<ns; ++i) fout << ids[i] << "\t";
		fout << "\n";
	}
	cudaDeviceSynchronize();
	T.stop();
	T.printTime();
//	for (int i=0; i<ns; ++i) fout << ids[i] << "\t" << iters[i] << "\n";
	fout.close();	

	//----------------------------------------------------------------------//

	// sample_roulette on GPU
	float* cumm_prob_dev;
	cudaMalloc((void**)&cumm_prob_dev, nc*(nc+1)*sizeof(float));

	T.reset(); T.start();
	cout << "roulette gpu...";
	fout.open("roulette_out_gpu.txt");
	for (int t=0; t<1; ++t){
		sample_roulette_kernel <<<(nc-1)/256+1, 256 >>> (nc, out_dev, cumm_prob_dev, ids_dev, dev_XWstates);
		getLastCudaError("sample_roulette kernel");
		cudaMemcpy(ids, ids_dev, ns*sizeof(int), cudaMemcpyDeviceToHost);

		for (int i=0; i<ns; ++i) fout << ids[i] << "\t";
		fout << "\n";
	}
	cudaDeviceSynchronize();
	T.stop();
	T.printTime();
//	for (int i=0; i<ns; ++i) fout << ids[i] << "\t" << iters[i] << "\n";
	fout.close();	



	return 0;
}







