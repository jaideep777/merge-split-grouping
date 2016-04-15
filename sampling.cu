#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "utils/simple_timer.h"
#include "utils/cuda_device.h"
#include "utils/cuda_vector_math.cuh"
#include <cuda_runtime.h>
#include <curand_kernel.h>
using namespace std;

/*---------------------------------------------------------------------------------------------

       0   1      2      3 4      5               6            <-- element of the range (1:n)
	|----|---|---------|--|-|------------|-----------------|
	0    1   2    ^    3  4 5            6                 7   <-- range edges (cumm_prob)
	              |    ^
	              |    |__ upper bound = 1st element > range selector
	              |
                  |__ range selector (selected element = 2 = upper_bound - 1)

---------------------------------------------------------------------------------------------*/
int sample_roulette(float * weights, int n, double * ranges= NULL){

	// create ranges array if not specified
	double cumm_prob[n+1]; 
	cumm_prob[0] = 0;
	for (int i=0; i<n; ++i) cumm_prob[i+1] = cumm_prob[i] + weights[i];

	double a = double(rand())/RAND_MAX*(1-1e-12);  // a is range selector. Must be in [0,1)
	a *= cumm_prob[n];				// transform a into [0, sum(weights) )

	int r = 0; 	// selected element

	// binary search init
	int lo = 0;
	int hi = n;
	int mid = (hi+lo)/2;

	// search for lower bound, then increment it till we go just beyond a
	while(hi != mid && lo != mid){
		if (cumm_prob[mid] > a){
			hi = mid;
			mid = (hi+lo)/2;
		}
		else{
			lo = mid;
			mid = (hi+lo)/2;
		}
	}
	r = lo;

	// increment r until lowest number > a is reached
	while(cumm_prob[r] <= a){
		++r;
	}

	// we want r-1 because upper bound is the right edge of the range for the desired element
	return r-1;
	
}  


// sampling using rejection algorithm
int sample_reject(float * weights, int n){
	bool accept = false;
	int r = -1;
	while(!accept){
		int chosen_one = rand() % n;
		if (double(rand())/RAND_MAX*(1-1e-12) < weights[chosen_one]){
			r = chosen_one;
			accept=true;
		}
	}
	return r;
}


// sampling using rejection algorithm
int sample_reject(float2 * pos, float* pd, int tid, float kisd, float L, int n){
	bool accept = false;
	int r = -1;
	while(!accept){
		int chosen_one = rand() % (n-1);
		if (chosen_one == tid) chosen_one = n-1;
//		float x = length(periodicDisplacement(pos[tid], pos[chosen_one], L, L));
//		float prob = exp(-x*x/5/5/2);
		float prob = pd[tid*n + chosen_one];
		if (tid == chosen_one) prob=0;
		if (float(rand())/RAND_MAX*(1-1e-12) < prob){
			r = chosen_one;
			accept=true;
		}
	}
	return r;
}



// pairwise distances calculation

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




__global__ void sample_roulette_kernel(int nc, float* prob, float* cumm_prob, int * ids, curandState* p_rngStates){
	int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid >= nc) return;
	
	// create ranges array if not specified
	//double cumm_prob[n+1]; 
	cumm_prob[tid*(nc+1) + 0] = 0;
	for (int i=0; i<nc; ++i) cumm_prob[tid*(nc+1) + i+1] = cumm_prob[tid*(nc+1) + i] + prob[tid*nc + i]*float(i!=tid);

	float a = 1-curand_uniform(&p_rngStates[tid]);  // a is range selector. Must be in [0,1)
	a *= cumm_prob[tid*(nc+1) + nc];				// transform a into [0, sum(weights) )

	int r = 0; 	// selected element

	// binary search init
	int lo = 0;
	int hi = nc;
	int mid = (hi+lo)/2;

	// search for lower bound, then increment it till we go just beyond a
	while(hi != mid && lo != mid){
		if (cumm_prob[tid*(nc+1) + mid] > a){
			hi = mid;
			mid = (hi+lo)/2;
		}
		else{
			lo = mid;
			mid = (hi+lo)/2;
		}
	}
	r = lo;

	// increment r until lowest number > a is reached
	while(cumm_prob[tid*(nc+1) + r] <= a){
		++r;
	}

	// we want r-1 because upper bound is the right edge of the range for the desired element
	ids[tid] = r-1;
	
	
}



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

		// calc distance from self to other individual
		//float dist = pd[chosen_one]; //pd[tid*nc + chosen_one];
		//float dist = length(periodicDisplacement(pos[tid], pos[chosen_one], L, L));

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
	
	int N = 512;

	// create triangle shaped parwise distance distribution
	float d_x[N];
	float w_x[N];
	for (int i=0; i<N; ++i) d_x[i] = i/1000.f*318.f;	// distances vector to sample from.. distances are from 0 - 70
	for (int i=0; i<N*0.7; ++i) w_x[i] = i/(N*0.7);
	for (int i=N*0.7; i<N; ++i) w_x[i] = 1-(i-0.7*N)/(0.3*N);
	
	float x[N];
	for (int i=0; i<N; ++i){
		int k = sample_roulette(w_x, N);
		x[i] = d_x[k];
	}
	
	float prob[N];
	double sd = 5;
	for (int i=0; i<N; ++i) {
		prob[i] = exp(-x[i]*x[i]/2/sd/sd);
		if (prob[i] < 0.0001) prob[i] = 0;
	}
	

	ofstream fout("roulette_setup.txt");
	for (int i=0; i<N; ++i){
		fout << d_x[i] << "\t" << w_x[i]<< "\t" << x[i] << "\t" << prob[i] << endl;
	}
	fout.close();
	

	SimpleTimer T; T.reset();
	fout.open("roulette_out.txt");
	cout << "roulette...";
	T.start();
	for (int i=0; i< 512; ++i){
		int k = sample_roulette(prob, N);
		fout << k << endl;
	}
	T.stop();
	T.printTime();
	fout.close();


	fout.open("reject_out.txt");
	T.reset(); T.start();
	cout << "reject...";
	for (int i=0; i< 512; ++i){
		int k = sample_reject(prob, N);
		fout << k << endl;
	}
	T.stop();
	T.printTime();
	fout.close();
	
	float prob1[] = {5, 2, 0, 0, 0, 0, 10, 10, 0.01, 5};
	ofstream fout1("roulette_out_discrete.txt");
	for (int i=0; i< 10000; ++i){
		int k = sample_roulette(prob1, 10);
		fout1 << k << endl;
	}
	
	
	// Check pairwise distances calc
	int nc=512;
	float kI = 5;
	float2 pos[nc];
	for (int i=0; i<nc; ++i) pos[i] = make_float2(runif()*225, runif()*225);
	float out[nc*nc];

	float2 * pos_dev;
	float * out_dev;
	float * out_h = new float[nc*nc];
	cudaMalloc((void**)&pos_dev, nc*sizeof(float2));
	cudaMalloc((void**)&out_dev, nc*nc*sizeof(float));
	cudaMemcpy(pos_dev, pos, nc*sizeof(float2), cudaMemcpyHostToDevice);
	
	// pairwise distances on GPU
	T.reset(); T.start();
	cout << "pairwise distances gpu...";
	dim3 nt(16,16);
	dim3 nb((nc-1)/16+1, (nc-1)/16+1); 
	pairwise_distance_kernel <<<nb, nt >>> (pos_dev, out_dev, nc, kI, 225, 0);
	getLastCudaError("pd_kernel");
	cudaMemcpy(out_h, out_dev, nc*nc*sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	T.stop();
	T.printTime();

	// pairwise distances on CPU
	T.reset(); T.start();
	cout << "pairwise distances cpu...";
	calc_pairwise(pos, out, nc, kI, 225, 0);
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
		sample_reject_kernel <<<(ns-1)/256+1, 256 >>> (nc, kI, out_dev, ids_dev, iters_dev, dev_XWstates, pos_dev, 225);
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

	
	// sample_reject_pop
	cout << "sample reject pop...";
	fout.open("reject_out_pop.txt");
	T.reset(); T.start();
	for (int i=0; i< 512; ++i){
		int k = sample_reject(pos, out, i, kI, 225, nc);
		fout << k << "\t";
	}
	fout << endl;
	T.stop();
	T.printTime();
	fout.close();



	return 0;
}







