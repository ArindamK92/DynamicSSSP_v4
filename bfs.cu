//#include "bfs.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)

//Use of below code:
//If our code in a.c is not using C++ features, and we want to call it from main.cpp which does use C++ features, then in a.h we need to add the following
#ifdef __cplusplus
extern "C" {
#endif

	void bfsGPU(int start, int nodes, int* SSSPTreeAdjListFull_device, int* SSSPTreeAdjListTracker_device, std::vector<int>& hop, std::vector<bool>& visited);
	void printHop(vector<int>& v);


#ifdef __cplusplus
}
#endif



/*
 * Given a graph and a current queue computes next vertices (vertex frontiers) to traverse.
 */
__global__
void computeNextQueue(int* SSSPTreeAdjListFull_device, int* SSSPTreeAdjListTracker_device, int* hop,
	int queueSize, int* currentQueue, int* nextQueueSize, int* nextQueue, int level) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id
	if (tid < queueSize) {  // visit all vertexes in a queue in parallel
		int current = currentQueue[tid];
		for (int i = SSSPTreeAdjListTracker_device[current]; i < SSSPTreeAdjListTracker_device[current + 1]; i++){
			int v = SSSPTreeAdjListFull_device[i];
			if (hop[v] == INT_MAX) {
				hop[v] = level + 1;
				int position = atomicAdd(nextQueueSize, 1);
				nextQueue[position] = v;
			}
		}
	}
}


void bfsGPU(int start, int nodes, int* SSSPTreeAdjListFull_device, int * SSSPTreeAdjListTracker_device, int* d_hop, vector<bool>& visited) {


	const int n_blocks = (nodes + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

	// Initialization of GPU variables
	int* d_firstQueue;
	int* d_secondQueue;
	int* d_nextQueueSize;
	 // output


	// Initialization of CPU variables
	int currentQueueSize = 1;
	const int NEXT_QUEUE_SIZE = 0;
	int level = 0;

	// Allocation on device
	const int size = nodes * sizeof(int);
	cudaMalloc((void**)&d_firstQueue, size);
	cudaMalloc((void**)&d_secondQueue, size);
	cudaMalloc((void**)&d_nextQueueSize, sizeof(int));


	// Copy inputs to device
	cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);

	/*auto startTime = chrono::steady_clock::now();*/
	

	while (currentQueueSize > 0) {
		int* d_currentQueue;
		int* d_nextQueue;
		if (level % 2 == 0) {
			d_currentQueue = d_firstQueue;
			d_nextQueue = d_secondQueue;
		}
		else {
			d_currentQueue = d_secondQueue;
			d_nextQueue = d_firstQueue;
		}
		computeNextQueue << <n_blocks, N_THREADS_PER_BLOCK >> > (SSSPTreeAdjListFull_device, SSSPTreeAdjListTracker_device, d_hop,
			currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
		cudaDeviceSynchronize();
		++level;
		cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	}

	
	/*auto endTime = chrono::steady_clock::now();
	long duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	printf("Elapsed time for naive linear GPU implementation (without copying graph) : %li ms.\n", duration);*/

	cudaFree(d_firstQueue);
	cudaFree(d_secondQueue);
}

void printHop(vector<int>& v) {
	cout << "{ ";
	for (int i = 0; i < v.size(); ++i) {
		cout << v[i];
		if (i < v.size() - 1)
			cout << ", ";
	}
	cout << " }" << endl;
}