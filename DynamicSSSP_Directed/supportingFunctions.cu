#ifndef SUPPORTING_CU
#define SUPPORTING_CU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>
#include <chrono>

#include "all_structure_dir.cuh"
#include "gpuFunctions_dir.cuh"
//#include "bfs.cu"
using namespace std;
using namespace std::chrono;


void transfer_data_to_GPU(vector<ColWtList>& InEdgesList, int*& InEdgesListTracker, vector<ColWt>& InEdgesListFull, ColWt*& InEdgesListFull_device, int*& InEdgesListTracker_device, vector<ColWtList>& AdjList, int*& AdjListTracker, vector<ColWt>& AdjListFull, ColWt*& AdjListFull_device,
	int nodes, int edges, int totalInsertion, int*& AdjListTracker_device, bool zeroInsFlag,
	vector<changeEdge>& allChange_Ins, changeEdge*& allChange_Ins_device, int totalChangeEdges_Ins,
	int deviceId, int totalChangeEdges_Del, bool zeroDelFlag, changeEdge*& allChange_Del_device,
	int*& counter_del, int*& affectedNodeList_del, int*& updatedAffectedNodeList_del, int*& updated_counter_del, vector<changeEdge>& allChange_Del, size_t  numberOfBlocks)
{
	cudaError_t cudaStatus;

	//create 1D array from 2D to fit it in GPU
	cout << "creating 1D array from 2D to fit it in GPU" << endl;
	AdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < nodes; i++) {
		AdjListTracker[i + 1] = AdjListTracker[i] + AdjList.at(i).size();
		AdjListFull.insert(std::end(AdjListFull), std::begin(AdjList.at(i)), std::end(AdjList.at(i)));
	}
	InEdgesListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < nodes; i++) {
		InEdgesListTracker[i + 1] = InEdgesListTracker[i] + InEdgesList.at(i).size();
		InEdgesListFull.insert(std::end(InEdgesListFull), std::begin(InEdgesList.at(i)), std::end(InEdgesList.at(i)));
	}
	cout << "creating 1D array from 2D completed" << endl;


	//Transferring input graph and change edges data to GPU
	cout << "Transferring graph data from CPU to GPU" << endl;
	auto startTime_transfer = high_resolution_clock::now();
	printf("edges: %d totalInsertion:%d sizeof(ColWt):%d \n", edges, totalInsertion, sizeof(ColWt));
	cudaStatus = cudaMallocManaged(&AdjListFull_device, (edges + totalInsertion) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	//printf("testA1");
	std::copy(AdjListFull.begin(), AdjListFull.end(), AdjListFull_device);
	//printf("testA2");

	cudaStatus = cudaMalloc((void**)&AdjListTracker_device, (nodes + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(AdjListTracker_device, AdjListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
	//printf("testB");
	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(AdjListFull_device, edges * sizeof(ColWt), deviceId);
	//printf("testC");





	cudaStatus = cudaMallocManaged(&InEdgesListFull_device, (edges + totalInsertion) * sizeof(ColWt));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListFull structure");
	}
	//printf("testA1");
	std::copy(InEdgesListFull.begin(), InEdgesListFull.end(), InEdgesListFull_device);
	//printf("testA2");

	cudaStatus = cudaMalloc((void**)&InEdgesListTracker_device, (nodes + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at InEdgesListTracker_device");
	}
	cudaMemcpy(InEdgesListTracker_device, InEdgesListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);
	//printf("testB");
	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(InEdgesListFull_device, edges * sizeof(ColWt), deviceId);






	if (zeroInsFlag != true) {
		cudaStatus = cudaMallocManaged(&allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed at allChange_Ins structure");
		}
		std::copy(allChange_Ins.begin(), allChange_Ins.end(), allChange_Ins_device);
		//printf("testD");
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		cudaMemAdvise(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
		//printf("testE");
		//Asynchronous prefetching of data
		cudaMemPrefetchAsync(allChange_Ins_device, totalChangeEdges_Ins * sizeof(changeEdge), deviceId);
		//printf("testF");
	}

	if (zeroDelFlag != true) {
		cudaStatus = cudaMallocManaged(&allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed at allChange_Del structure");
		}
		std::copy(allChange_Del.begin(), allChange_Del.end(), allChange_Del_device);
		//set cudaMemAdviseSetReadMostly by the GPU for change edge data
		cudaMemAdvise(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), cudaMemAdviseSetReadMostly, deviceId);
		//Asynchronous prefetching of data
		cudaMemPrefetchAsync(allChange_Del_device, totalChangeEdges_Del * sizeof(changeEdge), deviceId);

		counter_del = 0;
		cudaMallocManaged(&counter_del, sizeof(int));
		cudaMallocManaged(&affectedNodeList_del, nodes * sizeof(int));
		cudaMallocManaged(&updatedAffectedNodeList_del, nodes * sizeof(int));
		updated_counter_del = 0;
		cudaMallocManaged(&updated_counter_del, sizeof(int));

		//modify adjacency list to adapt the deleted edges
		deleteEdgeFromAdj << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, totalChangeEdges_Del, InEdgesListFull_device, InEdgesListTracker_device, AdjListFull_device, AdjListTracker_device);
		cudaDeviceSynchronize();
	}



	auto stopTime_transfer = high_resolution_clock::now();//Time calculation ends
	auto duration_transfer = duration_cast<microseconds>(stopTime_transfer - startTime_transfer);// duration calculation
	cout << "**Time taken to transfer graph data from CPU to GPU: "
		<< float(duration_transfer.count()) / 1000 << " milliseconds**" << endl;
}

void read_and_transfer_input_SSSPtree_to_GPU(char* inputSSSPfile, vector<ColList>& SSSPTreeAdjList, int*& SSSPTreeAdjListTracker, vector<int>& SSSPTreeAdjListFull,
	RT_Vertex*& SSSP, int nodes, int edges, int*& SSSPTreeAdjListFull_device, int*& SSSPTreeAdjListTracker_device, /*vector<int>& hop,*/ int deviceId/*, int*& d_hop*/)
{
	cudaError_t cudaStatus;

	SSSPTreeAdjList.resize(nodes);
	SSSPTreeAdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row

	cudaStatus = cudaMallocManaged(&SSSP, nodes * sizeof(RT_Vertex));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSP structure");
	}
	cout << "Reading input SSSP tree data..." << endl;
	auto readSSSPstartTime = high_resolution_clock::now();//Time calculation starts
	read_SSSP(SSSP, inputSSSPfile, &nodes, SSSPTreeAdjList);


	//New addition
	SSSPTreeAdjListTracker[0] = 0; //start pointer points to the first index of InEdgesList
	for (int i = 0; i < nodes; i++) {
		SSSPTreeAdjListTracker[i + 1] = SSSPTreeAdjListTracker[i] + SSSPTreeAdjList.at(i).size();
		SSSPTreeAdjListFull.insert(std::end(SSSPTreeAdjListFull), std::begin(SSSPTreeAdjList.at(i)), std::end(SSSPTreeAdjList.at(i)));
	}


	//Transferring SSSP tree data to GPU

	cudaStatus = cudaMallocManaged(&SSSPTreeAdjListFull_device, (nodes) * sizeof(int)); //new change to nodes from nodes -1 as 0 0 0 is also a row in SSSP file//SSSP tree has n-1 edges and we consider each edge 1 time
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSPTreeAdjListFull_device structure");
	}
	std::copy(SSSPTreeAdjListFull.begin(), SSSPTreeAdjListFull.end(), SSSPTreeAdjListFull_device);


	cudaStatus = cudaMalloc((void**)&SSSPTreeAdjListTracker_device, (nodes + 1) * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at SSSPTreeAdjListTracker_device");
	}
	cudaMemcpy(SSSPTreeAdjListTracker_device, SSSPTreeAdjListTracker, (nodes + 1) * sizeof(int), cudaMemcpyHostToDevice);


	//compute hop
	//vector<bool> visited;
	//int startVertex = 0; //0 is considered as root vertex
	//visited = vector<bool>(nodes);
	////hop = vector<int>(nodes);

	//const int size = nodes * sizeof(int);
	//cudaMalloc((void**)&d_hop, size);
	//hop = vector<int>(nodes, INT_MAX);
	//hop[startVertex] = 0;
	//cudaMemcpy(d_hop, hop.data(), size, cudaMemcpyHostToDevice);

	//??we don't need this hop computing now
	/*auto startTime = chrono::steady_clock::now();
	bfsGPU(startVertex, nodes, SSSPTreeAdjListFull_device, SSSPTreeAdjListTracker_device, d_hop, visited);
	auto endTime = std::chrono::steady_clock::now();
	long duration = chrono::duration_cast<chrono::milliseconds>(endTime - startTime).count();
	printf("Elapsed time for hop computation : %li ms.\n", duration);*/

	//cudaDeviceSynchronize();
	//cudaMemcpy(&hop[0], d_hop, size, cudaMemcpyDeviceToHost);

	auto readSSSPstopTime = high_resolution_clock::now();//Time calculation ends
	auto readSSSPduration = duration_cast<microseconds>(readSSSPstopTime - readSSSPstartTime);// duration calculation
	cout << "Reading input SSSP tree data completed" << endl;
	cout << "Time taken to read input input SSSP tree: " << readSSSPduration.count() << " microseconds" << endl;
	//set cudaMemAdviseSetPreferredLocation at GPU for SSSP data
	cudaMemAdvise(SSSP, nodes * sizeof(RT_Vertex), cudaMemAdviseSetPreferredLocation, deviceId);

	//Asynchronous prefetching of data
	cudaMemPrefetchAsync(SSSP, nodes * sizeof(RT_Vertex), deviceId);

}
#endif