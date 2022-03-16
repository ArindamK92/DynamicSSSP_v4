#include <stdio.h>
#include "all_structure_dir.cuh"
#include "gpuFunctions_dir.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include<vector>
#include <chrono>
#include <algorithm>
#include "cuCompactor.cuh"
#include "supportingFunctions.cu"


#define THREADS_PER_BLOCK 1024 //we can change it

using namespace std;
using namespace std::chrono;



/*
1st arg: original graph file name
2nd arg: no. of nodes
3rd arg: no. of edges
4th arg: input SSSP file name
5th arg: change edges file name
****main commands to run****
nvcc -o op_main CudaSSSPmain.cu
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name
*/
int main(int argc, char* argv[]) {

	int nodes, edges, deviceId, numberOfSMs;
	cudaError_t cudaStatus;
	nodes = atoi(argv[2]);
	edges = atoi(argv[3]);
	char* inputSSSPfile = argv[4];
	int totalInsertion = 0;
	bool zeroDelFlag = false, zeroInsFlag = false;
	vector<ColWtList> InEdgesList; //stores inedge neighbors
	InEdgesList.resize(nodes);
	int* InEdgesListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row 
	vector<ColWt> InEdgesListFull;
	ColWt* InEdgesListFull_device; 
	int* InEdgesListTracker_device; 
	vector<ColWtList> AdjList; //stores input graph in 2D adjacency list
	vector<ColWt> AdjListFull; //Row-major implementation of adjacency list (1D)
	ColWt* AdjListFull_device; //1D array in GPU to store Row-major implementation of adjacency list 
	int* AdjListTracker_device; //1D array to track offset for each node's adjacency list
	vector<changeEdge> allChange_Ins, allChange_Del;
	changeEdge* allChange_Ins_device; //stores all change edges marked for insertion in GPU
	changeEdge* allChange_Del_device; //stores all change edges marked for deletion in GPU
	int* counter_del;
	int* affectedNodeList_del;
	int* updatedAffectedNodeList_del;
	int* updated_counter_del;
	vector<ColList> SSSPTreeAdjList;
	int* SSSPTreeAdjListTracker;
	vector<int> SSSPTreeAdjListFull;
	RT_Vertex* SSSP;
	int* SSSPTreeAdjListFull_device;
	int* SSSPTreeAdjListTracker_device;
	/*vector<int> hop;
	int* d_hop;*/

	//Get gpu device id and number of SMs
	cudaGetDevice(&deviceId);
	cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
	size_t  numberOfBlocks = 32 * numberOfSMs;

	//Read Original input graph
	AdjList.resize(nodes);
	int* AdjListTracker = (int*)malloc((nodes + 1) * sizeof(int));//we take nodes +1 to store the start ptr of the first row
	read_graphEdges(InEdgesList, AdjList, argv[1], &nodes);


	//Read change edges input
	readin_changes(argv[5], allChange_Ins, allChange_Del, AdjList, InEdgesList, totalInsertion);
	int totalChangeEdges_Ins = allChange_Ins.size();
	if (totalChangeEdges_Ins == 0) {
		zeroInsFlag = true;
	}
	int totalChangeEdges_Del = allChange_Del.size();
	if (totalChangeEdges_Del == 0) {
		zeroDelFlag = true;
	}

	//Transfer input graph, changed edges to GPU and set memory advices
	transfer_data_to_GPU(InEdgesList, InEdgesListTracker, InEdgesListFull, InEdgesListFull_device, InEdgesListTracker_device, AdjList, AdjListTracker, AdjListFull, AdjListFull_device,
		nodes, edges, totalInsertion, AdjListTracker_device, zeroInsFlag,
		allChange_Ins, allChange_Ins_device, totalChangeEdges_Ins,
		deviceId, totalChangeEdges_Del, zeroDelFlag, allChange_Del_device,
		counter_del, affectedNodeList_del, updatedAffectedNodeList_del, updated_counter_del, allChange_Del, numberOfBlocks);


	//Read input SSSP Tree and storing on unified memory
	read_and_transfer_input_SSSPtree_to_GPU(inputSSSPfile, SSSPTreeAdjList, SSSPTreeAdjListTracker, SSSPTreeAdjListFull,
		SSSP, nodes, edges, SSSPTreeAdjListFull_device, SSSPTreeAdjListTracker_device, /*hop,*/ deviceId/*, d_hop*/);


	//Initialize supporting variables
	int* change = 0;
	cudaStatus = cudaMallocManaged(&change, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at change structure");
	}
	int* affectedNodeList;
	cudaStatus = cudaMallocManaged(&affectedNodeList, nodes * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at affectedNodeList structure");
	}
	int* counter = 0;
	cudaStatus = cudaMallocManaged(&counter, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed at counter structure");
	}
	int* updatedAffectedNodeList_all;
	cudaMallocManaged(&updatedAffectedNodeList_all, nodes * sizeof(int));
	int* updated_counter_all = 0;
	cudaMallocManaged(&updated_counter_all, sizeof(int));





	//**process change edges**
	auto startTimeDelEdge = high_resolution_clock::now(); //Time calculation start
	//Process del edges
	if (zeroDelFlag != true) {

		deleteEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Del_device, SSSP, totalChangeEdges_Del, AdjListFull_device, AdjListTracker_device);
		cudaDeviceSynchronize();
	}
	auto stopTimeDelEdge = high_resolution_clock::now();//Time calculation ends
	auto durationDelEdge = duration_cast<microseconds>(stopTimeDelEdge - startTimeDelEdge);// duration calculation
	cout << "**Time taken for processing deleted edges: "
		<< float(durationDelEdge.count()) / 1000 << " milliseconds**" << endl;





	//Process ins edges
	auto startTimeinsertEdge = high_resolution_clock::now();
	if (zeroInsFlag != true) {

		insertEdge << < numberOfBlocks, THREADS_PER_BLOCK >> > (allChange_Ins_device, SSSP, totalChangeEdges_Ins, AdjListFull_device, AdjListTracker_device);
		cudaDeviceSynchronize();
	}
	auto stopTimeinsertEdge = high_resolution_clock::now();//Time calculation ends
	auto durationinsertEdge = duration_cast<microseconds>(stopTimeinsertEdge - startTimeinsertEdge);// duration calculation
	cout << "**Time taken for processing inserted Edges: "
		<< float(durationinsertEdge.count()) / 1000 << " milliseconds**" << endl;



	//**make the subtree under deletion affected vertices disconnected (make wt = inf)
	auto startTimeupdateNeighbors_del = high_resolution_clock::now();
	if (zeroDelFlag != true) {
		*counter_del = cuCompactor::compact<RT_Vertex, int>(SSSP, affectedNodeList_del, nodes, predicate(), THREADS_PER_BLOCK);
		*change = 1;
		while (*change > 0) {
			*change = 0;
			updateNeighbors_del << <numberOfBlocks, THREADS_PER_BLOCK >> >
				(SSSP, updated_counter_del, updatedAffectedNodeList_del, affectedNodeList_del, counter_del, SSSPTreeAdjListFull_device, SSSPTreeAdjListTracker_device, change);
			*counter_del = cuCompactor::compact<RT_Vertex, int>(SSSP, affectedNodeList_del, nodes, predicate(), THREADS_PER_BLOCK);
			//printf("number of elements in the compacted list: %d\n", *counter_del);
			//cudaDeviceSynchronize();//not required as cudaMalloc/cudaFree perform heavy-weight synchronizations. cuCompactor::compact uses both in it.
		}
	}
	cudaFree(SSSPTreeAdjListFull_device); //we can free memory at the end if we have enough GPU memory. That will decrease some time
	cudaFree(SSSPTreeAdjListTracker);

	auto stopTimeupdateNeighbors_del = high_resolution_clock::now();//Time calculation ends
	auto durationupdateNeighbors_del = duration_cast<microseconds>(stopTimeupdateNeighbors_del - startTimeupdateNeighbors_del);// duration calculation
	cout << "**Time taken for updateNeighbors_del: "
		<< float(durationupdateNeighbors_del.count()) / 1000 << " milliseconds**" << endl;



	//**Update neighbors and connect disconnected vertices with main SSSP tree**
	auto startTimeupdateNeighbors = high_resolution_clock::now();

	//collect all vertices where update value > 0
	*counter = cuCompactor::compact<RT_Vertex, int>(SSSP, affectedNodeList, nodes, predicate2(), THREADS_PER_BLOCK);

	*change = 1;
	while (*change == 1) {
		*change = 0;
		updateNeighbors << <(*counter / THREADS_PER_BLOCK) + 1, THREADS_PER_BLOCK >> > (SSSP, counter, affectedNodeList, InEdgesListFull_device, InEdgesListTracker_device, AdjListFull_device, AdjListTracker_device, change);
		*counter = cuCompactor::compact<RT_Vertex, int>(SSSP, affectedNodeList, nodes, predicate2(), THREADS_PER_BLOCK);
		//cudaDeviceSynchronize(); //not required as cudaMalloc/cudaFree perform heavy-weight synchronizations. cuCompactor::compact uses both in it.
	}

	auto stopTimeupdateNeighbors = high_resolution_clock::now();//Time calculation ends
	auto durationupdateNeighbors = duration_cast<microseconds>(stopTimeupdateNeighbors - startTimeupdateNeighbors);// duration calculation
	cout << "**Time taken for updateNeighbors: "
		<< float(durationupdateNeighbors.count()) / 1000 << " milliseconds**" << endl;






	cout << "****Total Time taken for SSSP update: "
		<< (float(durationDelEdge.count()) + float(durationupdateNeighbors_del.count()) + float(durationinsertEdge.count()) + float(durationupdateNeighbors.count())) / 1000 << " milliseconds****" << endl;



	//cout << "Total affected nodes by Delete edge only: " << totalAffectedNodes_del << endl;

	cout << "from GPU: \n[";
	printSSSP << <1, 1 >> > (SSSP, nodes);
	cudaDeviceSynchronize();
	int x;
	if (nodes < 40) {
		x = nodes;
	}
	else {
		x = 40;
	}
	cout << "from CPU: \n[";
	for (int i = 0; i < x; i++) {
		cout << i << ":" << SSSP[i].Dist << " ";
	}
	cout << "]\n";
	//print output ends



	if (zeroDelFlag != true) {
		cudaFree(affectedNodeList_del);
		cudaFree(updatedAffectedNodeList_del);
		cudaFree(counter_del);
		cudaFree(updated_counter_del);
		cudaFree(allChange_Del_device);

	}
	if (zeroInsFlag != true) {
		cudaFree(allChange_Ins_device);
	}
	cudaFree(change);
	cudaFree(affectedNodeList);
	cudaFree(counter);
	cudaFree(AdjListFull_device);
	cudaFree(AdjListTracker_device);

	cudaFree(SSSP);
	/*cudaFree(d_hop);*/ //try to free this at some earlier place
	return 0;
}