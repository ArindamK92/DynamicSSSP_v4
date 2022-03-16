#ifndef GPUFUNCTIONS_DIR_CUH
#define GPUFUNCTIONS_DIR_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector>
#include <fstream>
#include <sstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "all_structure_dir.cuh"
using namespace std;

#define THREADS_PER_BLOCK 1024 //we can change it


__global__ void deleteEdgeFromAdj(changeEdge* allChange_Del_device, int totalChangeEdges_Del, ColWt* InEdgesListFull_device, int* InEdgesListTracker_device, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
		int edge_weight = allChange_Del_device[index].edge_wt;

		//mark the edge as deleted in Adjlist
		for (int j = InEdgesListTracker_device[node_2]; j < InEdgesListTracker_device[node_2 + 1]; j++) {
			if (InEdgesListFull_device[j].col == node_1 && InEdgesListFull_device[j].wt == edge_weight) {
				InEdgesListFull_device[j].wt = -1;
				//printf("inside del inedge: %d %d %d \n", node_1, node_2, edge_weight);
			}

		}
		for (int j = AdjListTracker_device[node_1]; j < AdjListTracker_device[node_1 + 1]; j++) {
			if (AdjListFull_device[j].col == node_2 && AdjListFull_device[j].wt == edge_weight) {
				AdjListFull_device[j].wt = -1;
				//printf("inside del outedge: %d %d %d \n", node_1, node_2, edge_weight);
			}
		}
	}
}

__global__ void deleteEdge(changeEdge* allChange_Del_device, RT_Vertex* SSSP, int totalChangeEdges_Del, ColWt* AdjListFull_device, int* AdjListTracker_device) {
	//int index = threadIdx.x + blockIdx.x * blockDim.x;
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Del; index += blockDim.x * gridDim.x)
	{
		////Deletion case
		int node_1 = allChange_Del_device[index].node1;
		int node_2 = allChange_Del_device[index].node2;
		int edge_weight = allChange_Del_device[index].edge_wt;

		//this will check if node1 is parent of node2
		//Mark edge as deleted by making edgewt = inf
		if (SSSP[node_2].Parent == node_1) {
			SSSP[node_2].Dist = 999999;
			//SSSP[node_2].EDGwt = inf;
			SSSP[node_2].Update = 1;
		}
		//else if (SSSP[node_1].Parent == node_2) {
		//	SSSP[node_1].Dist = 999999;
		//	//SSSP[node_1].EDGwt = inf;
		//	SSSP[node_1].Update = 1;
		//}
	}
}

__global__ void insertEdge(changeEdge* allChange_Ins_device, RT_Vertex* SSSP, int totalChangeEdges_Ins, ColWt* AdjListFull_device, int* AdjListTracker_device) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < totalChangeEdges_Ins; index += blockDim.x * gridDim.x)
	{
		//Insertion case
		int node_1 = allChange_Ins_device[index].node1;
		int node_2 = allChange_Ins_device[index].node2;
		int edge_weight = allChange_Ins_device[index].edge_wt;


		int flag = 1;
		if (SSSP[node_1].Parent == node_2) { flag = 0; } //avoiding 1st type loop creation
		if (SSSP[node_2].Dist == 999999 && SSSP[node_2].Update == 1) { flag = 0; } //avoiding 2nd type loop creation

		//Check whether node1 is relaxed
		if ((SSSP[node_2].Dist > SSSP[node_1].Dist + edge_weight) && flag == 1) {
			//Update Parent and EdgeWt
			SSSP[node_2].Parent = node_1;
			SSSP[node_2].EDGwt = edge_weight;
			SSSP[node_2].Dist = SSSP[node_1].Dist + edge_weight;
			SSSP[node_2].Update = 2;
			//Mark Edge to be added
			//Edgedone[index] = 1;
		}
	}
}


/*
updateNeighbors_del function makes dist value of child nodes of a disconnected node to inf
It marks the child nodes also as disconnected nodes
*/

__global__ void updateNeighbors_del(RT_Vertex* SSSP, int* updated_counter_del, int* updatedEdgesList_del, int* affectedNodeList_del, int* counter_del, int* SSSPTreeAdjListFull_device, int* SSSPTreeAdjListTracker_device, int* change) {

	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter_del; index += blockDim.x * gridDim.x)
	{
		int tmpValueforIndex = affectedNodeList_del[index];
		SSSP[tmpValueforIndex].Update = 2;
		for (int j = SSSPTreeAdjListTracker_device[tmpValueforIndex]; j < SSSPTreeAdjListTracker_device[tmpValueforIndex + 1]; j++) {
			int myn = SSSPTreeAdjListFull_device[j];
			//int mywt = AdjListFull_device[j].wt;
			//if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

			//if (SSSP[myn].Parent == changedEdgesList_del[index] && SSSP[myn].Dist != 999999) {
			SSSP[myn].Dist = 999999;
			SSSP[myn].Update = 1;  //uncomment this if not using asynchrony

			//Asynchrony code starts
			//SSSP[myn].Update = 2;
			//for (int k = AdjListTracker_device[myn]; k < AdjListTracker_device[myn + 1]; k++) {
			//	int myn2 = AdjListFull_device[k].col;
			//	int mywt2 = AdjListFull_device[k].wt;

			//	if (mywt2 < 0) { continue; } //if mywt = -1, that means edge was deleted
			//	if (SSSP[myn2].Parent == myn && SSSP[myn2].Dist != 999999) {
			//		SSSP[myn2].Dist = 999999;
			//		SSSP[myn2].Update = 1;
			//		//updatedEdgesList_del[atomicAdd(updated_counter_del, 1)] = myn2;
			//	}
			//}
			//Asynchrony code stops

			//updatedEdgesList_del[atomicAdd(updated_counter_del, 1)] = myn;
			//				changedEdgesList[atomicAdd(counter, 1)] = myn;
			* change = 1;
			//}

		}
	}
}


//1. This method tries to connect the disconnected nodes(disconnected by deletion) with other nodes using the original graph
//2. This method propagates the dist update till the leaf nodes


__global__ void updateNeighbors(RT_Vertex* SSSP, int* counter, int* affectedNodeList, ColWt* InEdgesListFull_device, int* InEdgesListTracker_device, ColWt* AdjListFull_device, int* AdjListTracker_device, int* change) {
	/*int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < *counter) {*/
	for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < *counter; index += blockDim.x * gridDim.x)
	{
		//If i is updated--update its neighbors as required

		int tmpValueforIndex = affectedNodeList[index];
		SSSP[tmpValueforIndex].Update = 0;

		
		//For incoming edges of the affected nodes
		for (int j = InEdgesListTracker_device[tmpValueforIndex]; j < InEdgesListTracker_device[tmpValueforIndex + 1]; j++) {
			int myn = InEdgesListFull_device[j].col;
			int mywt = InEdgesListFull_device[j].wt;

			if (mywt < 0) { continue; } //if mywt = -1, that means edge was deleted

			//update where parent of myn != index
			if (SSSP[tmpValueforIndex].Dist > SSSP[myn].Dist + mywt) {
				if (SSSP[myn].Parent != tmpValueforIndex) {  //avoiding type 1 loop formation
					SSSP[tmpValueforIndex].Dist = SSSP[myn].Dist + mywt;
					SSSP[tmpValueforIndex].Update = 2;
					SSSP[tmpValueforIndex].Parent = myn;
					*change = 1;
					continue;
				}
			}
		}
		//For neighbor vertices of the affected nodes
		for (int j = AdjListTracker_device[tmpValueforIndex]; j < AdjListTracker_device[tmpValueforIndex + 1]; j++) {
			int myn = AdjListFull_device[j].col;
			int mywt = AdjListFull_device[j].wt;
			if (SSSP[myn].Dist > SSSP[tmpValueforIndex].Dist + mywt) {
				if (SSSP[tmpValueforIndex].Parent != myn) {//avoiding type 1 loop formation
					SSSP[myn].Dist = SSSP[tmpValueforIndex].Dist + mywt;
					SSSP[myn].Update = 2;
					SSSP[myn].Parent = tmpValueforIndex;
					* change = 1;
				}

			}
		}

	}
}


__global__ void printSSSP(RT_Vertex* SSSP, int nodes) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < 2) {
		int x;
		if (nodes < 40) {
			x = nodes;
		}
		else {
			x = 40;
		}
		printf("from GPU:\n[");
		for (int i = 0; i < x; i++) {
			printf("%d:%d:%d ", i, SSSP[i].Dist, SSSP[i].Parent);
		}
		printf("]\n");
	}
}

//used for affected_del
struct predicate
{
	__host__ __device__
		bool operator()(int x)
	{
		return x == 1;
	}
};

//used for affected_all
struct predicate2
{
	__host__ __device__
		bool operator()(int x)
	{
		return x > 0;
	}
};


#endif