#ifndef ALL_STRUCTURE_DIR_CUH
#define ALL_STRUCTURE_DIR_CUH
#include <stdio.h>
#include <iostream>
//#include<list>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <chrono>

using namespace std;
using namespace std::chrono;


#include <omp.h>

typedef struct /* the edge data structure */
{
	long head;
	long tail;
	double weight;
} edge;


typedef struct /* the graph data structure */
{
	long numVertices;        /* Number of columns                                */
	long sVertices;          /* Number of rows: Bipartite graph: number of S vertices; T = N - S */
	long numEdges;           /* Each edge stored twice, but counted once        */
	long* edgeListPtrs;     /* start vertex of edge, sorted, primary key        */
	edge* edgeList;         /* end   vertex of edge, sorted, secondary key      */
} graph;






/******* Network Structures *********/
struct ColWt {
	int col;
	int wt;
};

//Structure for Edge
struct Edge
{
	int node1;
	int node2;
	double edge_wt;
};



struct changeEdge {
	int node1;
	int node2;
	double edge_wt;
	int inst;
};

typedef vector<ColWt> ColWtList;
typedef vector<int> ColList;






// Data Structure for each vertex in the rooted tree
struct RT_Vertex
{
	int Parent; //mark the parent in the tree
	int EDGwt; //mark weight of the edge
	int Dist;  //Distance from root
	int Update;  //Whether the distance of this edge was updated / affected



};


////functions////
//Node starts from 0




/*
 readin_changes function reads the change edges
 Format of change edges file: node1 node2 edge_weight insert_status
 insert_status = 1 for insertion. insert_status = 0 for deletion.
 */
void readin_changes(char* myfile, vector<changeEdge>& allChange_Ins, vector<changeEdge>& allChange_Del, vector<ColWtList>& AdjList, vector<ColWtList>& InEdgesList, int& totalInsertion)
{
	cout << "Reading input changed edges data..." << endl;
	auto readCEstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		if (line[0] == '%') continue;
		if (line[0] == '#') continue;
		int n1, n2, wt, inst_status;
		changeEdge cE;
		sscanf(line, "%d %d %d %d", &n1, &n2, &wt, &inst_status);
		cE.node1 = n1;
		cE.node2 = n2;
		cE.edge_wt = wt;
		cE.inst = inst_status;


		//add change edges with inst status = 1 to Adjlist
		if (inst_status == 1)
		{
			totalInsertion++;
			ColWt colwt;
			colwt.col = n1;
			colwt.wt = wt;
			InEdgesList.at(n2).push_back(colwt);
			ColWt colwt2;
			colwt2.col = n2;
			colwt2.wt = wt;
			AdjList.at(n1).push_back(colwt2);
			allChange_Ins.push_back(cE);
		}
		else if (inst_status == 0) {
			allChange_Del.push_back(cE);
		}

	}
	fclose(graph_file);
	auto readCEstopTime = high_resolution_clock::now();//Time calculation ends
	auto readCEduration = duration_cast<microseconds>(readCEstopTime - readCEstartTime);// duration calculation
	cout << "Reading input changed edges data completed. totalInsertion:" << totalInsertion << endl;
	cout << "Time taken to read input changed edges: " << readCEduration.count() << " microseconds" << endl;
	return;
}


/*
read_SSSP reads input SSSP file.
accepted SSSP data format: node parent distance
*/
void read_SSSP(RT_Vertex* SSSP, char* myfile, int* nodes, vector<ColList>& SSSPTreeAdjList)
{
	FILE* graph_file;
	char line[128];

	int prev_node = 0;
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		if (line[0] == '%') continue;
		if (line[0] == '#') continue;
		int node, parent, dist;
		sscanf(line, "%d %d %d", &node, &parent, &dist);
		if (node > prev_node + 1) {
			for (int i = prev_node + 1; i < node; i++) //if some node id were missing in the graph we consider it is disconnected
			{
				SSSP[i].Dist = 9999999;
				SSSP[i].Parent = -1;
				SSSP[i].Update = 0;
				//SSSP[i].EDGwt = 9999999;
			}
		}
		if (parent == -1)
		{
			dist = 9999999;
		}
		SSSP[node].Parent = parent;
		SSSP[node].Dist = dist;
		prev_node = node;
		SSSP[node].Update = 0;

		SSSPTreeAdjList.at(parent).push_back(node); //directed representation. parent to child connection
	}
	for (int j = prev_node + 1; j < *nodes; j++)
	{
		SSSP[j].Dist = 9999999;
		SSSP[j].Parent = -1;
		SSSP[j].Update = 0;
	}
	fclose(graph_file);

	return;
}

/*
read_graphEdges reads the original graph file
accepted data format: node1 node2 edge_weight
we consider only undirected graph here. for edge e(a,b) with weight W represented as : a b W
*/
void read_graphEdges(vector<ColWtList>& InEdgesList, vector<ColWtList>& AdjList, char* myfile, int* nodes)
{
	cout << "Reading input graph..." << endl;
	auto readGraphstartTime = high_resolution_clock::now();//Time calculation starts
	FILE* graph_file;
	char line[128];
	graph_file = fopen(myfile, "r");
	while (fgets(line, 128, graph_file) != NULL)
	{
		if (line[0] == '%') continue;
		if (line[0] == '#') continue;
		int n1, n2, wt;
		sscanf(line, "%d %d %d", &n1, &n2, &wt);
		ColWt colwt;
		colwt.col = n1;
		colwt.wt = wt;
		InEdgesList.at(n2).push_back(colwt);
		ColWt colwt2;
		colwt2.col = n2;
		colwt2.wt = wt;
		AdjList.at(n1).push_back(colwt2);
	}
	fclose(graph_file);
	auto readGraphstopTime = high_resolution_clock::now();//Time calculation ends
	auto readGraphduration = duration_cast<microseconds>(readGraphstopTime - readGraphstartTime);// duration calculation
	cout << "Reading input graph completed" << endl;
	cout << "Time taken to read input graph: " << readGraphduration.count() << " microseconds" << endl;
	return;
}



#endif