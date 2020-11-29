#ifndef BFS_CUH
#define BFS_CUH


//this file is not in use


//#include "graph.h"
void bfsGPU(int start, int nodes, int* SSSPTreeAdjListFull_device, int* SSSPTreeAdjListTracker_device, int* d_hop, std::vector<bool>& visited);
void printHop(vector<int>& v);


/*
 * start - vertex number from which traversing a graph starts
 * hop - placeholder for vector of distances (filled after invoking a function)
 * visited - placeholder for vector indicating that vertex was visited (filled after invoking a function)
 */

#endif // BFS_GPU_CUH
