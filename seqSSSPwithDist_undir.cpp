#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <queue>
using namespace std;

typedef pair<int, double> int_double;
struct ADJ_Bundle
{
	int Row;
	vector <int_double> ListW;

	//Constructor
	ADJ_Bundle() { ListW.resize(0); }

	//Destructor
	void clear()
	{
		while (!ListW.empty()) { ListW.pop_back(); }
	}


};
typedef  vector<ADJ_Bundle> A_Network;

void readin_graphU2(A_Network* X, int nodes, char* myfile)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	int l = 0;
	int prev_node = 0;
	int_double dummy;
	dummy.first = 1;
	dummy.second = 0;
	int_double dummy2;
	dummy2.first = 1;
	dummy2.second = 0;
	vector <int_double> ListW;

	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		int wt;
		sscanf(line, "%d %d %d", &n1, &n2, &wt);	
		dummy.first = n2;
		dummy.second = (double)wt;
		X->at(n1).ListW.push_back(dummy);
        X->at(n1).Row = n2;
		
		dummy2.first = n1;
		dummy2.second = (double)wt;
		X->at(n2).ListW.push_back(dummy2);
		X->at(n2).Row = n1;
		
	}
	fclose(graph_file);
	return;
}

/*Sequential Dijkstra's Algorithm for SSSP with a given source */
void djk(int src, A_Network* X, A_Network* Y, int node1) {


	int nodes = node1;
	//Set Edge Weights to high values
	double maxEdgeWeight = 1000.00;

	//Initialize the distance for the nodes to INF
	vector<double> dist(nodes, maxEdgeWeight * nodes);
	//Initialize distance of src
	dist[src] = 0;

	//Initialize parents of nodes to -1
	vector<int> parent(nodes, -1);
	//Initialize parent of source to itself
	parent[src] = src;

	//Initialize weight of the edge connecting the node to the parent to -1
	vector<double> EdgeW(nodes, -1);


	//Create priotiy queue for traversing the graph
	priority_queue<int_double, vector<int_double>, greater<int_double> > prtQ;
	//Add source to priority Q
	prtQ.push(make_pair(0, src));

	//Mark whether node is in priorityQ--initialized to false
	vector<bool> inQ(nodes, false);
	//Set src to true
	inQ[src] = true;

	while (!prtQ.empty())
	{
		int u = prtQ.top().second;
		prtQ.pop();
		inQ[u] = false;


		//For all neighbors of thisn
		for (int i = 0; i < X->at(u).ListW.size(); i++)
		{
			int v = X->at(u).ListW[i].first;
			double weight = X->at(u).ListW[i].second;


			//If relaxed update weight and push to priorityQ
			if (dist[v] > dist[u] + weight) {
				dist[v] = dist[u] + weight;
				parent[v] = u;
				EdgeW[v] = weight;

				if (!inQ[v])
				{
					prtQ.push(make_pair(dist[v], v));
					inQ[v] = true;
				}

			}

		}


	}
	
	//===Adding the Edges to Y

	   //Create rows for Y
	ADJ_Bundle AList;
	for (int i = 0; i < nodes; i++)
	{
		//Create Rows for Y;
		AList.Row = i;
		AList.ListW.clear();
		Y->at(i).Row = i;
	}

	//Add edges according to Parent Relation
	int_double myval;
	int j;
	for (int i = 0; i < nodes; i++)
	{

		//If source continue
		if (i == src) { continue; }

		//If edge in different component, then continue
		if (parent[i] == -1) { continue; }

		j = parent[i];

		myval.first = j;
		myval.second = EdgeW[i];
		Y->at(i).ListW.push_back(myval);

		myval.first = i;
		myval.second = EdgeW[i];
		Y->at(j).ListW.push_back(myval);



	}



	for (int i = 0; i < nodes; i++)
	{
		cout << i << " " <<  parent[i] << " " << dist[i] << " " << endl;
	}


	return;
}

/*
1st argument is filename
2nd argument is number of nodes
*/
int main(int argc, char* argv[])
{
	/*string file1 = "./fullGraph.txt";*/
	string file1 = argv[1];
	char* cstr1 = &file1[0];
	A_Network X, Y;

	int nodes = 0;
	nodes = atoi(argv[2]); //we pass the number of total nodes as argument
	ADJ_Bundle adjobj;
	X.resize(nodes, adjobj);
	Y.resize(nodes, adjobj);
	readin_graphU2(&X, nodes, cstr1);
	djk(0, &X, &Y, nodes);
	return 0;
}

