#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include<vector> 
#include <fstream> 
#include <sstream>
#include <queue>
using namespace std;


using namespace std;

typedef pair<int, double> int_double;
struct edge
{
	int node1, node2, wt;
};
typedef  vector<edge> Edges;
void readin_graphedges(Edges* X, char* myfile)
{
	FILE* graph_file;
	char line[128];

	graph_file = fopen(myfile, "r");
	

	while (fgets(line, 128, graph_file) != NULL)
	{
		int n1, n2;
		int wt;
		edge e;
		//Read line
		sscanf(line, "%d %d %d", &n1, &n2, &wt);
		e.node1 = n1;
		e.node2 = n2;
		e.wt = wt;
		X->push_back(e);
		
	}
}

/*
arg1: <fullGraph.txt>
arg2: <no. of nodes in actual graph> 
arg3: <no of change edge>
arg4: <percentage of insertion>
*/

int main(int argc, char* argv[])
{


	/***** Preprocessing to Graph (GUI) ***********/

	//Assume Processed Input
	//Form node node weight
	//Nodes are numbered contiguously
	//Starts from zero
	//Edges are undirected (both a-->b and b-->a present)
	//Check if valid input is given
	if (argc < 3) { cout << "INPUT ERROR:: Four inputs required. First: filename. Second: no. of nodes in actual graph Third: number of changed edges. Fourth: Percentage of Inserted Edges (in values 0 to 100) \n"; return 0; }
	//Check to see if file opening succeeded
	ifstream the_file(argv[1]); if (!the_file.is_open()) { cout << "INPUT ERROR:: Could not open main file\n"; }
	/*** Create DataStructure Sparsifictaion Tree **/


		/******* Read Graph to EdgeList****************/

	string file1 = argv[1];
	char* cstr1 = &file1[0];
	Edges X;
	int nodes = 0;
	nodes = atoi(argv[2]);
	readin_graphedges(&X, cstr1);



	//  printf("done reading graph\n");
	  /******* Read Graph to EdgeList****************/

	  /**** Create Set of Edges to Modify ****/

	int numE = atoi(argv[3]);
	int ins_per = atoi(argv[4]);


	double numF = (double)numE * ((double)ins_per / (double)100);
	int numI = (int)numF;
	int numD = numE - numI;

	int iI = 0;//number of inserts
	int iD = 0;//number of deletes
	int k;

	while (1)
	{
		k = rand() % 2;

		//Edges to Insert
		if (k == 1 && iI < numI)
		{

			int nx = rand() % nodes;
			int ny = rand() % nodes;
			int wt = rand() % 100; //highest wt 99
			if (nx == ny) { continue; }

			int n1, n2;
			if (nx < ny) { n1 = nx; n2 = ny; }
			else
			{
				n1 = ny; n2 = nx;
			}

			printf("%d  %d %d 1 \n", n1, n2, wt);
			iI++;


		}//end of if


			// Edge to Delete
		if (k == 0 && iD < numD)
		{
			int nz = rand() % (X.size());
			edge mye = X.at(nz);
			printf("%d %d %d 0 \n", mye.node1, mye.node2, mye.wt);
			iD++;
			continue;
		} //end of if

		if (iI == numI && iD == numD) { break; }
	}//end of while

	return 0;
}//end of main

	//==========================================//



