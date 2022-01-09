"# DynamicSSSP_v4" 
Here all edges are considered as undirected edges.
 
**compile:**
_____________
```shell
nvcc -o op_main SSSPmainForUndirected.cu
```

**run:**
_____________
```shell
./op_main original_graph_file_name number_of_nodes number_of_edges input_SSSP_file_name change_edge_file_name
```

example:  
```shell
./op_main graph.txt 6 6 SSSP.txt cE.txt
```


**File format:**
_____________

Original graph (3 col):
node1 node2 edge_weight
graph should be undirected. If 'a b W' is part of the graph file, then 'b a W' should not be included in the graph file.


SSSP (3 col):
node parent distance

change edges file (4 col):
node1 node2 edge_weight inst_status
edges are considered as undirected edges.

inst_status can be 0(for deletion) or 1(for insertion)

**How to prepare input SSSP data from original graph:**
___________________________________________________
Use below command to generate input SSSP tree:
```shell
nvcc -o op_ssspSeq seqSSSPwithDist_undir.cpp
./op_ssspSeq original_graph_file_name number_of_nodes
```
example:  
```shell
./op_ssspSeq graph.txt 6
```

**How to prepare change edges:**
____________________________
```shell
nvcc -o op_cE createChangedEdges.cpp  
./op_cE original_graph_file_name number_of_nodes number_of_change_edges percent_of_insertion  
```
example:  
```shell
./op_cE graph.txt 6 10 50
<<<<<<< HEAD
```
=======
```
>>>>>>> f40b468839ac9e1aab791410b123aeff4eaf2b12
