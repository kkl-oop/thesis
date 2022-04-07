##### 1.1 Graph problems in NLP

Graph types

+ semantic graph 

  ​	It visualizes the underlining meaning of a given sentence by abstracting the sentence into several concepts and their relations.

+ dependency graphs

  ​	It simply captures word-to-word dependencies.

+ knowledge graphs

  ​	It represents real-word knowledge by entities

##### 1.2 Previous approaches for modeling graphs

Method : 

+ statistical

+ rule-based approached

  e.g synchronous grammar-based methods



CNN : by stacking the layers, more global correspondences can be captured, with drawback of amount of computation.

RNN : Based on depth-first traversal algorithms, linear in terms of graph scale, losing some structural information after linearization. (solve : insert brackets into linearization results)



##### 1.3 Motivation and overview of our model

​	Explore better alternatives for encoding graphs, which are general enough to be applied on arbitrary without destroying the original structures.



##### 2.1 Encoding graphs with RNN or DAG network

​	AMRS are rooted and directed graphs, where the graph nodes represent the concepts and edges represent the relations between nodes.

​	Linearization causes loss of the structural information. For instance, originally closed-located graph nodes can be far away, especially when graph is large. And there is no specific order among the children for a graph node, resulting multiple linearization possibilities.



##### 2.2 Encoding graphs with Graph Neural Network(GNN)

​	To update node states within a graph, GNNs rely on message passing mechanism that iteratively updates the node states in parallel



 

