## Introduction

Ⅰ. Graph-based ranking algorithm (similar to HITS algorithm and Google's PageRank).

## TextRank Model

G = (V, E): directed graph with set of verties V and set of edges E, where E is subset of V .

The score of a vertex Vi is defined as follows:
$$
S(Vi) = (1-d) + d*\sum_{j\in(In(Vi))} \frac{1}{|Out(Vj)|}S(Vj)
$$
+ **In(Vi) : set of vertices that point to Vi, making the role of integrating into model to probability of jumping from a given vertex to another random vertex in the graph (usually set to 0.85) .**

+ **Out(Vi) : set of vertices that Vi points to**.

+ **d : a damping factor that can be set between 0 and 1.**

> Notice that the final values obtained after TextRank runs to completion are not affected by the choice of the initial value, only the number of iterations to convergence may be different.

## Undirected Graphs

For loosely connected graphs, with the number of vertices,  undirected graphs tend to have more gradual convergence curves.

## Weighted Graphs

​	The original PageRank definition for graphs-based ranking is assuming unweighted graphs.

​	It is useful to indicate and incorporate into the model the "strength" of the connection between two vertices Vi and Vj as a weight Wij added to the corresponding edge that connects the two vertices.

​	Consequently, a new formula for graph-based ranking that takes into account edge weights when computing the score associated with a vertex in the graph. Notice that a similar formula can be defined to integrate vertex weights.
$$
WS(Vi) = (1-d) + d*\sum_{V_j\in(In(Vi))}\frac{w_{ji}}{\sum_{V_k\in Out(V_j)}{w_{jk}}WS(V_j)}
$$


​	<img src="C:\Users\ljj12\AppData\Roaming\Typora\typora-user-images\image-20211219091828389.png" alt="image-20211219091828389" style="zoom:33%;" />

​	Figure 1 plots that while the final vertex scores (and therefore rankings) differ significantly as compared to their unweighted alternatives, ***the number of convergence and the shape of the convergence curves is almost identical for wighted and unweighted graphs***

## Text as a Graph

​	**Goal** : Building a graph that represents the text, and interconnects words or other text entities with meaningful relations.

​	Following is the main steps of graph-based ranking algorithms to natural language texts:

1. Identify ***<u>text units</u>*** that best define the task at hand, and add them as vertices in graph.
2. Identify ***<u>relations</u>*** that connect such text uinits, and use these relations to draw edges between vertices in the graph, Edges can be directed or undirected. weighted or unweighted.
3. ***<u>Iterate</u>*** the graph-based ranking algorithm until convergence
4. ***<u>Sort vertices</u>*** based on their final score. Use the values attached to each vertex for ranking/selection decisions

**NLP Tasks** involving ranking of text units :

- [x] **A keyword extraction task,** consisting of the selection of keyphrases representative for a give text.
- [x] **A sentence extraction task**, consisting of identification of the most "important" sentences in a text, which can be used to build extractive summaries.



## Keyword Extraction

​	To automatically identify in a text a set if term that best  describe the document.

​	The simplest possible approach is perhaps to use **frequency criterion** to select "import" keywords. (usually poor performance)



## TextRank for Key Extraction

​	<u>Any relation that can be defined between two lexical units is a potentially useful connection (edge) that can be added between two such vertices.</u>

​	Co-occurrence relation is used, controlled by the distance between word occurrences: two vertices are connected if their corresponding lexical units co-occur within a window of maximum N words, where N can be set anywhere from 2 to 10 words.

​	Co-occurrence links <u>express the relations between syntactic elements</u>, and similar to the semantic links found useful for the task of word sense disambiguation.

​	TextRank keyword extraction algorithm is fully unsupervised, and proceeds as follows.

1.  The <u>text is tokenized, and annotated</u> with part of speed tags.

2.  All <u>lexical units</u> that **pass syntactic filter** <u>are added to graph</u>, and an edge is added between those lexical units that co-occur within a window of N words. 

   The score associated with each vertex is set to an initial value of 1, and ranking algorithm is run on graph for several iterations until it converges.

3.  Once a final score is obtained for each vertex in the graph, <u>vertices are sorted in reversed order of their score</u>, and the top T vertices in the ranking are retained for post-processing.



​	***<u>Sequences of adjacent keywords are collapsed into a multi-word keyword</u>***



Four features that are determined for each "candidate" keyword:

+ **within-document frequency**
+ **collection frequency**
+ **relative position of the first occurrence**
+ **sequence of par of speech tags**



## Sentence Extraction

​	The problem of sentence extraction can be regarded as similar to keyword extraction.

​	The goal is to rank entire sentences, and therefore a vertex is added to the graph for each sentence in the text.

​	Define a new different relation : if there is a "similarity" relation between two sentences, where  <u>***"similarity" is measured as a function of their content overlap***</u>. Such a relation between two sentences can be seen as a process of "<u>recommendation</u>" : <u>a sentence that addresses certain concepts in a text, gives the reader a "recommendation" to refer other sentences in the text that address the same concepts, and therefore a link can be drawn between any two such sentences that share common content</u>.

​	Formally, given two sentences Si and Sj, with a sentence being represented by the set of Ni words that appear in the sentence：
$$
S_i = w_1^i , w_2^i ,...,w_{N_i}^i
$$
​	The similarity of Si and Sj is defined as :


$$
Similarity(S_i, S_j) = \frac{|\{w_k|w_k\in S_i \& w_k\in S_j\}|}{log(|S_i|)+log(|S_j|)}
$$

​	Other sentence similarity measures : 

- [x] String kernels
- [x] Cosine similarity
- [x] Longest common subsequence



​	After ranking algorithm is run on the graph, sentences are sorted in reversed order of their score, and top ranked sentences are sentences are selected for inclusion in the summary



## Evaluation

​	Compared with other system, 



## Discussion

​	TextRank ***succeeds in identifying the most important sentences*** in a text based on information exclusively drawn from text itself. TextRank ***relies only on the given text to derive an extractive summary***, which represents a summarization model closer to what humans are doing when producing an abstract for a given document.

TextRank goes beyond the sentence "connectivity" in a text.



## Why TextRank works

​	Following as reasons:

+ **Do not only rely on the local context of a text unit, but rather it takes into account information recursively drawn from the entire text (graph).**
+ **Connect various entities, and implements the concept of recommendation.**
+ **Through its iterative mechanism, TextRank goes beyond simple graph connectivity, and it is able to score text units based also on the "importance" of other text units they link to**. 



​	TextRank for a given application are the one most recommended by related text units in the text, <u>with preference given to the recommendations made by most influential one</u>, i.e. the ones that are in turn highly recommended by other related units. 

​	The underlying hypothesis is that ***<u>in a cohesive text frag-ment, related text units tend to form a "Web" of connections that approximates the model humans build about a given context in the process of discourse understanding</u>.***

### Conclusion



