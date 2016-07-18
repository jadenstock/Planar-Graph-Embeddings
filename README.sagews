###Definitions###
# a graph is said to be quadrangualted if all of the faces of the graph (exept for an specified outter face) have length 4.
# a geodesic on a graph is defined to be a path though the graph. This can be represented by a list of edges in the graph which the geodesic goes through. no more than one geodesic can go through one edge.
# two geodesics in a quadrangulated graph are said to intersect if they both go through opposite edges of the same face. If they go thorugh adjacent edges of the face they are not said to intersect.
# a face of a quadrangualed graph is said to be active if the two geodesics that go through that face intersect more than once.
# an orrientation of a face is a way of decomposing the four edges in the face into two grous of two edges. There are three possible orrintations on a face which are called "GF", "type1", and "type2" in the code
# a connection on a graph is an assignment of an orrientation to every face in the graph. In the code it is represented by a dictionary with keys as frozensets of faces in the graph and values as a string for the orrientation.

### Creating and modifying a graph a graph ###

# there are several ways to construct a graph in sage. One can provide a dictionary like so, cube = Graph({1:[2,3,5], 2:[1,4,6], 3:[1,4,7], 4:[3,2,8], 5:[6,7,1], 6:[5,8,2], 7:[5,8,3], 8:[7,6,4]}). Or one can create the graph from a matrix, k_23 = Graph(matrix([[0,1,1,1,0], [1,0,0,0,1], [1,0,0,0,1], [1,0,0,0,1], [0,1,1,1,0]])). Sage also has some built in graphs but if you use these you will not be able to modify them without raising an error. So instead I have provided some functions for constructing graphs such as grid(n,m), cycle(n), and k_2n(n). I have also provided some function for modifying a given graph. These functions are
# - quadrangualte(graph): this function adds nodes and edges to the graph to make all faces have length 4 while preserving the shorted distnace between the original nodes.
# - make_bipartite(graph): this function adds a node to each edge of the graph making it a bipartite graph
# - add_random_k2m(grpah, n, m=3): this function takes a quadrangulate graph and places a node in the center of n faces of the graph and adds edges to two of the four vertices in those faces. Essentially adding copies of k_23 to the graph. Copies of k_2m can be added instead with the optional argument m.

#example of constructing graphs:
# g = grid(5,5)
# add_random_k2m(g, 4)
# h = cycle(14)
# quadrangulate(h)
# d = Graph(diamond)
# quadrangulate(d)


### getting and modifying a connection on a graph and getting the geodesics on a graph ###

# we usually start with the flat connection on a grpah which can be getten with the function flat_connection(graph). We can then modify this connection with a number of different functions. For example we can replace the connection by the anti scaher decomposition with connection = anti_schafer_decomposition(g, connection). We can also untangle a random subset of active faces subject to the constraint that an odd number of faces remain unchanged for each pair of intersecting geodesics. The function to do this for all geodesics in the graph is uncross_random_subset_of_faces_in_graph(graph, connection).

#connection = flat_connection(g)
#connection = anti_schafer_decomposition(g, connection)
#connection = uncross_random_subset_of_faces_in_graph(g, connection)
#connection = anti_schafer_decomposition(g, connection)

#A connection on a graph uniquely determines the set of geodesics on the graph. To get a list of those geodesics we use all_geodesics(graph, connection).



### getting a metric on a graph and computing the distortion of the metric ###

# To get a metric on a graph we need the cuts and to get the cuts we need the geodesics. For a doubly indexed dictionary of distance between nodes in the cut metric we can call,
# geodesics = all_geodesics(graph, connection)
# cuts = get_cuts(graph, geodesics)
# metric = cut_metric(graph, cuts)

# to get the distortion of a metric from the shortest path metric on the graph we can call distortion_of_metric(graph, metric)


### the main embedding ###
# there are three function for the main embedding. They are embedding_connections(graph, k), embedding_cuts(graph, embeddings), embedding_distortion(graph, family_of_cuts). The first returns a list with k different connections on the graph which are a result of calling the function uncross_random_subset_of_faces_in_graph. The second function returns a family of cuts which are gotten from the connections returned by the first function. The third function takes the graph and the cuts from the second function and computes the distortion between the matric computted by the embedding and the shortest path metric on the graph


### drawing a graph ###
# The main function for drawing a graph with it's geodesics is '
# draw_graph(graph, connection, pos = None, node_size = 300, epicenters = False, midpoints = True, geodesics = True):
# if pos = None then tuttes embedding will be used. There are optional arguments for the node size, and boolean values for if you want to draw the geodesics, midpoints of edges, or epicenters of faces.
