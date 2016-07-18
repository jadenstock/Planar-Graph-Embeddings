import copy
import networkx as nx
import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import random
from random import choice
from random import shuffle

### several graphs of interest ###
cube = {1:[2,3,5], 2:[1,4,6], 3:[1,4,7], 4:[3,2,8], 5:[6,7,1], 6:[5,8,2], 7:[5,8,3], 8:[7,6,4]}
grid5x5 = {0:[1,5], 1:[0,2,6], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10], 6:[1,5,7,11], 7:[6,8,2,12], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16], 12:[11,13,7, 17], 13:[14,18,8,12], 14:[13,19,9], 15:[10,16,20], 16:[15,17,11,21], 17:[16,12,18,22], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16], 22:[21,23,17], 23:[22,24, 18], 24:[23,19]}
grid5x5_with_one_k23 = {0:[1,5], 1:[0,2,6], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10], 6:[1,5,7,11], 7:[6,8,2,12], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16], 12:[25,11,13,7, 17], 13:[14,18,8,12], 14:[13,19,9], 15:[10,16,20], 16:[25,15,17,11,21], 17:[16,12,18,22], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16], 22:[21,23,17], 23:[22,24, 18], 24:[23,19], 25:[16,12]}
grid5x5_with_three_k23 = {0:[1,5], 1:[0,2,6], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10], 6:[1,5,7,11], 7:[6,8,2,12], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16], 12:[11,13,], 13:[14,18], 14:[13,19,9], 15:[16,20], 16:[15,17,11,21], 17:[16,12,18,22], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16], 22:[21,23,17], 23:[22,24, 18], 24:[23,19], 25:[12,16], 26:[2,8], 27:[13,19]}
grid_with_one_self_intersection = {0:[1,5], 1:[0,2,6], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10, 26], 6:[1,5,7,11], 7:[6,8,2,12, 27], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16, 25, 26, 27], 12:[11,13,7, 17], 13:[14,18,8,12], 14:[13,19,9], 15:[10,16,20, 25], 16:[15,17,11,21], 17:[16,12,18,22], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16], 22:[21,23,17], 23:[22,24, 18], 24:[23,19]}
grid_with_two_self_intersection = {0:[1,5], 1:[0,2,6], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10, 26], 6:[1,5,7,11], 7:[6,8,2,12, 27], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16, 25, 26, 27], 12:[11,13,7, 17], 13:[14,18,8,12], 14:[13,19,9], 15:[10,16,20, 25], 16:[15,17,11,21], 17:[16,12,18,22, 28, 29], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16, 28], 22:[21,23,17], 23:[22,24, 18, 29], 24:[23,19], 28:[21,17], 29:[17,23]}
grid_with_three_self_intersection = {0:[1,5], 1:[0,2,6, 30], 2:[1,3,7], 3:[2,4,8], 4:[3,9], 5:[0,6,10, 26], 6:[1,5,7,11], 7:[6,8,2,12, 27, 30, 31], 8:[7,9,3,13], 9:[4,8,14], 10:[5,11,15], 11:[10,12,6,16, 25, 26, 27], 12:[11,13,7, 17], 13:[14,18,8,12, 31], 14:[13,19,9], 15:[10,16,20, 25], 16:[15,17,11,21], 17:[16,12,18,22, 28, 29], 18:[13,17, 23, 19], 19:[24,18,14], 20:[21,15], 21:[20,22,16, 28], 22:[21,23,17], 23:[22,24, 18, 29], 24:[23,19], 28:[21,17], 29:[17,23], 30:[7,1], 31:[7, 13]}
diamond = matrix([[0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0],[0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],[0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0]])


### functions for constructing and modifying graphs ###

#input: two integers representing the dimensions of the grid
#output: a grid graph with n*m vertices
def grid(n,m):
    g = Graph()
    names = [[g.add_vertex() for i in range(m)] for j in range(n)]
    for i in range(len(names)):
        for j in range(len(names[i])):
            if j != 0: #connect the vertices in each row to each other
                g.add_edge(names[i][j], names[i][j-1])
            if i != 0: #connect the vertices in each column to each other
                g.add_edge(names[i][j], names[i-1][j])
    return g

#input: desired length of the cycle graph
#output: graph which is a cycle of legnth n
#note: this is neccecary because calling SageMath's build in cycle graph function wont allow you to add edges or vertices
def cycle(n):
    g = Graph()
    names = []
    for i in range(n):
        name = g.add_vertex()
        names.append(name)
    for i in range(len(names)):
        if (i==0):
            g.add_edge(names[0], names[len(names)-1])
        else:
            g.add_edge(names[i-1], names[i])
    return g

#input: the n to construct a k_2n graph
#output: the k_2n graph
def k_2n(n):
    g = Graph()
    names = []
    for i in range(n+2):
        name = g.add_vertex()
        names.append(name)
    for i in range(1,n+1):
        g.add_edge(names[0], names[i])
        g.add_edge(names[i], names[len(names)-1])
    return g

#input: quadrangulated graph
#output: None
#modifies: modifies graph by adding random copies of k_2m in n faces of the graph. M is set to 3 by default. Optional outterface to which no copies of k_2m will be added.
def add_random_k2m(graph, n , m=3, outter_face = None):
    if outter_face == None:
        outter_face = maximal_face(graph)
    faces = graph.faces()
    faces.remove(outter_face)
    shuffle(faces)
    n = min(n, len(faces)) #make sure n is less than or equal to the number of faces
    choosen_faces = faces[:n] #choose a random n faces
    for face in choosen_faces:
        names = []
        for i in range(m-2):
            name = graph.add_vertex()
            names.append(name)
        vertices = vertices_in_face(face)
        if (random.random() <= .5):
            a = vertices[0]
            b = vertices[2]
        else:
            a = vertices[1]
            b = vertices[3]
        for name in names:
            graph.add_edge(a,name)
            graph.add_edge(name,b)

#input: graph
#output: None
#modifies: modifies graph by  replacing every edge with a copy of k_2m where m is set to 3 by default.
def next_diamond_graph(graph, m = 3):
    for edge in graph.edges():
        a,b,c = edge
        names = []
        for i in range(m): #add the new vertices
            name = graph.add_vertex()
            names.append(name)
        graph.delete_edge(a,b) #delete the old edge
        for name in names: #add new edges
            graph.add_edge(a, name)
            graph.add_edge(name, b)

#input: graph
#output: None
#modifies: modifies the graph by replacing every edge in the graph by a path of length two making the graph bipartite
def make_bipartite(graph):
    for edge in graph.edges():
        a,b,c = edge
        name = graph.add_vertex()
        graph.add_edge(a,name)
        graph.add_edge(name,b)
        graph.delete_edge(a,b)

#input: graph and a face of even length of the graph. If length is less than 6 does nothing.
#output: none
#modifies: adds vertices and edges in face so that the maximum length of the new faces produced is 2 less than the length of the curtrent face. Moreover, the distances between the original vertices is unchanged.
def quadrangulate_helper(graph, face):
    assert len(face)%2 == 0, "the face %s was given as input to quadrangulate face" %face
    if (len(face)==6): #if the face has 6 edges then we can add a vertex in the middle and connect it two 3 of the 6 vertices
        vertices = vertices_in_face(face) #connect the new vertex to 3 of the 6 vertices that are non-adjacent
        first = vertices[0]
        second = vertices[2]
        third = vertices[4]
        name = graph.add_vertex()
        graph.add_edge(name, first)
        graph.add_edge(name, second)
        graph.add_edge(name, third)
        quadrangulate(graph) #now we pass the graph back to quadrangualte the rest of the faces.
    elif(len(face) > 6): #reduce the length of the face by 2 without changing the distances between points
        number_of_vertices = ((len(face)-6)/2)+1
        names = []
        for i in range(number_of_vertices):
            name = graph.add_vertex()
            names.append(name)
        for i in range(1, len(names)): #connect all vertices just added so they form a path
            graph.add_edge(names[i-1], names[i])

        vertices = vertices_in_face(face)
        vertices = vertices[:number_of_vertices + 4] #get a path of size number_of_vertices + 4 in the face
        graph.add_edge(vertices[0], names[0]) #attach the newly added path to the old face
        graph.add_edge(vertices[len(vertices)-1], names[len(names)-1])
        for i in range(len(names)):
            graph.add_edge(names[i],vertices[i+2]) #connect the newly added path at each vertex creating a series of boxes
        quadrangulate(graph) #now we need to pass to the function again for the recursion

#input: graph and an optional outterface for the graph. If no outterface is sspecified then the maximal length face is used instead.
#ouput: None
#modifies: if the graph has cycles of length 2 or 3 then first we call make_bipartite on the graph. Then the method modifies graph by adding new vertices and edges such that all faces (except the outter_face) are of length 4 and distances between original vertices is preserved
def quadrangulate(graph, outter_face = None):
    faces = graph.faces()
    triangles = False
    odd_length = False
    for face in faces:
        if (len(face) < 4):
            triangles = True
        if (len(face) % 2 != 0):
            odd_length = True
    if triangles or odd_length:
        make_bipartite(graph) #if there are faces of length 3 we make the graph bipartite and double the length of every face
    if outter_face == None:
        outter_face = maximal_face(graph)
    faces = graph.faces()
    for face in faces:
        if face != outter_face:
            if len(face) != 4:
                quadrangulate_helper(graph, face)
                break #we only want to change one face at a time because after we do the faces in the graph will be different


### Helper functions to be used later in the program ###

#input: graph
#ouput: the face in the graph which has maximal length
def maximal_face(graph):
    faces = graph.faces()
    max_length = -float("inf")
    max_face = None
    for face in faces:
        if (len(face) > max_length):
            max_length = len(face)
            max_face = face
    return max_face

#input: a face of the form [(u1,v1), (u2,v2), ..., (un,vn)]
#output: a list of the vertices in the face in either clockwise or counterclockwise order
def vertices_in_face(face):
    new_face = list(face) #make a copy of the face so we dont modify it
    size = len(face)
    vertices = []
    edges_seen = set()
    current_edge = new_face[0] #get the first edge in the face
    new_face = new_face[1:len(new_face)]
    edges_seen.add(current_edge)
    a,b = current_edge
    vertices.append(a)
    vertices.append(b)
    while(len(edges_seen) != size):
        for u,v in face:
            if ((u,v) not in edges_seen):
                if (vertices[len(vertices)-1] == u):
                    if v not in vertices:
                        vertices.append(v)
                    edges_seen.add((u,v))
    return vertices

#input: and edge of the form (u,v), a face of the form [(u1,v1), ..., (un,vn)]
#output: a list of two faces containing the edge
def faces_containing_edge(edge, faces):
    a,b = edge
    faces_with_edge = []
    for face in faces:
        if (((a,b) in face) or ((b,a) in face)):
            faces_with_edge.append(face)
    assert len(faces_with_edge) == 2, "faces with edge, %s has length greater than two" % faces_with_edge
    return faces_with_edge

#input: graph
#output: list of lists containing vertices with the same set of neighbors in the graph. only non-trivial sets (len > 1) are included.
def same_neighbors(graph):
    same_neighbors = []
    vertices = graph.vertices()
    for u in vertices:
        same_neighbors_u = [u]
        for v in vertices:
            if v != u:
                if graph.neighbors(u) == graph.neighbors(v):
                    same_neighbors_u.append(v)
        if len(same_neighbors_u) > 1:
            same_neighbors.append(same_neighbors_u)
    same_neighbors = [set(x) for x in same_neighbors]
    same = []
    for i in same_neighbors: #currently same_neighbors has n copies of any set of size n
        if i not in same: #so same will only have one copy for each set of vertices with the same neighbors
            same.append(i)
    same = [list(x) for x in same] #turns list of sets into a list of lists to be easier to work with
    return same


#input: graph
#output: a connection dictionary with frozensets of the faces of the graph as keys and the string "GF" as the value representing the flat connection on that face.
def flat_connection(graph):
    faces = graph.faces()
    flat_connection = {}
    for face in faces:
        flat_connection[frozenset(face)] = "GF"
    return flat_connection

#input: a face in a graph and a list of all geodesics in the graph
# the two geodesics that go through the face. Doesn't matter if the geodesics intersect or not.
#geodesics returned are a copy of the original geodesics.
def geodesics_in_face(face, geodesics):
    geodesics_in_face = []
    for geodesic in geodesics:
        for edge in geodesic:
            a,b = edge
            if((a,b) in face) or ((b,a) in face):
                if geodesic not in geodesics_in_face:
                    geodesics_in_face.append(list(geodesic))
    assert len(geodesics_in_face) <= 2, "more than two geodesics in the face %s" % face
    return geodesics_in_face

def faces_in_geodesic(faces, geodesic):
    faces_in_geoedsic = []
    for edge in geodesic:
        a,b = edge
        for face in faces:
            if (((a,b) in face) or ((b,a)in face)):
                if face not in faces_in_geodesic:
                    faces_in_geodesic.append(face)
    return faces_in_geodesic



### Methods for using networkx and matplotlib to draw graphs ###

#input: the edges in a graph and a position dictionary for vertices in the graph
#ouput: matplotlib will plot a point at the midpoints of each edge in the graph
def plot_midpoints(edges, pos):
	x=[]
	y=[]
	for edge in edges:
		a,b = edge
		x.append((pos[a][0] + pos[b][0])/2)
		y.append((pos[a][1] + pos[b][1])/2)
	plt.scatter(x,y)

#input: faces of a graph and a position dictionary for the vertices in the graph
#output: matplotlib will plot the points in the epicenters of each face in the graph
def plot_epicenters(faces, pos):
    for i in range(len(faces)):
        faces[i] = vertices_in_face(faces[i])
    x = []
    y = []
    for face in faces:
        if len(face) == 4:
            a,b,c,d = face
            x.append((pos[a][0]+pos[b][0]+pos[c][0]+pos[d][0])/4)
            y.append((pos[a][1]+pos[b][1]+pos[c][1]+pos[d][1])/4)
    plt.scatter(x,y)

#input: a list of edges representing a single geodesic in the graph, and a position dictionary for vertices in the graph.
#output: matplotlib will draw a directed line using the epicenters and midpoints in the graph.
def plot_single_geodesic(geodesic, pos, directed = False):
    x = []
    y = []
    for i in range(len(geodesic)-1):
        a,b = geodesic[i]
        c,d = geodesic[i+1]
        x.append((pos[a][0]+pos[b][0])/2)
        y.append((pos[a][1]+pos[b][1])/2)
        x.append((pos[a][0]+pos[b][0]+pos[c][0]+pos[d][0])/4)
        y.append((pos[a][1]+pos[b][1]+pos[c][1]+pos[d][1])/4)
        x.append((pos[c][0]+pos[d][0])/2)
        y.append((pos[c][1]+pos[d][1])/2)
    plt.plot(x,y, linestyle = "dashed")

#input: a list of geodesics, and a position dictionary for the graph. Directed option is not yet supported.
#output: a plot of every geodesic in the geodesics given.
def plot_geodesics(geodesics, pos, directed = False):
    for geodesic in geodesics:
        plot_single_geodesic(geodesic, pos)

#input: graph, optional outter face
#output: position dictionary for the vertices in the graph computed with modified Tuttes embedding
def tutte_embedding(graph, outter_face = None):
    if outter_face == None:
        outter_face = maximal_face(graph)
    pos = {}
    tmp = Graph()
    for edge in outter_face: #first just create a graph for the outterface and plot those points on a circle
        tmp.add_edge(edge)
    tmp_pos = nx.spectral_layout(tmp.networkx_graph()) #ensures that outterface is a convex shape
    pos.update(tmp_pos)
    outter_vertices = tmp.vertices()
    remaining_vertices = [x for x in graph.vertices() if x not in outter_vertices]
    if (len(remaining_vertices) > 0):
        #get the vertices that will be mapped to the same place
        overlapping = same_neighbors(graph)
        for i in overlapping:
            for j in i:
                if j in outter_vertices:
                    i.remove(j)
        overlapping = [x for x in overlapping if len(x) > 1]
        edges = []
        for x in overlapping:
            for u in x:
                faces = []
                for face in graph.faces():
                    if u in vertices_in_face(face):
                        faces.append(face)
                faces = [vertices_in_face(face) for face in faces]
                for face in faces:
                    face.remove(u)
                for face in faces:
                    for v in face:
                        if v not in graph.neighbors(u):
                            if (((u,v) not in edges) and (v,u) not in edges):
                                edges.append((u,v))
        for edge in edges:
            graph.add_edge(edge) #this creates triangles in the graph which may violate 3-connectivity but we take them out later
        size = len(remaining_vertices)
        A = [[0 for i in range(size)] for i in range(size)] #create the the system of equations that will determine the x and y positions of remaining vertices
        b = [0 for i in range(size)] #the elements of these matrices are indexed by the remaining_vertices list
        C = [[0 for i in range(size)] for i in range(size)]
        d = [0 for i in range(size)]
        for u in remaining_vertices:
            # each vertex gets an equation of the form u_x = 1/n(v1_x + v2_x + .. +vn_x), where vi are the neighbors of u, and an identitcle equation for y coords.
            i = remaining_vertices.index(u)
            neighbors = graph.neighbors(u)
            n = len(neighbors)
            A[i][i] = 1
            C[i][i] = 1
            for v in neighbors:
                if v in outter_vertices: #these vertices already have positions so we move them to the right side of the equation.
                    b[i] += float(pos[v][0])/n
                    d[i] += float(pos[v][1])/n
                else: #these rest will be variables and they are moved to the left side of the equation.
                    j = remaining_vertices.index(v)
                    A[i][j] = -(1/float(n))
                    C[i][j] = -(1/float(n))
        x = np.linalg.solve(A, b)
        y = np.linalg.solve(C, d)
        for u in remaining_vertices:
            i = remaining_vertices.index(u)
            pos[u] = [x[i],y[i]]
        for edge in edges: #delete the edges we added in earlier to make sure vertices are not mapped to the same location.
            graph.delete_edge(edge)
    return pos


# this is the main method for drawing a graph
# input: a graph, a connection for the graph, and a position dictionary which is set to be the Tutte embedding by default.
# options: node_size can be set (default 300), for nodesize less than 200 the vertex labels will be removed. epicenters, midpoints, and geodesics, cwill all be plotted y default but they can be turned off individually.
# output: a network drawing of the graph
def draw_graph(graph, connection, pos = None, node_size = 300, epicenters = False, midpoints = True, geodesics = True):
    if pos == None:
        pos = tutte_embedding(graph)
    N = graph.networkx_graph()
    faces = graph.faces()
    edges = [(a,b) for a,b,c in graph.edges()] # we don't care about the label feild in the edges
    draw_labels = (node_size >= 200)
    plt.clf() #if we don't clear the figure then we wil lbe drawing on top of previous pictures.
    nx.draw_networkx(N, pos, node_size = node_size, with_labels = draw_labels)
    if epicenters:
        plot_epicenters(faces, pos)
    if midpoints:
        plot_midpoints(edges, pos)
    if geodesics:
        geodesics = all_geodesics(graph, connection)
        plot_geodesics(geodesics, pos)
    plt.show()


### functions for getting the geodesic on the graph and the active faces of the graph ###

#input: an edge, a face of length 4, and an orrientation on that face.
#output: the outgoing edge for the face determined by the orrientation.
def next_edge(edge, face, orrientation):
    assert len(face)==4, "face %s is not of length 4" % face
    a,b = edge
    if (edge not in face):
        if((b,a) in face):
            a,b = b,a
    index = -1
    for i in range(len(face)):
        if (a,b) == face[i]:
            index = i
    assert index in range(4), "edge %s is not in the face" % (a,b)
    #given an edge of entry in a face of length 4 there are only three possibilities for a second edge to go out of. These 3 possibilities are labeled GF, type1, and type2.
    if (orrientation == "GF"):
        if (index == 0):
            return face[2]
        elif(index == 1):
            return face[3]
        elif (index == 2):
            return face[0]
        elif(index == 3):
            return face[1]
    elif(orrientation == "type1"):
        if (index==0):
            return face[1]
        elif(index==1):
            return face[0]
        elif(index==2):
            return face[3]
        elif(index==3):
            return face[2]
    elif(orrientation == "type2"):
        if (index==0):
            return face[3]
        elif(index==1):
            return face[2]
        elif(index==2):
            return face[1]
        elif(index==3):
            return face[0]

#input: a graph, a single edge to start the geodesic, a connection dictionary on the graph faces, optional outter face
#output: a list of edges representing the geodesic through the graph induced from the connection dictionary
def single_geodesic(graph, edge, connection, outter_face = None):
    faces = graph.faces()
    if outter_face == None:
        outter_face = maximal_face(graph)
    geodesic = [edge]
    faces_with_edge = faces_containing_edge(edge, faces)
    assert len(faces_with_edge) == 2, "there are more than two faces with the edge %s" % edge
    for face in faces_with_edge:
        if face == outter_face:
            faces_with_edge.remove(face)
    faces_with_edge = faces_with_edge[0] #get the face that contains the edge that isn't the outter face, if edge is internal than it doesn't matter which one we pick.
    current_edge = next_edge(edge, faces_with_edge, connection[frozenset(faces_with_edge)])
    a,b = current_edge
    while(((a,b) not in outter_face) and ((b,a) not in outter_face) and ((a,b) not in geodesic) and ((b,a) not in geodesic)): # we want to fall out of the loop when we have reached the outter face or an edge we have already seen
        geodesic.append(current_edge)
        faces_with_edge = faces_containing_edge(current_edge, faces)
        assert len(faces_with_edge) == 2, "there are more than two faces with the edge %s" % current_edge
        edges = []
        for face in faces_with_edge:
            edges.append(next_edge(current_edge, face, str(connection[frozenset(face)]))) #get the two edges in front of and behind the current edge
        for edge in edges:
            a,b = edge
            if ((a,b) in geodesic) or ((b,a) in geodesic):
                edges.remove(edge)
        if len(edges) == 1:
            current_edge = edges[0]
        elif len(edges) == 0:
            current_edge = geodesic[0] #if the edges in both directions are in the geodesic then just add the first edge of the geodesic again
        a,b = current_edge
    geodesic.append((a,b)) #once we break out of the loop we still want to add the lsat edge. If the geodesic is a loop the last edge will be the same as the first edge so drawing it becomes easier.
    return geodesic

#input: a graph, and a connection dictionary for the faces of the graph. Outter_face is an optional argument
#output: a list of lists which are the geodesics in the graph
def all_geodesics(graph, connection, outter_face = None):
    if outter_face == None:
        outter_face = maximal_face(graph)
    geodesics = []
    edges_in_outterface = set(outter_face)
    edges_remaining = set()
    for edge in graph.edges():
        a,b,c = edge
        edges_remaining.add((a,b))
    while(len(edges_remaining) != 0):
        if(len(edges_in_outterface) != 0): #we can to create geodesics going in from the outterface before we go back through the graph and get the cycle geodesics.
            edge = edges_in_outterface.pop()
        else:
            edge = edges_remaining.pop()
        geo = single_geodesic(graph, edge, connection, outter_face = outter_face) # use the outterface in this function to avoid the functions using different outter faces.
        geodesics.append(geo)
        for geo_edge in geo: #now we remove the edges we have already seen
            a,b = geo_edge
            if (a,b) in edges_in_outterface:
                edges_in_outterface.remove((a,b))
            elif (b,a) in edges_in_outterface:
                edges_in_outterface.remove((b,a))
            if (a,b) in edges_remaining:
                edges_remaining.remove((a,b))
            elif (b,a) in edges_remaining:
                edges_remaining.remove((b,a))
    return geodesics


#input: a graph and a list of geodesics (usually 1 or 2).
#output: a list of all the faces in the graph where one geodesic the set of geodesics intersects another geodesic in the set. Possibly including self intersections.
def faces_of_intersection(graph, geodesics):
    faces = graph.faces()
    all_edges = []
    intersection_faces = []
    for geodesic in geodesics:
        all_edges += geodesic #put all edges in all geodesics into one list
    all_edges = set(all_edges)
    for face in faces:
        is_intersection = True
        for edge in face:
            a,b = edge
            if(not (((a,b) in all_edges) or ((b,a) in all_edges))): #check to see if all edges in the face are in some geodesic
                is_intersection = False
        if is_intersection: #if they are all contained in some geodesic then add it to the list
            intersection_faces.append(face)
    #if any of the faces in the list has a geodesic that goes through two adjacent edges of the face then that face isn't really a face of intersection.
    for geodesic in geodesics:
        intersection_faces_copy = list(intersection_faces)
        for face in intersection_faces_copy:
            order_through_face = {}
            for i in range(len(geodesic)):
                a,b = geodesic[i]
                if (((a,b) in face) or ((b,a) in face)):
                    order_through_face[i] = (a,b) #order_though_face.keys() now holds the indicies for when the geodesic goes through the face
            if(len(order_through_face) != 0): #it could be the case that this geodesic never enters the face. If it does, then it must do so at least twice.
                assert len(order_through_face.keys()) >= 2, "faces of intersection: only one edge of the geodesic goes through this face"
                first_index = min(order_through_face.keys())
                a,b = order_through_face[first_index]
                del order_through_face[first_index]
                second_index = min(order_through_face.keys())
                c,d = order_through_face[second_index]
                del order_through_face[second_index]
                #(a,b) and (c,d) now holds the first entry edge and the first exist edge of the geodesic through the face.
                if(len(set([a,b,c,d])) != 4): #if a,b,c,d are not all distinct then the geodesic goes through two adjacent edges in the face. Therefore it cannot be a true intersection face.
                    intersection_faces.remove(face)
    return intersection_faces


#input: a graph, a face in the graph, and all geodesics on the graph
#ouput: a boolean value representing if the face is "active" meaning the geodesics that go through it intersect more than once.
def face_is_active(graph, face, geodesics):
    geodesics_in_face = []
    for geodesic in geodesics: # we need to find the two geodesics that flow through this face
        for edge in face:
            a,b = edge
            if(((a,b) in geodesic) or ((b,a) in geodesic)):
                if geodesic not in geodesics_in_face:
                    geodesics_in_face.append(geodesic)
    active = False
    if len(faces_of_intersection(graph, geodesics_in_face)) > 1:
        active = True
    return active


#input: a graph, all geodesics on the graph, optional outter face
#output: a list of the faces that are "active" in the graph.
def active_faces(graph, geodesics, outter_face = None):
    faces = graph.faces()
    if outter_face == None:
        outter_face = maximal_face(graph)
    active_faces = []
    for face in faces:
        if face != outter_face:
            if face_is_active(graph, face, geodesics):
                active_faces.append(face)
    return active_faces



### functions for finding metrics from cuts and computing the distortion of metrics ###

#input: a graph and a geodesic on the graph. The geodesic cannot have self intersections.
#output: a set of vertices which is an indicator for a cut in the graph
def get_cut(graph, geodesic):
    assert len(faces_of_intersection(graph, geodesic)) == 0, "the geodesic %s has self intersections" % geodesic
    g = Graph(graph)
    for edge in geodesic:
        g.delete_edge(edge)
    components = g.connected_components()
    #assert len(components) == 2, "cut has more than two components"
    cut = None
    sizeofCut = -float(Infinity) #return the set of vertices that is bigger
    for component in components:
        if (len(component) >= sizeofCut):
            sizeofCut = len(component)
            cut = component
    return cut #I think this wont work in some cases

#input: a graph and a list of geodesics in the graph
#output:a list of sets which are the cuts on the graph
def get_cuts(graph, geodesics):
    cuts = []
    for geodesic in geodesics:
        cuts.append(get_cut(graph, geodesic))
    return cuts

#input: a list of cuts which are sets of integers and two integers
#output: the distance between the two integers which is defined to be the number of sets that divides them.
def dist_cut_metric(cuts, a, b):
    dist = 0
    points = set([a,b])
    for cut in cuts:
        points = points.intersection(cut)
        if ((len(points) == 1) and (a != b)): #add one to the distance if exactly one of [a,b] is in the cut and a != b.
            dist += 1
        points = set([a,b])
    return dist

#Input: a graph and a list of cuts which are sets of vertices in the graph
#output: a dictionary of dictionaries storing the distances between vertices as given but a family of cuts
def cut_metric(graph, cuts):
    vertices = graph.vertices()
    metric = {}
    for u in vertices:
        metric[u] = {}
        for v in vertices:
            metric[u][v] = dist_cut_metric(cuts, u, v)
    return metric

#input: a graph, a double dictionary representing a distance between any two vertices in the graph
#output: the distorion of the metric relative to the shortest path metric.
def distortion_of_metric(graph, metric, verbose = False):
    shortest_path_metric = graph.distance_all_pairs()
    vertices = graph.vertices()
    contraction = -float("inf")
    expansion = -float("inf")
    for u in vertices:
        for v in vertices:
            if (u != v):
                if (shortest_path_metric[u][v] != 0):
                    if (float(metric[u][v])/shortest_path_metric[u][v] > expansion):
                        expansion = float(metric[u][v])/shortest_path_metric[u][v]
                elif (shortest_path_metric[u][v] == 0 and metric[u][v] != 0):
                    expansion = float("inf")
                if (metric[u][v] != 0):
                    if (float(shortest_path_metric[u][v])/metric[u][v] > contraction):
                        contraction = float(shortest_path_metric[u][v])/metric[u][v]
                elif(metric[u][v] == 0 and shortest_path_metric[u][v] != 0):
                    contraction = float("inf")
                    if verbose:
                        print("vertices " + str(u) + "," + str(v) +" are distance 0 in cut metric")
    print("contraction: " + str(contraction))
    print("expansion: " + str(expansion))
    distortion = contraction * expansion
    return distortion


### functions that alter the connection on the graph ###

#input: a graph, a single geodesic on that graph, and a connection on the graph
#output: a new connection on the graph which represents the anti_schafer decomposition of the geodesic given
def single_anti_schafer_decomposition(graph, geodesic, connection, anti = True, testing = True):
    new_connection = dict(connection) #make a copy of the connection so we don't modify it
    intersection_faces = faces_of_intersection(graph, [geodesic])
    for face in intersection_faces:
        sorted_edges = []
        sorted_edges_test = []
        for i in range(len(geodesic)):
            a,b = geodesic[i]
            if geodesic[i] in face:
                sorted_edges.append(geodesic[i]) #this gets the edges from the geodesic in the order in which they go through the face
                if testing: #just a sanity check
                    sorted_edges_test.append(geodesic[i])
            elif (b,a) in face:
                sorted_edges.append((b,a))
                if testing:
                    sorted_edges_test.append((a,b))
        if testing:
            for i in range(1, len(sorted_edges)):
                assert (geodesic.index(sorted_edges_test[i-1]) < geodesic.index(sorted_edges_test[i])),"the edges in single-anti-schafer decomposition are not in order"
        outgoing = [sorted_edges[1], sorted_edges[3]] #the 2nd and 4th element correspond to the geodesic leaving the face
        if ((set(outgoing) == set([face[0], face[1]])) or (set(outgoing) == set([face[2], face[3]]))):
            if anti:
                new_connection[frozenset(face)] = "type1"
            else:
                new_connection[frozenset(face)] = "type2"
        elif((set(outgoing) == set([face[1], face[2]])) or (set(outgoing) == set([face[0], face[3]]))):
            if anti:
                new_connection[frozenset(face)] = "type2"
            else:
                new_connection[frozenset(face)] = "type1"
        else:
            assert False, "geodesic in anti_schafer_decomp doesn't actually intersect itself"
    return new_connection


#input: a graph, and a connection on that graph
#ouput: a new connection on the graph that represents the anti_schafer decomposition for each geodesic on the graph
def anti_schafer_decomposition(graph, connection, anti = True, testing = True):
    new_connection = dict(connection)
    geodesics = all_geodesics(graph, connection)
    for geodesic in geodesics:
        new_connection = single_anti_schafer_decomposition(graph, geodesic, new_connection, anti = anti, testing = testing)
    if testing:
        geodesics = all_geodesics(graph, new_connection)
        for geodesic in geodesics:
            assert len(faces_of_intersection(graph, [geodesic])) == 0, "after anti-schafer decomp geodesic %s still self intersects" % geodesic
    return new_connection


#input: a graph, a face of the graph which is a face of intersection for some geodesics, and a connection on the graph
#requires: face is the intersection of two geodesics
#output: a new connection on the graph such that the face is no long
def uncross_geodesics_in_face(graph, face, geodesics, testing = True):
    original = list([list(geodesic) for geodesic in geodesics]) #save the original geodesics
    geodesics_through_face = geodesics_in_face(face, geodesics)
    if testing:
        assert len(geodesics_through_face) == 2, "the face %s is not the intersection of two geodesics. geodesics through face is %s, geodesics is %s" % (face, geodesics_through_face, geodesics)
        for geodesic in geodesics:
            assert face not in faces_of_intersection(graph, [geodesic]), "face is an intersection face for the geodesic %s" %geodesic

    for geodesic in geodesics_through_face:
        for edge in list(geodesic):
            a,b = edge
            if not (((a,b) in face) or ((b,a) in face)):
                geodesic.remove(edge)
    if testing:
        for geodesic in geodesics_through_face:
            assert len(geodesic) == 2, "there are not 2 edges left in the geodesic %s after removing edges not in the face" % geodesic

    if((set([geodesics_through_face[0][1], geodesics_through_face[1][0]]) == set([face[0], face[1]])) or (set([geodesics_through_face[0][1], geodesics_through_face[1][0]]) == set([face[2], face[3]]))):
        orrientation = "type1"
    else:
        orrientation =  "type2"
    geodesics = original #now that we have changeds the geodesics we can return them to their original value.
    #print("geodesics after call to box orrientations: %s" % geodesics)
    return orrientation


#intput: a graph, a list of two geodesics on the graph, and a connection on the graph
#output: a new connection on the graph where a random subset of boxes in the intersection of the two geodesics have been uncrossed. An odd number of boxes will remain unchanged.
def uncross_random_subset_of_faces_in_geodesics(graph, geodesics, connection, testing = True):
    new_connection = dict(connection)
    faces = faces_of_intersection(graph, geodesics)
    assert (len(faces)>1), "the geoedsics %s intersect only once" % geodesics
    if (len(faces) > 1):
        shuffle(faces)
        faces = faces[random.choice(range(1, len(faces), 2)):] #throw away a random, odd number of boxes to stay unchanged
        #original_geodesics = list([list(geodesic) for geodesic in geodesics])
        for face in faces:
            #geodesics = list(original_geodesics)
            #print("original are %s, and geodeiscs are %s" % (original_geodesics, geodesics))
            new_connection[frozenset(face)] = uncross_geodesics_in_face(graph, face, geodesics)
    return new_connection

#input: a grpah, and a connection on the graph
#output: a new connection on the graph where a random subset of boxes in the intersecting faces of every pair of geodesics have been unrossed
def uncross_random_subset_of_faces_in_graph(graph, connection, testing = True):
    new_connection = dict(connection)
    new_connection = anti_schafer_decomposition(graph, connection)
    geodesics = all_geodesics(graph, connection)
    pairs_of_geodesics_seen = []
    for geodesic_1 in geodesics:
        for geodesic_2 in geodesics:
            if geodesic_1 != geodesic_2:
                a,b = geodesic_1, geodesic_2
                if (len(faces_of_intersection(graph, [a,b])) > 1):
                    if ([a,b] not in pairs_of_geodesics_seen) and ([b,a] not in pairs_of_geodesics_seen):
                        new_connection = uncross_random_subset_of_faces_in_geodesics(graph, [a,b], new_connection)
                        pairs_of_geodesics_seen.append([a,b])
    return new_connection



### functions for computing the embedding ###

def embedding_connections(graph, k):
    connections = []
    connection = flat_connection(graph)
    connection = anti_schafer_decomposition(graph, connection)
    for i in range(k):
        new_connection = uncross_random_subset_of_faces_in_graph(graph, connection)
        new_connection = anti_schafer_decomposition(graph, new_connection)
        connections.append(new_connection)
    return connections

def embedding_cuts(graph, embeddings):
    cuts = []
    for embedding in embeddings:
        cuts.append(get_cuts(graph, all_geodesics(graph, embedding)))
    return cuts

def embedding_distortion(graph, family_of_cuts):
    distances = {}
    k = len(family_of_cuts)
    vertices = graph.vertices()
    for cuts in family_of_cuts:
        single_cut_metric = cut_metric(graph, cuts)
        for u in vertices:
            if u not in distances:
                distances[u] = {}
            for v in vertices:
                if v not in distances[u]:
                    distances[u][v] = 0
                distances[u][v] += single_cut_metric[u][v]
    for u in vertices:
        for v in vertices:
            distances[u][v] = float(distances[u][v])/k
    return distortion_of_metric(graph, distances, verbose = True)
