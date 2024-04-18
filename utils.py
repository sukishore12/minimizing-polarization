import sys
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import scipy
import scipy.io
from scipy.stats import beta
import time
import itertools
from ast import literal_eval

from joblib import Parallel, delayed



########################### Useful Functions ###########################

def flip_edge(G, i, j):
    """
    Creates edge between two nodes if edge is not present.
    If edge already exists, removes.

    Inputs:
        G (nx.Graph): graph
        i, j: nodes to check edge existance
    
    Returns:
        None
    """

    if (i,j) in list(G.edges()):
        G.remove_edge(i,j)
    else:
        G.add_edge(i,j)


def get_expressed(G,s):
    """
    Returns expressed opinions z, given by z = (I+L)^{-1} s

    Inputs:
        G (nx.Graph): graph
        s (nd.array): innate opinion vector
    """

    n = len(G.nodes())
    L = nx.laplacian_matrix(G).todense()
    z = np.dot(np.linalg.inv(np.identity(n) + L), s) 
    
    return z
        

def get_measure(G, s, measure = 'pol'):
    """
    Returns specified measure given graph and innate opinions.

    Inputs:
        G (nx.Graph): graph
        s (ndarray): array of innate opinions
        measure (str): optional, measure to return
    """

    n = len(G.nodes())
    e = len(G.edges())
    
    L = nx.laplacian_matrix(G).todense()
    s_mean = (s - np.mean(s)).reshape((len(s),1))
    z_mean = np.dot(np.linalg.inv(np.identity(n) + L), s_mean)

    # Expressed Polarization
    if measure =='pol':
        return np.round(np.dot(np.transpose(z_mean), z_mean)[0,0], 4)

    elif measure == 'dis':
        return np.dot(np.dot(np.transpose(z_mean), L), z_mean)[0,0]

    # Polarizatio-Disagreement Index
    # I_{G,s} = P_{G,s} + D_{G,s}
    elif measure == 'pol_dis':
        pol = np.round(np.dot(np.transpose(z_mean), z_mean)[0,0], 4)
        dis = np.dot(np.dot(np.transpose(z_mean), L), z_mean)[0,0]

        return pol + dis
    
    elif measure == 'innate_dis':
        return np.dot(np.dot(np.transpose(s_mean), L), s_mean)[0,0] 

    elif measure == 'spectral_gap':
        return np.real(np.sort_complex(np.linalg.eigvals(L))[1])

    # Assortivity of Innate Opinions
    elif measure == 'homophily':
        return np.round(nx.numeric_assortativity_coefficient(G,'innate'),4)

    else:
        Exception('Unknown measure requested.')
        return


def SM_inv(A_inv, u, v):
    return A_inv - (A_inv @ np.outer(u, v) @ A_inv)/(1+v.T @ A_inv @ u)



########################### Loading Data ###########################



def load_twitter():
    s_df = pd.read_csv('data/in/opinion_twitter.txt', sep = '\t', header = None)
    w_df = pd.read_csv('data/in/edges_twitter.txt', sep = '\t', header = None)

    n = len(s_df[0].unique())

    s_df.columns = ["ID", "Tick", "Opinion"]

    # we take the opinion from the last appearance of the vertex ID in the list
    # as its innate opinion
    s = s_df.groupby(["ID"]).last()["Opinion"].values.reshape(n, 1)

    s_last = s_df.groupby(["ID"]).first()["Opinion"].values.reshape(n, 1)

    #s = (s - min(s))/max(s - min(s))

    # create adjacency matrix
    A = np.zeros((n, n))
    for i in range(1, n + 1):
        idx = np.where(w_df[0].values == i)[0]
        js = w_df[1].values[idx]
        for j in js:
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1
            
        idx = np.where(w_df[1].values == i)[0]
        js = w_df[0].values[idx]
        for j in js:
            A[i-1, j-1] = 1
            A[j-1, i-1] = 1

    G = nx.from_numpy_array(A)
    L = nx.laplacian_matrix(G).todense()

    s_dict = dict(zip(np.arange(len(s)),[int(np.round(item*10)) for item in s]))
    nx.set_node_attributes(G, s_dict, "innate")

    return (n, s, A, G, L)


def load_reddit():
    data = scipy.io.loadmat('data/in/Reddit.mat')

    n = data['Reddit'][0,0][0].shape[0]     # number of vertices = 556
    A = data['Reddit'][0,0][0].toarray()     # adjacency matrix in compressed sparse column format, convert to array
    nodemap = data['Reddit'][0, 0][1]     # mapping from node ID to labels 1-556 (not important)
    edges = data['Reddit'][0,0][2]     # list of edges (same as G, not used)
    s = data['Reddit'][0,0][5]     # labeled "recent innate opinions"

    # remove isolated vertices from the graph
    s = np.delete(s, 551)
    s = np.delete(s, 105)
    s = np.delete(s, 52)
    n -= 3
    s = s.reshape((n , 1))

    A = np.delete(A, 551, 1)
    A = np.delete(A, 551, 0)
    A = np.delete(A, 105, 1)
    A = np.delete(A, 105, 0)
    A = np.delete(A, 52, 1)
    A = np.delete(A, 52, 0)

    G = nx.from_numpy_array(A)
    L = nx.laplacian_matrix(G).todense()

    s_dict = dict(zip(np.arange(len(s)),[int(np.round(item*10)) for item in s]))
    nx.set_node_attributes(G, s_dict, "innate")

    return (n, s, A, G, L)


def load_blogs():
    G_raw =  nx.read_gml('data/in/polblogs.gml')

    # make undirected, remove multiedges
    G = nx.Graph(G_raw.to_undirected())

    # get largest connected component only
    G = G.subgraph(max(nx.connected_components(G), key=len))

    # change node labels to integers, keep old ones
    G = nx.convert_node_labels_to_integers(G, label_attribute = 'name')

    # set node attributes for 'innate'
    s_dict = nx.get_node_attributes(G, 'value')
    nx.set_node_attributes(G, s_dict, 'innate')

    n = len(G.nodes())
    s = np.array([list(s_dict.values())]).T
    A = nx.adjacency_matrix(G).todense()
    L = nx.laplacian_matrix(G).todense()

    return (n, s, A, G, L)


def process_df_cols(df, cols):
    df_out = df.copy()
    
    for colname in cols:
        for i in range(len(df)):
            
            if colname[0] == 'G':
                df_out.at[i,colname] = literal_eval(df.loc[:,colname].iloc[i])
                
            else:
#            df_out.loc[:,colname].iloc[i]= literal_eval(','.join(df.loc[:,colname].iloc[i].replace('\n','').split()))
                #df_out.at[i,colname]= literal_eval(','.join(df.loc[:,colname].iloc[i].replace('\n','').split()))            
                as_list = df.loc[:,colname].iloc[i].replace('\n','').replace('[','').replace(']','').replace(',','').split()
                #print(as_list)
                df_out.at[i,colname] = [float(item) for item in as_list]


    return df_out


########################### Graph Generation ###########################
def make_erdos_renyi(n, p, weighted = False):
    rand_count = int(0.5*(n**2 - n))
    weights = scipy.sparse.random(1, rand_count, density = p).A[0]    
    A = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                # diagonal is 0s
                continue
            elif i < j:
                if weighted:
                    A[i][j] = weights[idx]
                else:
                    if weights[idx] > 0:
                        A[i][j] = 1
                idx += 1
            else:
                # adjacency matrix is symmetric 
                A[i][j] = A[j][i]

    G = nx.from_numpy_array(A)

    # set innate opinions
    s = np.random.uniform(size = (n,1)) # between 0 and 1
    # multiplies innate opinions by 10, rounding to nearing int
    s_dict = dict(zip(np.arange(len(s)),[int(np.round(item*10)) for item in s]))
    nx.set_node_attributes(G, s_dict, "innate")
                
    return (G, s)


def make_block(n, p1, p2, a, b, weighted = False):
    # create two communities connected with density d1
    c1 = np.sort(np.random.choice(n, int(n/2), replace=False))
    n1 = len(c1)
    
    c2 = np.sort(list(set(np.arange(n)) - set(c1)))
    n2 = len(c2)

    weights1 = scipy.sparse.random(1, int(0.5*n1*(n1 - 1)), density=p1).A[0]
    weights2 = scipy.sparse.random(1, int(0.5*n2*(n2 - 1)), density=p1).A[0] 
    
    A = np.zeros((n, n))
    idx = 0
    for i in c1:
        for j in c1:
            if i == j:
                continue
            elif i < j:
                if weighted:
                    A[i][j] = weights1[idx]
                else:
                    if weights1[idx] > 0:
                        A[i][j] = 1
                idx += 1
            else:
                A[i][j] = A[j][i]
    
    idx = 0
    for i in c2:
        for j in c2:
            if i == j:
                continue
            elif i < j:
                if weighted:
                    A[i][j] = weights2[idx]
                else:
                    if weights2[idx] > 0:
                        A[i][j] = 1
                idx += 1
            else:
                A[i][j] = A[j][i]

    
    # weights for connections in between are of density d2
    idx = 0
    weights_between = scipy.sparse.random(1, n1*n2, density=p2).A[0]    
    for i in c1:
        for j in c2:
            if weighted:
                A[i][j] = weights_between[idx]
            else:
                if weights_between[idx] > 0:
                    A[i][j] = 1
            idx += 1        
    for i in c2:
        for j in c1:
            A[i][j] = A[j][i]
                

    G = nx.from_numpy_array(A)
    s = make_beta_opinions(a, b, n, c1, c2)
    s_dict = dict(zip(np.arange(len(s)),[int(np.round(item*10)) for item in s]))
    nx.set_node_attributes(G, s_dict, "innate")

    return (c1, c2, G, s)


def make_beta_opinions(a, b, n, c1, c2):
    """
    Create innate opinion vector with community 1 ~ beta(a, b), 
    community 2 ~ beta(b, a).

    Inputs:
        a and b: parametrize beta distribution
        n (int): number of nodes
        c1 and c2: describe partition of vertices into communities using SBM
    
    Returns:
        s (ndarray): (n,1) innate opinion vector
    """

    s1 = beta.rvs(a, b, size=int(n/2))
    s2 = beta.rvs(b, a, size=int(n/2))

    s = np.zeros((n, 1))
    idx1 = 0
    idx2 = 0
    for i in range(len(s)):
        if i in c1:
            s[i] = s1[idx1]
            idx1 += 1
        else:
            s[i] = s2[idx2]
            idx2 += 1
            
    return s



def make_pref_attach(n, G_0, m = 1, weighted = True):
    """
    Create graph using preferential attachment model.

    Inputs:
        n (int): number of vertices the final graph has
        G_0 (nx.Graph): graph to build on
        m (int): number of nodes an incoming node connects to1
    
    Returns:
        G (nx.Graph): graph of preferential attachment
        s (ndarray): (n,1) innate opinion vector
    """


    n0 = len(G_0.nodes())

    init_A = nx.adjacency_matrix(G_0).A

    # create array containing each vertex's (weighted) degree
    links = np.zeros(n)
    for i in range(n0):
        links[i] = np.sum(init_A[i, :])
        
    # create n x n adjacency matrix with existing init_G
    A = np.zeros((n, n))
    A[:n0, :n0] = init_A
        
    graph_size = n0
    for i in range(n0, n):        
        # choose m nodes to connect new node to
        vs = np.random.choice(graph_size, m, p = links[:graph_size]/sum(links[:graph_size]))
        
        # update adjacency matrix and links
        for v in vs:
            w = np.random.rand()
            if weighted:
                A[i, v] = w
                A[v, i] = w
                links[i] += w
                links[v] += w

            else:
                A[i, v] = 1
                A[v, i] = 1
                links[i] += 1
                links[v] += 1
    
        graph_size += 1

    G = nx.from_numpy_array(A)

    s = np.random.uniform(size = (n,1))
    s_dict = dict(zip(np.arange(len(s)),[int(np.round(item*10)) for item in s]))
    nx.set_node_attributes(G, s_dict, "innate")
                
    return (G, s)


########################### Optimization Heuritics ###########################
def opt_random_add(G, s = None, nonedges = None,
                   G0 = None, constraint = None, max_deg = None, max_deg_inc = None,
                   parallel = False, n_cores = -1):
    """
    Goal: Optimization procedure based on adding non-edge that maximizes 
            difference in fiedler vector values

    Inputs:
        G (nx.Graph): networkx Graph object on n nodes
        everything else: unused, exists for consistency with other functions
    
    Returns:
        G_new (nx.Graph): updated graph with new optimal edge
        new_edge (tuple): new edge added
    """

    n = len(G.nodes())
    G_new = G.copy()  

    if nonedges is None:
        #find vertices not currently connected by an edge
        nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))
    
    new_edge = list(nonedges)[np.random.choice(range(len(nonedges)), 1)[0]]

    G_new.add_edges_from([new_edge])

    return (G_new, nonedges.difference(set([new_edge])))


def get_diff(x, i, j):
    return abs(x[i] - x[j])


def opt_max_dis(G, s, nonedges = None,
                G0 = None, constraint = None, max_deg = None, max_deg_inc = None,
                G_ops = None, 
                parallel = False, n_cores = -1):
    """
    Goal: Optimization procedure that adds non-edges with large 
        expressed disagreement (w/ various optional constraints)
    
    Inputs:
        G (nx.Graph): current graph object on n nodes
        G0 (nx.Graph): initial graph object pre-perturbation
        s (ndarray): (n,1) array-like, the innate opinions on G
        constraint (str): one of [None, 'max-deg', 'max-deg-inc'] 
                            indicating constraint type
        max_deg (int): scalar, maximum degree allowed in output graph
        max_deg_inc (ndarray): n-dim array-like, the maximum degree increase 
                                allowed for each node of the graph
        G_ops (nx.Graph): optional, graph with which to compute expressed 
                            opinions to use
    
    Returns:
        G_new (nx.Graph): updated graph with new optimal edge
        new_edge (tuple): new edge added
    """

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()

    if G_ops is None:
        x = get_expressed(G,s)
    else:
        x = get_expressed(G_ops,s)

    if nonedges is None:
        nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))
    
    if not parallel:
        if constraint is None:
            #Calculates objective function for each pair of nodes: disagreement D_{ij}(x) = (x_i - x_j)^2
            obj_nonedges = [(x[i] - x[j])**2 for (i,j) in nonedges]  

        elif constraint == 'max-deg':
            #Only connects nodes that do not already exceed a maximum degree
            if max_deg is None:
                raise ValueError('Must pass a value for Max. Degree')
            else:
                obj_nonedges = [(x[i] - x[j])**2 if (max(d[i],d[j]) < max_deg) else 0 for (i,j) in nonedges] 
            
        elif constraint == 'max-deg-inc':
            # Calculates similar objective function to none case, but checks that the increase in degree compared to
            # degree found in G0 graph does not exceed some number
            if max_deg_inc is None:
                raise ValueError('Must pass an array-like for Max. Degree Increase')
            else:
                d0 = G0.degree()
                obj_nonedges = [(x[i] - x[j])**2 if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else 0 for (i,j) in nonedges]  

    else:
        #runs parallel jobs
        if constraint is None:
            obj_nonedges = Parallel(n_jobs = n_cores)(delayed(get_diff)(x, i, j) for (i,j) in nonedges)

        else:
            raise ValueError('Not Implemented...')

    # Finds index of maximum objective function and corresponding two vertices. Adds an edge between these two vertices.
    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]
    G_new.add_edges_from([new_edge])

    return (G_new, nonedges.difference(set([new_edge])))



def opt_max_fiedler_diff(G, s = None, nonedges = None,
                         G0 = None, constraint = None, max_deg = None, max_deg_inc = None,
                         parallel = False, n_cores = -1):
    """
    Goal: Optimization procedure based on adding non-edge that maximizes difference in fiedler vector values
    
    Inputs:
        G (nx.Graph): networkx Graph object on n nodes
        s: unused, exists for consistency with other functions
        constraint (str): string, one of [None, 'max-deg', 'max-deg-inc'] 
                            indicating constraint type
        max_deg (int): scalar, maximum degree allowed in output graph
        max_deg_inc (ndarray): n-dim array-like, the maximum degree increase 
                                allowed for each node of the graph

    Returns:
        G_new (nx.Graph): updated graph with new optimal edge
        new_edge (tuple): new edge added
    """

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()

    if nonedges is None:
        nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))

    L = nx.laplacian_matrix(G).todense()
    (l,V) = np.linalg.eig(L)

    # Find second smallest eigenvector- Fiedler vector
    v = V[:,list(l).index(np.sort(list(l))[1])]

    # Same objective function, this time with v instead of x
    if not parallel:
        if constraint is None:
            obj_nonedges = [(v[i] - v[j])**2 for (i,j) in nonedges]

        elif constraint == 'max-deg':
            if max_deg is None:
                raise ValueError('Must pass a value for Max. Degree')
            else:
                obj_nonedges = [(v[i] - v[j])**2 if (max(d[i],d[j]) < max_deg) else 0 for (i,j) in nonedges]
            
        elif constraint == 'max-deg-inc':
            if max_deg_inc is None:
                raise ValueError('Must pass an array-like for Max. Degree Increase')
            else:
                d0 = G0.degree()
                obj_nonedges = [(v[i] - v[j])**2 if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else 0 for (i,j) in nonedges]  

    else:
        if constraint is None:
            obj_nonedges = Parallel(n_jobs = n_cores)(delayed(get_diff)(v, i, j) for (i,j) in nonedges)

        else:
            raise ValueError('Not Implemented...')


    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]
    G_new.add_edges_from([new_edge])

    return (G_new, nonedges.difference(set([new_edge])))




def get_grad(grad_pt, i, j):
    return grad_pt[i,i] + grad_pt[j,j] - 2*grad_pt[i,j]


def opt_max_grad(G, s, nonedges = None,
                 G0 = None, constraint = None, max_deg = None, max_deg_inc = None,
                 parallel = False, n_cores = -1):
    """
    Goal: Optimization procedure that maximizes the derivative of polarization
    
    Inputs:
        G (nx.Graph): graph object on n nodes
        s (ndarray): (n,1) array-like, the innate opinions on G
        constraint (str): one of [None, 'max-deg', 'max-deg-inc']
                          indicating constraint type
        max_deg (int): scalar, maximum degree allowed in output graph
        max_deg_inc (ndarray): n-dim array-like, the maximum degree increase 
                            allowed for each node of the graph
    
    Returns:
        G_new (nx.Graph): updated graph with new optimal edge
        new_edge (tuple): new edge added
    """

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()
    x = get_expressed(G,s)
    x_tilde = x - x.mean()

    if nonedges is None:
        nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))


    # Precomputed for speed
    I_n = np.identity(n)
    grad_pt = 2*np.outer(x_tilde, x_tilde) @ np.linalg.inv(I_n+ nx.laplacian_matrix(G).todense())

    #sys.stdout.write("Computed Persistent Grad Matrix\n")
    #sys.stdout.flush()

    # Same as before, optimizing over gradient of polarization
    # Coordinate descent
    if not parallel:
        if constraint is None:
            obj_nonedges = [get_grad(grad_pt, i, j) for (i,j) in nonedges]


        elif constraint == 'max-deg':
            if max_deg is None:
                raise ValueError('Must pass a value for Max. Degree')
            else:
                obj_nonedges = [get_grad(grad_pt, i, j) if (max(d[i],d[j]) < max_deg) else -np.inf for (i,j) in nonedges]
            
        elif constraint == 'max-deg-inc':
            if max_deg_inc is None:
                raise ValueError('Must pass an array-like for Max. Degree Increase')
            else:
                d0 = G0.degree()
                obj_nonedges = [get_grad(grad_pt, i, j) if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else -np.inf for (i,j) in nonedges]  

    else:
        if constraint is None:
            obj_nonedges = Parallel(n_jobs = n_cores)(delayed(get_grad)(grad_pt, i, j) for (i,j) in nonedges)

        else:
            raise ValueError('Not Implemented...')

    #sys.stdout.write("Computed Objective\n")
    #sys.stdout.flush()
   
    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]
    G_new.add_edges_from([new_edge])

    return (G_new, nonedges.difference(set([new_edge])))


def opt_stepwise_best(G, s, G0 = None, 
                      constraint = None, max_deg = None, max_deg_inc = None,
                      bounds = False, parallel = False):
    """
    Goal: Optimization procedure that adds the optimal edge stepwise.
    Greedy stepwise approach where weight of k edges are saturated iteratively,
    one at a time. Simpler setting is tractable. Described in section 4.2. 

    Inputs:
        G (nx.Graph): graph on n nodes
        s (ndarray): (n,1) array-like, innate opinions on G

    Returns:
        G_new (nx.Graph): updated graph with new optimal edge
        new_edge (tuple): new edge added
    """

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()
    x = get_expressed(G,s)
    x_tilde = x - x.mean()    
    
    nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))

    # Precomputed for speed
    IpL_inv = np.linalg.inv(np.identity(n)+ nx.laplacian_matrix(G).todense())
    I_n = np.identity(n)


    if constraint is None:
        obj_nonedges = [- x_tilde.T @ np.linalg.matrix_power(SM_inv(IpL_inv, I_n[i] - I_n[j], I_n[i] - I_n[j]), 2) @ x_tilde for (i,j) in nonedges]
        #obj_nonedges = [- x_tilde.T @ np.linalg.inv(I_n + nx.laplacian_matrix(G).todense() + np.outer(I_n[i] - I_n[j], I_n[i] - I_n[j]))**2 @ x_tilde for (i,j) in nonedges]

    elif constraint == 'max-deg':
        if max_deg is None:
            raise ValueError('Must pass a value for Max. Degree')
        else:
            obj_nonedges = [- x_tilde.T @ SM_inv(IpL_inv, I_n[i] - I_n[j], I_n[i] - I_n[j])**2 @ x_tilde if (max(d[i],d[j]) < max_deg) else -np.inf for (i,j) in nonedges]
        
    elif constraint == 'max-deg-inc':
        if max_deg_inc is None:
            raise ValueError('Must pass an array-like for Max. Degree Increase')
        else:
            d0 = G0.degree()
            obj_nonedges = [- x_tilde.T @ SM_inv(IpL_inv, I_n[i] - I_n[j], I_n[i] - I_n[j])**2 @ x_tilde if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else -np.inf for (i,j) in nonedges]  


    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]
    G_new.add_edges_from([new_edge])

    return (G_new, nonedges.difference(set([new_edge])))





########################### Network Optimization ###########################
def test_heuristics(funs, G, s, k = None, G0 = None,
                    constraint = None, max_deg = None, max_deg_inc = None,
                    parallel = False, n_cores = -1):
    """
    Measure heuristics for expressed polarization, spectral gap, 
    assortativity of innate opinions.

    Inputs:
        funs (List[str]): list of functions to optimize over 
                        ex: ['opt_random_add', 'opt_max_dis']
        G (nx.Graph): input graph
        k (int): planner's budget
    """ 

    if k is None:
        #k = int(0.1*len(G.edges())) # Default to 10% of num. edges
        k = int(0.5*len(G.nodes()))

    if max_deg_inc is not None:
        constraint = 'max-deg-inc'
        
    if max_deg is not None:
        constraint = 'max-deg'
    
    df = pd.DataFrame(columns = ['type', 'constraint', 'pol_vec', 'homophily_vec', 's_gap_vec',
                                 'G_in', 's', 'G_out', 'elapsed'], dtype = 'object')
    
    for fn_name in funs:
        sys.stdout.write("\n-----------------------------------\n"+ fn_name+"\n-----------------------------------\n")
        sys.stdout.flush()

        start = time.time()

        G_new = G.copy()

        pol_tmp = np.zeros(k+1)
        pol_tmp[0] = get_measure(G,s,'pol')
        homophily_tmp = np.zeros(k+1)
        homophily_tmp[0] = get_measure(G,s,'homophily')
        s_gap_tmp = np.zeros(k+1)
        s_gap_tmp[0] = get_measure(G,s,'spectral_gap')

        sys.stdout.write("Progress: 0% Complete\n")
        sys.stdout.flush()
        prog = 10

        nonedges = None

        for i in range(k):
           
            (G_new, nonedges) = eval(fn_name+'(G_new, s,'+
                                 'G0 = G0, constraint = constraint, max_deg = max_deg,'+
                                 'max_deg_inc = max_deg_inc, nonedges = nonedges,'+
                                 'parallel = parallel, n_cores = n_cores)')

            pol_tmp[i+1] = get_measure(G_new, s, 'pol')
            homophily_tmp[i+1] = get_measure(G_new, s,'homophily')
            s_gap_tmp[i+1] = get_measure(G_new,s,'spectral_gap')
            
            if (i+1)*100/k >= prog:
                sys.stdout.write("Progress: " +str(prog) + "% Complete\n")
                sys.stdout.flush()
                prog = prog + 10
            
            
        end = time.time()
        elapsed = np.round(end - start, 4)

        df_tmp = pd.DataFrame({'type': fn_name, 'constraint': constraint, 'pol_vec': None, 'homophily_vec': None, 's_gap_vec': None,
                               'G_in': None, 's': None, 'G_out': None, 'elapsed': elapsed}, index = [0], dtype = 'object')
        
        df_tmp.at[0,'pol_vec'] = pol_tmp.tolist()
        df_tmp.at[0,'homophily_vec'] = homophily_tmp.tolist()
        df_tmp.at[0,'s_gap_vec'] = s_gap_tmp.tolist()
        df_tmp.at[0,'G_in'] = nx.adjacency_matrix(G).todense().tolist()
        df_tmp.at[0,'s'] = np.transpose(s)[0,:].tolist()
        df_tmp.at[0,'G_out'] = nx.adjacency_matrix(G_new).todense().tolist()

        df = pd.concat([df, df_tmp], ignore_index = True)

        sys.stdout.write("Done. Elapsed Time: " + time.strftime('%H:%M:%S', time.gmtime(elapsed))+"\n")
        sys.stdout.flush()

    return df



