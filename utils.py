import sys
import numpy as np
import pandas as pd
import networkx as nx
import scipy
import matplotlib
import matplotlib.pyplot as plt
import itertools
import json
from scipy.stats import beta
from ast import literal_eval





########################### Utilities ###########################


def flip_edge(G, i, j):
    if (i,j) in list(G.edges()):
        G.remove_edge(i,j)
    else:
        G.add_edge(i,j)


def get_expressed(G,s):
    n = len(G.nodes())
    L = nx.laplacian_matrix(G).todense()
    z = np.dot(np.linalg.inv(np.identity(n) + L), s) 
    
    return z
        

def get_measure(G, s, measure = 'pol'):
    n = len(G.nodes())
    e = len(G.edges())
    
    L = nx.laplacian_matrix(G).todense()
    s_mean = (s - np.mean(s)).reshape((len(s),1))
    z_mean = np.dot(np.linalg.inv(np.identity(n) + L), s_mean)

    if measure =='pol':
        return np.dot(np.transpose(z_mean), z_mean)[0,0]     

    elif measure == 'dis':
        return np.dot(np.dot(np.transpose(z_mean), L), z_mean)[0,0]  
    
    elif measure == 'innate_dis':
        return np.dot(np.dot(np.transpose(s_mean), L), s_mean)[0,0] 

    elif measure == 'spectral_gap':
        lambdas = np.linalg.eigvals(L)
        return np.sort(lambdas)[1]
    else:
        Exception('Unknown measure requested.')
        return
    

def plot_graph(G, s, **kwargs):
    try:
        edge_color = [item for (key, item) in kwargs.items() if key =='edge_color'][0]
    except:
        edge_color = None

    plt.figure(figsize=(10,10))
    nx.draw(G, node_color = list(s[:,0]), **kwargs)
    if edge_color is not None:
        plt.gca().collections[0].set_edgecolor(edge_color)
    return



def save_data(outputs, s, name):
    data = dict()

    data['s'] = s.tolist()
    data['pol'] = outputs['pol']

    for (key,G) in outputs.items():
        if key != 'pol':
            data[key] = nx.adjacency_matrix(G).todense().tolist()

    with open('data/out/'+name+'.json', 'w') as fp:
        json.dump(data, fp)

    return


def load_data(name):
    with open('data/out/'+name+'.json', 'r') as fp:
        data = json.load(fp)

    for (key,val) in data.items():
        if key not in set(['s','pol']):
            data[key] = nx.from_numpy_matrix(np.array(data[key]))

    return data





########################### Loading Data ###########################



def load_twitter():
    s_df = pd.read_csv('data/in/opinion_twitter.txt', sep = '\t', header = None)
    w_df = pd.read_csv('data/in/edges_twitter.txt', sep = '\t', header = None)

    n = len(s_df[0].unique())

    s_df.columns = ["ID", "Tick", "Opinion"]

    # we take the opinion from the last appearance of the vertex ID in the list as its innate opinion
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

    G = nx.from_numpy_matrix(A)

    return (n, s, A, G, nx.laplacian_matrix(G).todense())




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

    G = nx.from_numpy_matrix(A)

    return(n, s, A, G, nx.laplacian_matrix(G).todense())



def process_df_cols(df, cols):
    
    df_out = df.copy()
    
    for colname in cols:
        for i in range(len(df)):
            
            if colname[0] == 'G':
                df_out.at[i,colname] = literal_eval(df.loc[:,colname].iloc[i])
                
            else:
                as_list = df.loc[:,colname].iloc[i].replace('\n','').replace('[','').replace(']','').replace(',','').split()
                df_out.at[i,colname] = [float(item) for item in as_list]


    return df_out




########################### Graph Generation ###########################


def make_erdos_renyi(n, p, weighted = True):
    rand_count = int(0.5*(n**2 - n))
    weights = scipy.sparse.random(1, rand_count, density = p).A[0]    
    G = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                # diagonal is 0s
                continue
            elif i < j:
                if weighted:
                    G[i][j] = weights[idx]
                else:
                    if weights[idx] > 0:
                        G[i][j] = 1
                idx += 1
            else:
                # adjacency matrix is symmetric 
                G[i][j] = G[j][i]
                
    return nx.from_numpy_matrix(G)



def make_block(n, p1, p2, weighted = True):
    # create two communities connected with density d1
    c1 = np.sort(np.random.choice(n, int(n/2), replace=False))
    n1 = len(c1)
    
    c2 = np.sort(list(set(np.arange(n)) - set(c1)))
    n2 = len(c2)

    weights1 = scipy.sparse.random(1, int(0.5*n1*(n1 - 1)), density=p1).A[0]
    weights2 = scipy.sparse.random(1, int(0.5*n2*(n2 - 1)), density=p1).A[0] 
    
    G = np.zeros((n, n))
    idx = 0
    for i in c1:
        for j in c1:
            if i == j:
                continue
            elif i < j:
                if weighted:
                    G[i][j] = weights1[idx]
                else:
                    if weights1[idx] > 0:
                        G[i][j] = 1
                idx += 1
            else:
                G[i][j] = G[j][i]
    
    idx = 0
    for i in c2:
        for j in c2:
            if i == j:
                continue
            elif i < j:
                if weighted:
                    G[i][j] = weights2[idx]
                else:
                    if weights2[idx] > 0:
                        G[i][j] = 1
                idx += 1
            else:
                G[i][j] = G[j][i]

    
    # weights for connections in between are of density d2
    idx = 0
    weights_between = scipy.sparse.random(1, n1*n2, density=p2).A[0]    
    for i in c1:
        for j in c2:
            if weighted:
                G[i][j] = weights_between[idx]
            else:
                if weights_between[idx] > 0:
                    G[i][j] = 1
            idx += 1        
    for i in c2:
        for j in c1:
            G[i][j] = G[j][i]
                
    return (c1, c2, nx.from_numpy_matrix(G))


def make_beta_opinions(a, b, n, c1, c2):
    # create innate opinion vector with community 1 ~ beta(a, b), community 2 ~ beta(b, a)
    # a and b parametrize the beta distribution
    # c1 and c2 describe the partition of vertices into communities using SBM
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
    # create graph using preferential attachment model
    # n: number of vertices the final graph has
    # G_0: the graph to build on
    # m: the number of nodes an incoming node connects to1

    n0 = len(G_0.nodes())

    init_G = nx.adjacency_matrix(G_0).A

    # create array containing each vertex's (weighted) degree
    links = np.zeros(n)
    for i in range(n0):
        links[i] = np.sum(init_G[i, :])
        
    # create n x n adjacency matrix with existing init_G
    G = np.zeros((n, n))
    G[:n0, :n0] = init_G
        
    graph_size = n0
    for i in range(n0, n):        
        # choose m nodes to connect new node to
        vs = np.random.choice(graph_size, m, p = links[:graph_size]/sum(links[:graph_size]))
        
        # update adjacency matrix and links
        for v in vs:
            w = np.random.rand()
            if weighted:
                G[i, v] = w
                G[v, i] = w
                links[i] += w
                links[v] += w

            else:
                G[i, v] = 1
                G[v, i] = 1
                links[i] += 1
                links[v] += 1
    
        graph_size += 1
        
    return nx.from_numpy_matrix(G)





########################### Getting Bounds ###########################


def get_bounds(G, s, edge):
    n = len(G.nodes())
    z = get_expressed(G,s)
    L = nx.laplacian_matrix(G).todense()
    I = np.identity(n)
    z_til = z - z.mean()

    (eigs,_) = np.linalg.eig(L)
    eigs.sort()

    l_gap = eigs[1]
    l_1 = eigs[n-1]

    eps = (np.transpose(z_til) @ (np.identity(n)+L) @ (I[edge[0]] - I[edge[1]]))[0,0]/(np.transpose(z_til) @ (I[edge[0]] - I[edge[1]]))[0,0] - 1/(2+(1+l_gap)**2)
    delta = ((z[edge[0]] - z[edge[1]])**2)[0,0]

    P_0 = get_measure(G,s,'pol')

    lb_r = (1-(2*eps*delta)/(3*n))*P_0
    ub_r = P_0 - (1+l_1)/(3+l_1)*(2*(np.transpose(z_til) @ (np.identity(n)+L) @ np.outer(I[edge[0]] - I[edge[1]],I[edge[0]] - I[edge[1]]) @ z_til)[0,0])

    lb_f = (-(2*eps*delta)/(3*n))*P_0
    ub_f = - (1+l_1)/(3+l_1)*(2*(np.transpose(z_til) @ (np.identity(n)+L) @ np.outer(I[edge[0]] - I[edge[1]],I[edge[0]] - I[edge[1]]) @ z_til)[0,0])


    return [[lb_f, ub_f], [lb_r, ub_r]]




########################### Optimization Heuritics ###########################

def opt_max_dis(G, s, G0 = None, 
                constraint = None, max_deg = None, max_deg_inc = None,
                G_ops = None, bounds = False):

    # Goal: Optimization procedure that adds non-edges with large expressed disagreement (w/ various optional constraints)
    #
    # G, G0: networkx Graph objects on n nodes (G0 is initial graph pre-perturbation, G is current)
    # s: (n,1) array-like, the innate opinions on G
    # constraint: string, one of [None, 'max-deg', 'max-deg-inc'] indicating constraint on max dis heuristic
    # max_deg: scalar, maximum degree allowed in output graph
    # max_deg_inc: n-dim array-like, the maximum degree increase allowed for each node of the graph
    # G_ops: optional, networkx Graph with which to check expressed opinions
    
    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()

    if G_ops is None:
        x = get_expressed(G,s)
    else:
        x = get_expressed(G_ops,s)


    nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))
    
    if constraint is None:
        dis_nonedges = [(x[i] - x[j])**2 for (i,j) in nonedges]  

    elif constraint == 'max-deg':
        if max_deg is None:
            raise ValueError('Must pass a value for Max. Degree')
        else:
            dis_nonedges = [(x[i] - x[j])**2 if (max(d[i],d[j]) < max_deg) else 0 for (i,j) in nonedges] 
        
    elif constraint == 'max-deg-inc':
        if max_deg_inc is None:
            raise ValueError('Must pass an array-like for Max. Degree Increase')
        else:
            d0 = G0.degree()
            dis_nonedges = [(x[i] - x[j])**2 if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else 0 for (i,j) in nonedges]  

    new_edge = list(nonedges)[dis_nonedges.index(max(dis_nonedges))]

    G_new.add_edges_from([new_edge])


    if bounds:
        return (G_new, get_bounds(G, s, new_edge))


    else:
        return (G_new, [[None, None], [None, None]])






def opt_max_grad(G, s, G0 = None, 
                 constraint = None, max_deg = None, max_deg_inc = None,
                 bounds = False):

    # Goal: Optimization procedure that maximizes the derivative of polarization
    #
    # G: networkx Graph object on n nodes
    # s: (n,1) array-like, the innate opinions on G
    
    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()
    x = get_expressed(G,s)
    x_tilde = x - x.mean()

    nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))

    # Precomputed for speed
    IpL_inv = np.linalg.inv(np.identity(n)+ nx.laplacian_matrix(G).todense())
    grad_pt_1 = 2*np.dot(np.transpose(x_tilde),IpL_inv)
    I_n = np.identity(n)


    if constraint is None:
        obj_nonedges = [np.dot(grad_pt_1 @ np.outer(I_n[i]-I_n[j], I_n[i]-I_n[j]),x_tilde) for (i,j) in nonedges]

    elif constraint == 'max-deg':
        if max_deg is None:
            raise ValueError('Must pass a value for Max. Degree')
        else:
            obj_nonedges = [np.dot(grad_pt_1 @ np.outer(I_n[i]-I_n[j], I_n[i]-I_n[j]),x_tilde) if (max(d[i],d[j]) < max_deg) else -np.inf for (i,j) in nonedges]
        
    elif constraint == 'max-deg-inc':
        if max_deg_inc is None:
            raise ValueError('Must pass an array-like for Max. Degree Increase')
        else:
            d0 = G0.degree()
            obj_nonedges = [np.dot(grad_pt_1 @ np.outer(I_n[i]-I_n[j], I_n[i]-I_n[j]),x_tilde) if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]) else -np.inf for (i,j) in nonedges]  


    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]

    G_new.add_edges_from([new_edge])


    if bounds:
        return (G_new, get_bounds(G, s, new_edge))


    else:
        return (G_new, [[None, None], [None, None]])






def opt_max_fiedler_diff(G, s = None, G0 = None, 
                         constraint = None, max_deg = None, max_deg_inc = None,
                         bounds = False):

    # Goal: Optimization procedure based on adding non-edge that maximizes difference in fiedler vector values
    #
    # G: networkx Graph object on n nodes
    # s: unused, exists for consistency with other functions

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()

    nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))

    L = nx.laplacian_matrix(G).todense()
    (l,V) = np.linalg.eig(L)

    v = V[:,list(l).index(np.sort(list(l))[1])]

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

    '''
    obj_nonedges = [(v[i] - v[j])**2 for (i,j) in nonedges] 
    '''

    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]

    G_new.add_edges_from([new_edge])


    if bounds:
        return (G_new, [[None, None], [None, None]])


    else:
        return (G_new, [[None, None], [None, None]])






def opt_stepwise_best(G, s, G0 = None, 
                      constraint = None, max_deg = None, max_deg_inc = None,
                      bounds = False):

    # Goal: Optimization procedure that maximizes the ratio of the derivative of polarization and the expressed disagreement
    #
    # G: networkx Graph object on n nodes
    # s: (n,1) array-like, innate opinions on G

    n = len(G.nodes())
    G_new = G.copy()  
    d = G.degree()
    
    nonedges = set(itertools.combinations(range(n),2)).difference(set(G_new.edges()))

    obj_nonedges = list()

    if constraint is None:
        for (i,j) in nonedges:
            G_tmp = G.copy()

            G_tmp.add_edge(i,j)
            obj_nonedges.append(-get_measure(G_tmp,s))

    elif constraint == 'max-deg':
        if max_deg is None:
            raise ValueError('Must pass a value for Max. Degree')

        else:
            for (i,j) in nonedges:
                G_tmp = G.copy()

                if max(d[i],d[j]) < max_deg:
                    G_tmp.add_edge(i,j)
                    obj_nonedges.append(-get_measure(G_tmp,s))
                else:
                    obj_nonedges.append(0)
        
    elif constraint == 'max-deg-inc':
        if max_deg_inc is None:
            raise ValueError('Must pass an array-like for Max. Degree Increase')

        else:
            d0 = G0.degree()

            for (i,j) in nonedges:
                G_tmp = G.copy()

                if (d[i]-d0[i]<max_deg_inc[i] and d[j]-d0[j]<max_deg_inc[j]):
                    G_tmp.add_edge(i,j)
                    obj_nonedges.append(-get_measure(G_tmp,s))
                else:
                    obj_nonedges.append(0)

    '''
    obj_nonedges = list()

    for (i,j) in nonedges:
        G_tmp = G.copy()

        G_tmp.add_edge(i,j)

        obj_nonedges.append(-get_measure(G_tmp,s))
    '''

    new_edge = list(nonedges)[obj_nonedges.index(max(obj_nonedges))]

    G_new.add_edges_from([new_edge])


    if bounds:
        return (G_new, get_bounds(G, s, new_edge))


    else:
        return (G_new, [[None, None], [None, None]])




########################### Optimization ###########################


def test_heuristics(G, s, k, G0 = None,
                    constraint = None, max_deg = None, max_deg_inc = None,
                    bounds = False):
    
    if max_deg_inc is not None:
        constraint = 'max-deg-inc'
        
    if max_deg is not None:
        constraint = 'max-deg'
    
    df = pd.DataFrame(columns = ['type', 'constraint', 'pol_vec', 'bounds_full', 'bounds_rel',
                                 'G_in', 's', 'G_out'], dtype = 'object')
    
    for fn_name in ['opt_max_dis', 'opt_max_grad', 'opt_max_fiedler_diff']:#,'opt_max_grad_dis_ratio', 'opt_stepwise_best']:
        print(fn_name)

        G_new = G.copy()

        pol_tmp = np.zeros(k+1)
        pol_tmp[0] = get_measure(G,s,'pol')

        bounds_tmp = [np.zeros((k+1,2)), np.zeros((k+1,2))]
        bounds_tmp[0][0:] = [pol_tmp[0], pol_tmp[0]]
        bounds_tmp[1][0:] = [pol_tmp[0], pol_tmp[0]]

        for i in range(k):

            (G_new, bounds) = eval(fn_name+'(G_new, s,'+
                                 'G0 = G ,constraint = constraint, max_deg = max_deg,'+
                                 'max_deg_inc = max_deg_inc, bounds = bounds)')

            pol_tmp[i+1] = get_measure(G_new, s, 'pol')
            bounds_tmp[0][i+1,:] = bounds[0]
            bounds_tmp[1][i+1,:] = bounds[1]
        
        df_tmp = pd.DataFrame({'type': fn_name, 'constraint': constraint, 'pol_vec': None, 'bounds_full': None, 'bounds_rel': None,
                               'G_in': None, 's': None, 'G_out': None}, index = [0], dtype = 'object')
        
        df_tmp.pol_vec.iloc[0] = pol_tmp#.tolist()
        df_tmp.bounds_rel.iloc[0] = bounds_tmp[1]
        if bounds:
            if fn_name != 'opt_max_fiedler_diff':
                df_tmp.bounds_full.iloc[0] = np.cumsum(bounds_tmp[0], axis = 0)
            else:
                df_tmp.bounds_full.iloc[0] = bounds_tmp[0]
        df_tmp.G_in.iloc[0] = nx.adjacency_matrix(G).todense().tolist()
        df_tmp.s.iloc[0] = np.transpose(s)[0,:]
        df_tmp.G_out.iloc[0] = nx.adjacency_matrix(G_new).todense().tolist()

        df = df.append(df_tmp, ignore_index = True)

        #print(df)

    return df


