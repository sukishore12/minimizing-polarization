from utils import *

def random_opinion_graph(graph_data):
    """
    Returns a graph with the same number of vertices, adjacency matrix, graph, and laplacian matrix.
    This graph will have random innate opinions to mimic opinions on a different issue.

    Inputs:
        graph_data: output of load functions
            n: number of vertices
            s: original innate opinions
            A: adjacency matrix
            G: graph
            L: Laplacian matrix
    """
    n, _, A, G, L = graph_data

    s_random = np.random.rand(n, 1)

    graph_data_new = (n, s_random, A, G, L)
    
    return graph_data_new

def related_opinion_graph(graph_data, correlation_factor=0.5, noise_level=0.1):
    """
    Returns a graph with the same number of vertices, adjacency matrix, graph, and laplacian matrix.
    This graph will have different innate opinions to mimic opinions on a different issue.
    These opinions are correlated with the original opinion. 
    Question: How uncorrelated can we make these to still see an effect?

    Inputs:
        graph_data: output of load functions
            n: number of vertices
            s: original innate opinions
            A: adjacency matrix
            G: graph
            L: Laplacian matrix
        correlation_factor: determines how correlated to make new innate opinions
        noise_level: noise added to calculation
    """
    n, s_twitter, A, G, L = graph_data
    
    transformation_matrix = np.eye(n) + correlation_factor * np.random.randn(n, n)

    s_new = np.dot(transformation_matrix, s_twitter)

    s_new += noise_level * np.random.randn(n, 1)

    s_new = np.clip(s_new, 0, 1)

    graph_data_new = (n, s_new, A, G, L)
    
    return graph_data_new

# TODO
# Function that takes new edge from G2 algorithm and adds to G1(politics) graph
# then calls test_heurestics 


########################### Network Optimization ###########################
def test_heuristics_two_graphs(G1, G2, s1, s2, k = None, G0 = None,
                    constraint = None, max_deg = None, max_deg_inc = None,
                    parallel = False, n_cores = -1):
    """
    Measure heuristics for expressed polarization, spectral gap, 
    assortativity of innate opinions.

    Inputs:
        funs (List[str]): list of functions to optimize over 
                        ex: ['opt_random_add', 'opt_max_dis']
        G1 (nx.Graph): input graph of opinions 1 (politics), graph trying to 
                        reduce polarization on
        G2 (nx.Graph): input graph of opinions 2 (sports), secondary opinions graph
        k (int): planner's budget, default: half of number of nodes
    """ 

    if k is None:
        #k = int(0.1*len(G.edges())) # Default to 10% of num. edges
        k = int(0.5*len(G1.nodes()))

    if max_deg_inc is not None:
        constraint = 'max-deg-inc'
        
    if max_deg is not None:
        constraint = 'max-deg'
    
    df = pd.DataFrame(columns = ['type', 'constraint', 'pol_vec', 
                                 'pol_dis_vec', 'homophily_vec', 's_gap_vec',
                                 'G_in', 's', 'G_out', 'elapsed'], 
                                 dtype = 'object')
    
    start = time.time()

    G1_new = G1.copy()
    G2_new = G2.copy()

    pol_tmp = np.zeros(k+1)
    pol_tmp[0] = get_measure(G1,s1,'pol')
    # Added
    pol_dis_tmp = np.zeros(k+1)
    pol_dis_tmp[0] = get_measure(G1,s1,'pol_dis')
    homophily_tmp = np.zeros(k+1)
    homophily_tmp[0] = get_measure(G1,s1,'homophily')
    s_gap_tmp = np.zeros(k+1)
    s_gap_tmp[0] = get_measure(G1,s1,'spectral_gap')

    sys.stdout.write("Progress: 0% Complete\n")
    sys.stdout.flush()
    prog = 10

    nonedges = None

    for i in range(k):
        # this is where the optimization functions is called
        # (G_new, nonedges) = eval(fn_name+'(G_new, s,'+
        #                      'G0 = G0, constraint = constraint, max_deg = max_deg,'+
        #                      'max_deg_inc = max_deg_inc, nonedges = nonedges,'+
        #                      'parallel = parallel, n_cores = n_cores)')
        (G2_new, nonedge, new_edge) = opt_max_dis(G2, s2, find_min_dis=True) # finding MIN
        G1_new.add_edges_from([new_edge])

        pol_tmp[i+1] = get_measure(G1_new, s1, 'pol')
        # Added
        pol_dis_tmp[i+1] = get_measure(G1_new, s1, 'pol_dis')
        homophily_tmp[i+1] = get_measure(G1_new, s1,'homophily')
        s_gap_tmp[i+1] = get_measure(G1_new,s1,'spectral_gap')
        
        if (i+1)*100/k >= prog:
            sys.stdout.write("Progress: " +str(prog) + "% Complete\n")
            sys.stdout.flush()
            prog = prog + 10
        
        
    end = time.time()
    elapsed = np.round(end - start, 4)

    df_tmp = pd.DataFrame({'type': 'opt_max_dis', 'constraint': constraint, 
                            'pol_vec': None, 'pol_dis_vec': None,
                            'homophily_vec': None, 
                            's_gap_vec': None, 'G_in': None, 
                            's': None, 'G_out': None, 'elapsed': elapsed}, 
                            index = [0], dtype = 'object')
    
    df_tmp.at[0,'pol_vec'] = pol_tmp.tolist()
    df_tmp.at[0,'pol_dis_vec'] = pol_dis_tmp.tolist()
    df_tmp.at[0,'homophily_vec'] = homophily_tmp.tolist()
    df_tmp.at[0,'s_gap_vec'] = s_gap_tmp.tolist()
    df_tmp.at[0,'G_in'] = nx.adjacency_matrix(G1).todense().tolist()
    df_tmp.at[0,'s'] = np.transpose(s1)[0,:].tolist()
    df_tmp.at[0,'G_out'] = nx.adjacency_matrix(G1_new).todense().tolist()

    df = pd.concat([df, df_tmp], ignore_index = True)

    sys.stdout.write("Done. Elapsed Time: " + time.strftime('%H:%M:%S', time.gmtime(elapsed))+"\n")
    sys.stdout.flush()

    return df



