from utils import *
import random
from tqdm import tqdm

def random_opinion_graph(s):
    """
    Returns a set of opinions of the same length as the innate set of opinions. 
    These will mimic opinions on a different issue 2.

    Input:
       s: original innate opinions
    Output: 
        s_random: randoly generated innate opinions

    """
    s_random = np.random.rand(len(s), 1)
    
    return s_random

def related_opinion_graph(s, noise, n_runs=10):
    """
    Returns new set of opinions. 
    This graph will have different innate opinions to mimic opinions on a different issue.
    These opinions are different from the original opinion by some noise factor. 
    Question: How uncorrelated can we make these to still see an effect? Lower noise = more correlation.

    Inputs:
        s: original innate opinion
        noise: noise added to calculation
    """
    all_runs_s = []

    # create n_runs of s values (each has different random noise)
    for value in s:
        s_new = []
        for _ in range(n_runs):
            rand_val = random.random()
            interpolated_value = value[0] * (1 - noise) + rand_val * noise
            s_new.append(interpolated_value)
        s_mean = np.mean(s_new)
        all_runs_s.append(s_mean)

    return all_runs_s
    

# TODO
# Function that takes new edge from G2 algorithm and adds to G1(politics) graph
# then calls test_heurestics 


########################### Network Optimization ###########################
def test_heuristics_two_graphs(G1, G2, s1, s2, 
                               k = None, 
                               constraint = None,
                               threshold = None):
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
        k = int(0.1*len(G1.edges())) # Default to 10% of num. edges
        # k = int(0.5*len(G1.nodes()))
        print(f'G1 current num edges: {len(G1.edges())}')
        print(f'Planner budget: {k}')
    
    df = pd.DataFrame(columns = ['type', 'constraint', 'pol_vec', 
                                 'pol_dis_vec', 'homophily_vec', 's_gap_vec',
                                 'G_in', 's', 'G_out', 'elapsed'], 
                                 dtype = 'object')
    
    start = time.time()

    G1_new = G1.copy()
    G2_new = G2.copy()

    pol_tmp = np.zeros(k+1)
    pol_tmp[0] = get_measure(G1,s1,'pol')
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
        (G2_new, nonedges, new_edge) = opt_min_dis(G1_new, s1, 
                                                  G2_new, s2,
                                                  constraint='g2_max_dis',
                                                  threshold=threshold)
        G1_new.add_edge(*new_edge)

        pol_tmp[i+1] = get_measure(G1_new, s1, 'pol')
        # Added
        pol_dis_tmp[i+1] = get_measure(G1_new, s1, 'pol_dis')

        homophily_tmp[i+1] = get_measure(G1_new, s1,'homophily')
        s_gap_tmp[i+1] = get_measure(G1_new, s1,'spectral_gap')
        
        if (i+1)*100/k >= prog:
            sys.stdout.write("Progress: " +str(prog) + "% Complete\n")
            sys.stdout.flush()
            prog = prog + 10
        
        
    end = time.time()
    elapsed = np.round(end - start, 4)

    df_tmp = pd.DataFrame({'type': 'opt_max_dis', 
                           'constraint': constraint, 
                           'threshold': threshold,
                           'pol_vec': None, 
                           'pol_dis_vec': None,
                           'homophily_vec': None, 
                           's_gap_vec': None, 
                           'G_in': None, 
                           's': None,
                           'G_out': None, 
                           'elapsed': elapsed},
                           index = [0], 
                           dtype = 'object')
    
    df_tmp.at[0,'threshold'] = threshold # added for opt_min_dis
    df_tmp.at[0,'pol_vec'] = pol_tmp.tolist()
    df_tmp.at[0,'pol_dis_vec'] = pol_dis_tmp.tolist()
    df_tmp.at[0,'homophily_vec'] = homophily_tmp.tolist()
    df_tmp.at[0,'s_gap_vec'] = s_gap_tmp.tolist()
    df_tmp.at[0,'G_in'] = nx.adjacency_matrix(G1).todense().tolist()
    df_tmp.at[0,'s'] = np.transpose(s1)[0,:].tolist()
    df_tmp.at[0,'G_out'] = nx.adjacency_matrix(G1_new).todense().tolist()

    df = pd.concat([df, df_tmp], ignore_index = True)

    sys.stdout.write(f"Done. Elapsed Time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}\n")
    sys.stdout.flush()

    return df
