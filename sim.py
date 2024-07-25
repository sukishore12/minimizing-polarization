import argparse
import sys
import os
from datetime import datetime
import numpy as np
import networkx as nx

from utils import *
from two_graph_utils import *

# funs = ['opt_random_add', 'opt_max_dis', 'opt_max_fiedler_diff',
#         'opt_max_grad']

FUNS = ['opt_random_add', 'opt_max_fiedler_diff', 
        'opt_max_common_ground', 'opt_max_2grad']
RELATED_VALS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

##### Two Opinions #####
def twitter_random(n=1, dir_path:str = f'data/out/raw/tw/'):
        """n = 1, random second opinion graph"""
        sys.stdout.write('---------------- Twitter and Random -----------------------\n')
        sys.stdout.write(f'-------------------- n = {n} -----------------------\n')
        sys.stdout.flush()

        (_, s, _, G, _) = load_twitter()
        test_random('tw', G, s, n=n, save_data_path=dir_path)

def test_random(graph_code, G1, s1, n=1, save_data_path:str = f'data/out/raw/'):
        """General function for testing all functions on input graph G1 and opinion s1.
        
        Args:
                graph_code (str): Code for the graph type ('tw' or 'rd'). Used to save the csv file.
                G1 (nx.Graph): graph of primary opinion
                s1 (np.array): Opinion vector corresponding to G1
                n (int, optional): Number of random second opinion graphs. Defaults to 1.
                save_data_path (str, optional): Directory to save csv files. Defaults to f'data/out/raw/'.
        """

        G_new = G1.copy()
        # Make second opinion graph, with relatedness n
        s_new = related_opinion_graph(s1, n)

        df = test_heuristics_two_graphs(FUNS, G1, G2=G_new, 
                                        s1=s1, s2=s_new)
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = os.path.join(save_data_path, current_date)  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/{graph_code}_{n}_{current_time}.csv')

##### Two Opinions #####
def reddit_random(n=1):
        
        sys.stdout.write('----------------------- Reddit and Random -----------------------\n')
        sys.stdout.flush()

        (n_rd, s_rd, A_rd, G_rd, L_rd) = load_reddit()
        G_new = G_rd.copy()
        s_new = related_opinion_graph(s_rd, n)

        df = test_heuristics_two_graphs(FUNS, G_rd, G2=G_new, 
                                        s1=s_rd, s2=s_new)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/rd/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/rd_{n}_{current_time}.csv')

def blogs_random(n=1):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_blogs()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n)

        df = test_heuristics_two_graphs(FUNS, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/bl/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/bl_rel_{n}_{current_time}.csv')

def er_random(n=1):
        sys.stdout.write('----------------------- ER Graphs -----------------------\n')
        sys.stdout.flush()

        num_verts = 1000
        p = 0.02

        np.random.seed(0)
        (G_er, s_er) = make_erdos_renyi(num_verts, p, weighted = False)
        G_new = G_er.copy()
        s_new = related_opinion_graph(s_er, n)

        df = test_heuristics_two_graphs(FUNS, G_er, G2=G_new, 
                                        s1=s_er, s2=s_new)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/er/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/er_{n}_{current_time}.csv')

def sbm_random(n=1):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p1 = 0.05
        p2 = 0.005
        a = 5
        b = 1

        np.random.seed(0)
        (c1, c2, G_sbm, s_sbm) = make_block(n, p1, p2, a, b, weighted = False)
        G_new = G_sbm.copy()
        s_new = related_opinion_graph(s_sbm, n)

        df = test_heuristics_two_graphs(FUNS, G_sbm, G2=G_new, 
                                        s1=s_sbm, s2=s_new)
        
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/sbm/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/sbm_rel_{n}_{current_time}.csv')

def pa_random(n=1):
        sys.stdout.write('----------------------- Preferential Attachment -----------------------\n')
        sys.stdout.flush()

        n = 1000
        n0 = 2
        d0 = 1
        m = 5

        np.random.seed(0)
        (G_0, _) = make_erdos_renyi(n0, d0, weighted = False)
        (G_pa, s_pa) = make_pref_attach(n, G_0, m = m, weighted = False)
        G_new = G_pa.copy()
        s_new = related_opinion_graph(s_pa, n)

        df = test_heuristics_two_graphs(FUNS, G_pa, G2=G_new, 
                                        s1=s_pa, s2=s_new)

        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/pa/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/pa_rel_{n}_{current_time}.csv')

def test_relatedness(G1, 
                     s1,
                     dir_name:str = f'data/out/raw/related/', 
                     related_vals:list = RELATED_VALS,
                     k = None):
        """General function to test relatedness for a given graph G1 and opinion s1.
        Varies the relatedness value and saves the results in a csv file.

        Args:
                G1 (nx.Graph): graph of primary opinion
                s1 (np.array): Opinion vector corresponding to G1
                dir_name (str, optional): Directory to save csv files. Should include graph type!! Defaults to f'data/out/raw/related/'.
                related_vals (list, optional): List of relatedness values to test. Defaults to RELATED_VALS.
                k (int, optional): Planner's budget. If None, default is half the number of nodes in G1.
        
        """
        if k is None:
                k = int(0.5*len(G1.nodes()))
        print(f'Graph current num edges: {len(G1.edges())}')
        print(f'Planner budget: {k}')
        G_new = G1.copy()

        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')
        os.makedirs(dir_name, exist_ok=True)

        # Calculate polarization for each relatedness value
        for val in related_vals:
                print(f'==== Related: {val} ====')
                # second arg is noise, thus do 1 - val to get noise
                s_new = related_opinion_graph(s1, 1 - val) 

                df = test_heuristics_set_k(FUNS, 
                                           G1, 
                                           G2=G_new, 
                                           s1=s1, 
                                           s2=s_new,
                                           k=k)
                
                df.to_csv(f'{dir_name}/r{val}_k{k}_{current_date}_{current_time}.csv')
                print(f'Saved to {dir_name}/r{val}_k{k}_{current_date}_{current_time}.csv')

def test_relatedness_twitter(dir_name:str = f'data/out/raw/tw/related/',
                             related_vals:list = RELATED_VALS,
                             k = None):
        sys.stdout.write('----------------------- Twitter and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        (_, s, _, G, _) = load_twitter()
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

def test_relatedness_reddit(dir_name:str = f'data/out/raw/rd/related/', 
                            related_vals:list = RELATED_VALS,
                            k = None):
        sys.stdout.write('----------------------- Reddit and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        (_, s, _, G, _) = load_reddit()
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

def test_relatedness_blogs(dir_name:str = f'data/out/raw/bg/related/', 
                           related_vals:list = RELATED_VALS,
                           k = None):
        sys.stdout.write('----------------------- Blogs and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        (_, s, _, G, _) = load_blogs()
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

def test_relatedness_er(dir_name:str = f'data/out/raw/er/related/', 
                        related_vals:list = RELATED_VALS,
                        k = None):
        sys.stdout.write('----------------------- ER and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p = 0.02

        np.random.seed(0)
        (G, s) = make_erdos_renyi(n, p, weighted = False)
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

def test_relatedness_sbm(dir_name:str = f'data/out/raw/sbm/related/', 
                        related_vals:list = RELATED_VALS,
                        k = None):
        sys.stdout.write('----------------------- SBM and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p1 = 0.05
        p2 = 0.005
        a = 5
        b = 1

        np.random.seed(0)
        (_, _, G, s) = make_block(n, p1, p2, a, b, weighted = False)
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

def test_relatedness_pa(dir_name:str = f'data/out/raw/pa/related/',
                        related_vals:list = RELATED_VALS,
                        k = None):
        sys.stdout.write('----------------------- PA and Vary Relatedness -----------------------\n')
        sys.stdout.flush()

        n = 1000
        n0 = 2
        d0 = 1
        m = 5

        np.random.seed(0)
        (G_0, _) = make_erdos_renyi(n0, d0, weighted = False)
        (G, s) = make_pref_attach(n, G_0, m = m, weighted = False)
        test_relatedness(G, s, dir_name=dir_name, related_vals=related_vals, k=k)

#'''
##### Reddit Network #####
def reddit():
        sys.stdout.write('----------------------- Reddit -----------------------\n')
        sys.stdout.flush()

        (n_rd, s_rd, A_rd, G_rd, L_rd) = load_reddit()

        df = test_heuristics(FUNS, G_rd, s_rd, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/rd.csv')    

##### Twitter Network #####
def twitter():
        sys.stdout.write('----------------------- Twitter -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_twitter()

        df = test_heuristics(FUNS, G_tw, s_tw, parallel = True, n_cores = n_cores)
        
        df.to_csv('data/out/raw/tw.csv')

##### Blogs Network #####
def blogs():
        sys.stdout.write('----------------------- Blogs -----------------------\n')
        sys.stdout.flush()

        (n_bg, s_bg, A_bg, G_bg, L_bg) = load_blogs()

        df = test_heuristics(FUNS, G_bg, s_bg, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/bg.csv')

##### Erdos-Renyi Network #####
def erdos_renyi():
        sys.stdout.write('----------------------- Erdos-Renyi -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p = 0.02

        np.random.seed(0)
        (G_er, s_er) = make_erdos_renyi(n, p, weighted = False)

        df = test_heuristics(FUNS, G_er, s_er, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/er.csv')

##### Stockastic Block Model Network #####
def sbm():
        sys.stdout.write('----------------------- Stochastic Block Model -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p1 = 0.05
        p2 = 0.005
        a = 5
        b = 1

        np.random.seed(0)
        (c1, c2, G_sbm, s_sbm) = make_block(n, p1, p2, a, b, weighted = False)

        df = test_heuristics(FUNS, G_sbm, s_sbm, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/sbm.csv')

##### Preferential Attachment Network #####
def pref_attach():
        sys.stdout.write('----------------------- Preferential Attachment -----------------------\n')
        sys.stdout.flush()

        n = 1000
        n0 = 2
        d0 = 1
        m = 5

        np.random.seed(0)
        (G_0, _) = make_erdos_renyi(n0, d0, weighted = False)
        (G_pa, s_pa) = make_pref_attach(n, G_0, m = m, weighted = False)

        df = test_heuristics(FUNS, G_pa, s_pa, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/pa.csv')
#'''


if __name__ == "__main__":
        parser = argparse.ArgumentParser(description='Run simulations')
        parser.add_argument('--save_dir', required=False, type=str, help='Directory to save csv outputs')
        parser.add_argument('--k', required=False, type=int, help='Planner\'s budget')

        args = parser.parse_args()

        # Test relatedness with defaults
        # test_relatedness_reddit()
        test_relatedness_twitter()
        test_relatedness_blogs()


        # twitter_random(1)
        # er_random()
        # blogs_random()

        # Example usuage:
        # python3 sim.py --save_dir data/out/raw/rd/related17Jul --k 100 

