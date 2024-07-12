import sys
import os
from datetime import datetime
import numpy as np
import networkx as nx

from utils import *
from two_graph_utils import *

# number of cores to use for parallelization 
# NOTE: parallel computation is only used for finding 
# the optimal edge at every step
n_cores = -1 


# funs = ['opt_random_add', 'opt_max_dis', 'opt_max_fiedler_diff',
#         'opt_max_grad']

FUNS = ['opt_random_add', 'opt_max_fiedler_diff', 
        'opt_max_common_ground', 'opt_max_2grad']


##### Two Opinions #####
def twitter_random(n=1, threshold=None):
        """n = 1, random second opinion graph"""
        sys.stdout.write('---------------- Twitter and Random -----------------------\n')
        sys.stdout.write(f'-------------------- n = {n} -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_twitter()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n)

        df = test_heuristics_two_graphs(FUNS, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new,
                                        threshold=threshold)
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/tw/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/tw_{n}_{current_time}.csv')

##### Two Opinions #####
def reddit_random(n=1, threshold=None):
        
        sys.stdout.write('----------------------- Reddit and Random -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_reddit()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, 1)
        # s_new = random_opinion_graph(s_tw)

        df = test_heuristics_two_graphs(funs2, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new,
                                        threshold=threshold)
        
        df2 = test_heuristics(funs, G_tw, s=s_tw, parallel = True, n_cores = n_cores)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/rd/cg/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/rd_rel_{n}_{current_time}.csv')
        df2.to_csv(f'{dir_name}/all_metrics_rd_rel_{n}_{current_time}.csv')


def blogs_random(n=1, threshold=None):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_blogs()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n)

        df = test_heuristics_two_graphs(FUNS, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new,
                                        threshold=threshold)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/bl/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/bl_rel_{n}_{current_time}.csv')


def er_random(n=1, threshold=None):
        sys.stdout.write('----------------------- ER Graphs -----------------------\n')
        sys.stdout.flush()

        num_verts = 1000
        p = 0.02

        np.random.seed(0)
        (G_er, s_er) = make_erdos_renyi(num_verts, p, weighted = False)
        G_new = G_er.copy()
        s_new = related_opinion_graph(s_er, n)

        df = test_heuristics_two_graphs(FUNS, G_er, G2=G_new, 
                                        s1=s_er, s2=s_new,
                                        threshold=threshold)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/er/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/er_{n}_{current_time}.csv')


def sbm_random(n=1, threshold=None):
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
                                        s1=s_sbm, s2=s_new,
                                        threshold=threshold)
        
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/sbm/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/sbm_rel_{n}_{current_time}.csv')

def pa_random(n=1, threshold=None):
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
                                        s1=s_pa, s2=s_new,
                                        threshold=threshold)

        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/pa/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/pa_rel_{n}_{current_time}.csv')

#'''
##### Reddit Network #####
def reddit():
        sys.stdout.write('----------------------- Reddit -----------------------\n')
        sys.stdout.flush()

        (n_rd, s_rd, A_rd, G_rd, L_rd) = load_reddit()

        df = test_heuristics(funs, G_rd, s_rd, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/rd.csv')    

##### Twitter Network #####
def twitter():
        sys.stdout.write('----------------------- Twitter -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_twitter()

        df = test_heuristics(funs, G_tw, s_tw, parallel = True, n_cores = n_cores)
        
        df.to_csv('data/out/raw/tw.csv')

##### Blogs Network #####
def blogs():
        sys.stdout.write('----------------------- Blogs -----------------------\n')
        sys.stdout.flush()

        (n_bg, s_bg, A_bg, G_bg, L_bg) = load_blogs()

        df = test_heuristics(funs, G_bg, s_bg, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/bg.csv')

##### Erdos-Renyi Network #####
def erdos_renyi():
        sys.stdout.write('----------------------- Erdos-Renyi -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p = 0.02

        np.random.seed(0)
        (G_er, s_er) = make_erdos_renyi(n, p, weighted = False)

        df = test_heuristics(funs, G_er, s_er, parallel = True, n_cores = n_cores)
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

        df = test_heuristics(funs, G_sbm, s_sbm, parallel = True, n_cores = n_cores)
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

        df = test_heuristics(funs, G_pa, s_pa, parallel = True, n_cores = n_cores)
        df.to_csv('data/out/raw/pa.csv')
#'''


if __name__ == "__main__":
        # twitter_random(1)
        # er_random()
        blogs_random()

