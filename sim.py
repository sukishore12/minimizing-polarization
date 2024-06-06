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

funs = ['opt_random_add', 'opt_max_fiedler_diff']

funs2 = ['opt_max_common_ground', 'opt_max_2grad']


##### Two Opinions #####
def twitter_random(n=3, threshold=None):
        sys.stdout.write('---------------- Twitter and Random -----------------------\n')
        sys.stdout.write(f'-------------------- n = {n} -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_twitter()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n * 0.1)
        # s_new = random_opinion_graph(s_tw)

        df = test_heuristics_two_graphs(funs2, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new,
                                        threshold=threshold)
        
        df2 = test_heuristics(funs, G_tw, s=s_tw, parallel = True, n_cores = n_cores)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/tw/cg/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/tw_rel_{n}_{current_time}.csv')
        df2.to_csv(f'{dir_name}/all_metrics_tw_rel_{n}_{current_time}.csv')


##### Two Opinions #####
def reddit_random(n=3, threshold=None):
        sys.stdout.write('----------------------- Reddit and Random -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_reddit()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n * 0.1)
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



def blogs_random(n=3, threshold=None):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        (n_tw, s_tw, A_tw, G_tw, L_tw) = load_blogs()
        G_new = G_tw.copy()
        s_new = related_opinion_graph(s_tw, n * 0.1)
        # s_new = random_opinion_graph(s_tw)

        df = test_heuristics_two_graphs(funs2, G_tw, G2=G_new, 
                                        s1=s_tw, s2=s_new,
                                        threshold=threshold)
        
        df2 = test_heuristics(funs, G_tw, s=s_tw, parallel = True, n_cores = n_cores)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/bl/cg/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/bl_rel_{n}_{current_time}.csv')
        df2.to_csv(f'{dir_name}/all_metrics_bl_rel_{n}_{current_time}.csv')


def er_random(n=3, threshold=None):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p = 0.02

        np.random.seed(0)
        (G_er, s_er) = make_erdos_renyi(n, p, weighted = False)

        np.random.seed(42)
        (G_new, s_new) = make_erdos_renyi(n, p, weighted = False)

        df = test_heuristics_two_graphs(funs2, G_er, G2=G_new, 
                                        s1=s_er, s2=s_new,
                                        threshold=threshold)
        
        df2 = test_heuristics(funs, G_er, s=s_er, parallel = True, n_cores = n_cores)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/er/cg/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/er_rel_{n}_{current_time}.csv')
        df2.to_csv(f'{dir_name}/all_metrics_er_rel_{n}_{current_time}.csv')


def sbm_random(n=3, threshold=None):
        sys.stdout.write('----------------------- Blogs and Random -----------------------\n')
        sys.stdout.flush()

        n = 1000
        p1 = 0.05
        p2 = 0.005
        a = 5
        b = 1

        np.random.seed(0)
        (c1, c2, G_sbm, s_sbm) = make_block(n, p1, p2, a, b, weighted = False)

        np.random.seed(42)
        (c1, c2, G_new, s_new) = make_block(n, p1, p2, a, b, weighted = False)

        df = test_heuristics_two_graphs(funs2, G_sbm, G2=G_new, 
                                        s1=s_sbm, s2=s_new,
                                        threshold=threshold)
        
        df2 = test_heuristics(funs, G_sbm, s=s_sbm, parallel = True, n_cores = n_cores)
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        dir_name = f'data/out/raw/sbm/cg/{current_date}'  
        os.makedirs(dir_name, exist_ok=True)
        df.to_csv(f'{dir_name}/sbm_rel_{n}_{current_time}.csv')
        df2.to_csv(f'{dir_name}/all_metrics_sbm_rel_{n}_{current_time}.csv')

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
        # reddit_random(2)
        # reddit_random(4)
        # reddit_random(6)
        # reddit_random(8)
        # twitter_random(2)
        # twitter_random(4)
        # twitter_random(6)
        # twitter_random(8)
        # twitter_random(10)
        # reddit_random(10)
        # blogs_random(10)
        er_random(10)


