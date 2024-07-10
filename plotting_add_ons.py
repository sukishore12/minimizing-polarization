import sys
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import itertools
import seaborn as sns
import os
from datetime import datetime

from utils import *

sns.set_theme(style="white", context="talk")
CUSTOM_PALETTE = [ "greyish", "amber", "windows blue", 
                  "faded green", "greenish", "dusty purple","black" ]
sns.set_palette(sns.xkcd_palette(CUSTOM_PALETTE))
CURRENT_PALETTE = sns.color_palette()
# sns.palplot(CURRENT_PALETTE) # show the color pallete
LINESTYLES = ['-', '--', '-.', ':']

'''
Setup
'''
legend = {'opt_max_dis': 'DS', 'opt_max_grad': 'CD', 
          'opt_max_fiedler_diff': 'FD', 'opt_random_add': 'Random',
          'common_ground': 'CG'}

NAMES = {'rd': 'Reddit', 'tw': 'Twitter', 'bg': 'Political Blogs',
         'er': 'Erdös-Rényi', 'sbm': 'Stochastic Block',
         'pa': 'Preferential Attachment'}

plt.rcParams.update({'font.size': 15, 'axes.linewidth': 1.5 })

def graph_node_distance(names, 
                        legend, 
                        linestyles,
                        save=True):
    """
    Plot histogram of node distance distribution before addition of new edges.
    """
    for name in names.keys():
        print('\n################\n'+name)
        data = pd.read_csv('data/out/raw/'+name+'.csv', index_col = 0)
        df = process_df_cols(data, ['s', 'G_in', 'G_out'])

        f, axes = plt.subplots(1, 4, figsize=(16, 6))
        f.subplots_adjust(hspace=0)
        custom_xlim = (1, 7)
        if name == 'er':
            custom_ylim = (0, 500) # er
        elif name == 'tw':
            custom_ylim = (0, 200) # tw
        elif name == 'rd':
            custom_ylim = (0, 250) 
        elif name == 'bg': 
            custom_ylim = (0, 400)
        elif name == 'pa': 
            custom_ylim = (0, 400)
        elif name == 'sbm': 
            custom_ylim = (0, 500)

        # Setting the values for all axes.
        plt.setp(axes, xlim=custom_xlim, ylim=custom_ylim)

        # Initial graph
        G_prev = nx.from_numpy_array(np.array(df.G_in.iloc[0]))
        distances = []
        # Process each updated graph
        for i in range(len(df)):
            G_in = nx.from_numpy_array(np.array(df.G_in.iloc[i]))
            G_out = nx.from_numpy_array(np.array(df.G_out.iloc[i]))
            difference = nx.difference(G_out, G_in)
            diff_edges = difference.edges
            graph_dist = []
            for edge in diff_edges:
                node1, node2 = edge
                distance = nx.shortest_path_length(G_in, node1, node2)
                graph_dist.append(distance)

            # Plot a histogram of the graph_dist values
            axes[i].hist(graph_dist, color='skyblue', edgecolor='black', 
                         bins=[1, 2, 3, 4, 5, 6, 7])
            axes[i].set_title(legend[df.type.iloc[i]])
            axes[0].set_xlabel('Distance')
            axes[0].set_ylabel('Frequency')

        f.suptitle(f'{names[name]}')
        f.savefig(f'fig/{name}_pre_dist_{name}.png')