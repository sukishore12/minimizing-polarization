import sys
import matplotlib
import matplotlib.pyplot as plt
import scipy.io
import itertools
import seaborn as sns
import os

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
LEGEND = {'opt_max_dis': 'DS', 'opt_max_grad': 'CD', 
          'opt_max_fiedler_diff': 'FD', 'opt_random_add': 'Random'}

NAMES = {'rd': 'Reddit', 'tw': 'Twitter', 'bg': 'Political Blogs',
         'er': 'Erdös-Rényi', 'sbm': 'Stochastic Block',
         'pa': 'Preferential Attachment'}

plt.rcParams.update({'font.size': 15, 'axes.linewidth': 1.5 })

'''
Budget and Polarization
'''
def budget_and_pol(names, legend, linestyles, 
                   save = True, innate=False, log=False):
    for name in names.keys():
    # for name in names.keys()[0]:
        print('\n################\n'+name)
        data = pd.read_csv('data/out/raw/'+name+'.csv', index_col = 0)

        df = process_df_cols(data, ['pol_vec', 'pol_dis_vec', 's'])

        # Remove column of poor-performing optimization heuristic
        df = df.loc[df.type != 'opt_max_grad_dis_ratio']

        # f,ax = plt.subplots(figsize = (8,6))
        f, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        
        if log:
            # plt.yscale('log')
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')

        # K_n is ER graph 
        (K_n,_) = make_erdos_renyi(len(df.s[0]), 1, weighted = False)
        s = np.transpose(np.array([df.s[0]])) - np.mean(df.s[0])
        
        pol_K_n = get_measure(K_n,s,'pol')
        dis_K_n = get_measure(K_n,s,'dis')
        pol_dis_K_n = get_measure(K_n,s,'pol_dis')

        axes[0].hlines(y=pol_K_n, xmin = 0, xmax = len(df.pol_vec[0]),
                linestyle='-', linewidth = 5, color = 'k',
                label = 'Best, $K_n$')

        # ax1.hlines(y=pol_K_n, xmin = 0, xmax = len(df.pol_vec[0]),
        #           linestyle='-', linewidth = 5, color = 'k',
        #           label = 'Best, $K_n$')
        
        axes[1].hlines(y=pol_dis_K_n, xmin = 0, xmax = len(df.pol_vec[0]),
                linestyle='-', linewidth = 5, color = 'k',
                label = 'Best, $K_n$')
        
        if innate:
            pol_K_0 = np.var(df.s.iloc[0])*len(df.s.iloc[0])

            axes[0].hlines(y=pol_K_0, xmin = 0, xmax = len(df.pol_vec[0]),
                    linestyle=':', linewidth = 3, color = 'g',
                    label = 'Innate Polarization')

            # ax1.hlines(y=pol_K_0, xmin = 0, xmax = len(df.pol_vec[0]),
            #           linestyle=':', linewidth = 3, color = 'g',
            #           label = 'Innate Polarization')
        
        print('initial', df.pol_vec.iloc[0][0])
        print('best', pol_K_n)
        
        if innate:
            print('worst', pol_K_0)
            
        for i in range(len(df)):
            # graph business
            print(df.type.iloc[i], df.pol_vec.iloc[i][len(df.pol_vec.iloc[i])-1])
            # plt.plot(pol_dis_K_n[i], linestyle = linestyles[i],
            #     label = legend[df.type.iloc[i]], linewidth = 3)
            # Previous plotting
            axes[0].plot(df.pol_vec.iloc[i], linestyle = linestyles[i],
                    label = legend[df.type.iloc[i]], linewidth = 3)
            axes[1].plot(df.pol_dis_vec.iloc[i], linestyle = linestyles[i],
                    label = legend[df.type.iloc[i]], linewidth = 3)
    #        plt.plot(np.array(df.pol_vec.iloc[i])/df.pol_vec.iloc[i][0],
    #                 label = legend[df.type.iloc[i]], linewidth = 3)

        axes[0].tick_params(direction='in', width=1.5)
        axes[0].set_title(f'{names[name]}')
        axes[0].legend(loc='lower left')
        axes[0].set_ylabel('Resulting Polarization, $P(\mathbf{z}\')$')
        axes[0].set_xlabel('Planner\'s Budget, $k$')

        axes[1].set_xlabel('Planner\'s Budget, $k$')
        axes[1].set_ylabel('Resulting Disagreement+Polarization, $P(\mathbf{z}\')$')

        axes[1].tick_params(direction='in', width=1.5)
        axes[1].legend(loc='lower left')
    #    plt.ylabel(r'Fraction of Remaining Polarization, $\frac{P(\mathbf{z}\')}{P(\mathbf{z})}$')
    #    plt.title('Performance of Polarization-Minimizing Heuristics for '+names[name]+ ' Network',
    #             position = (0.5,0.9))
        
        if save:
            os.makedirs("fig/pol_dis/", exist_ok=True)
            plt.savefig('fig/pol_dis/'+name+'_pol.pdf')

def graph_node_distance(names, 
                        legend, 
                        linestyles,
                        save=True):
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

if __name__ == "__main__":
    names = {'rd': 'Reddit'} # for testing
    # graph_node_distance(NAMES, LEGEND, LINESTYLES)
    names = {'tw_rand': 'Twitter and Random Opinion'} # for testing
    legend = {'opt_max_dis': 'DS of G2'}
    budget_and_pol(names, legend, LINESTYLES)