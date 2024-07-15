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
RELATED_VALS = [0.2, 0.8, 1]

'''
Setup
'''
LEGEND = {'opt_max_fiedler_diff': 'FD', 'opt_random_add': 'Random',
          'opt_max_common_ground': 'CG', 'opt_max_2grad': 'CGGD'}

NAMES = {'rd': 'Reddit', 'tw': 'Twitter', 'bg': 'Political Blogs',
         'er': 'Erdös-Rényi', 'sbm': 'Stochastic Block',
         'pa': 'Preferential Attachment'}

plt.rcParams.update({'font.size': 15, 'axes.linewidth': 1.5 })

def plot_common_ground(names,
                       file_paths, 
                       pol_weights=[0.7, 0.3],
                       log=False):
                       
    for idx, name in enumerate(names.keys()):
        print('\n################\n'+name)
        print(f'{idx}: {file_paths}')
        file_path = file_paths[idx]
        data = pd.read_csv(file_path, index_col = 0)
        df = process_df_cols(data, ['pol1_vec', 'pol2_vec'])

        f, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
        
        if log:
            # plt.yscale('log')
            axes[0].set_yscale('log')
            axes[1].set_yscale('log')
            
        for i in range(len(df)):
            axes[0].plot(df.pol1_vec.iloc[i], 
                         linestyle = LINESTYLES[i],
                         label = LEGEND[df.type.iloc[i]], linewidth = 3)
            
            weighted_pol1 = [pol_weights[0] * val for val in df.pol1_vec.iloc[i]]
            weighted_pol2 = [pol_weights[1] * val for val in df.pol2_vec.iloc[i]]
            weighted_pol = [val1 + val2 for val1, val2 in zip(weighted_pol1, weighted_pol2)]
            axes[1].plot(weighted_pol, 
                         linestyle = LINESTYLES[i],
                         label = LEGEND[df.type.iloc[i]], linewidth = 3)

        axes[0].tick_params(direction='in', width=1.5)
        axes[0].set_title(f'{names[name]}')
        # axes[0].legend()
        axes[0].set_ylabel('Polarization, $P(\mathbf{z}\')$')
        axes[0].set_xlabel('Planner\'s Budget, $k$')

        axes[1].set_xlabel('Planner\'s Budget, $k$')
        axes[1].set_ylabel('Weighted Polarization, $0.7P(\mathbf{z_1}) + 0.3P(\mathbf{z_1})$')

        axes[1].tick_params(direction='in', width=1.5)
        axes[1].legend(loc='upper right', bbox_to_anchor=(1.35, 1))
        plt.tight_layout()
        plt.title('Performance of Common Ground Maximizing Heuristics',
                position = (0.5,0.9))
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        os.makedirs(f'fig/{current_date}/', exist_ok=True)
        plt.savefig(f'fig/{current_date}/{current_time}_{name}.pdf')

def plot_set_k(relatedness,file_paths,
               k = 1, 
               pol_weights=[0.7, 0.3],
               log=False):
    files = os.listdir(file_paths)                   
    for idx, file in enumerate(files):
        file = os.path.join(file_dir, file)
        parts = file.split('rd_')

        # Further split the resulting part to isolate the number
        related_val = parts[1].split('_')[0]
        print(f'Relatedness: {related_val}')

        data = pd.read_csv(file, index_col = 0)
        df = process_df_cols(data, ['pol1_vec', 'pol2_vec'])
        print(len(df))
        print(df.pol1_vec.iloc[0][-1])

        f, axes = plt.subplots(1, 1, figsize=(16, 6), sharey=False)
        
        if log:
            # plt.yscale('log')
            axes.set_yscale('log')
            # axes[1].set_yscale('log')
            
        x_values = list(range(len(df.pol1_vec.iloc[0])))  # Example x-axis values
        print(f'x_values: {x_values}')
        
        for i in range(len(df)):
            y_values = df.pol1_vec.iloc[i]
            print(f'(x, y): ({x_values}, {y_values})')
            axes.plot(x_values, y_values,
                      linestyle=LINESTYLES[i % len(LINESTYLES)],
                      label=LEGEND.get(df.type.iloc[i], f'Type {i}'), linewidth=3)
        
        
        axes.set_title(f'Polarization vs Relatedness for Planner\'s Budget, {k}')
        axes.legend()
        axes.set_ylabel('Polarization, $P(\mathbf{z}\')$')
        axes.set_xlabel('Planner\'s Budget, $k$')
        plt.tight_layout()
        plt.title('Performance of Common Ground Maximizing Heuristics',
                position = (0.5,0.9))
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        os.makedirs(f'fig/{current_date}/', exist_ok=True)
        plt.savefig(f'fig/{current_date}/{current_time}.pdf')


if __name__ == "__main__":
    # names = {'rd': 'Reddit Graph', 'er': f"Erdos-Renyi Graph, n = 1000, p = 0.02"} # for testing
    # file_paths = [f'data/out/raw/er/2024-07-11/er_1_18-13-56.csv', 'data/out/raw/rd/2024-07-11/rd_1_18-50-13.csv']
    # plot_common_ground(names, file_paths)
    file_dir = 'data/out/raw/rd/related2'
    plot_set_k(RELATED_VALS, file_dir)

