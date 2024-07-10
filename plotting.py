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
LEGEND = {'opt_max_fiedler_diff': 'FD', 'opt_random_add': 'Random',
          'opt_max_common_ground': 'CG', 'opt_max_2grad': 'CGGD'}

NAMES = {'rd': 'Reddit', 'tw': 'Twitter', 'bg': 'Political Blogs',
         'er': 'Erdös-Rényi', 'sbm': 'Stochastic Block',
         'pa': 'Preferential Attachment'}

plt.rcParams.update({'font.size': 15, 'axes.linewidth': 1.5 })

def plot_common_ground(names,
                       file_path, 
                       pol_weights=[0.7, 0.3],
                       log=False):
                       
    for name in names.keys():
        print('\n################\n'+name)
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
        axes[0].legend(loc='lower left')
        axes[0].set_ylabel('Polarization, $P(\mathbf{z}\')$')
        axes[0].set_xlabel('Planner\'s Budget, $k$')

        axes[1].set_xlabel('Planner\'s Budget, $k$')
        axes[1].set_ylabel('Weighted Polarization, $0.7P(\mathbf{z_1}) + 0.3P(\mathbf{z_1})$')

        axes[1].tick_params(direction='in', width=1.5)
        axes[1].legend(loc='lower left')
        plt.title('Performance of Common Ground Maximizing Heuristics',
                position = (0.5,0.9))
        
        current_date = datetime.now().strftime('%Y-%m-%d')
        current_time = datetime.now().strftime('%H-%M-%S')

        os.makedirs(f'fig/{current_date}/', exist_ok=True)
        plt.savefig(f'fig/{current_date}/{current_time}_{name}.pdf')


if __name__ == "__main__":
    names_tw = {'tw_rel': 'Twitter and Related Opinion'} # for testing
    file_path = f'data/out/raw/tw/2024-07-09/tw_1_16-51-56.csv'
    plot_common_ground(names_tw, file_path)

