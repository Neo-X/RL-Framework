

"""
sudo pip3 install pandas
sudo pip3 install seaborn
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set(font_scale=1.5)
import pdb

colors = {'SMiRL (ours)': 'k',
          'SMiRL VAE (ours)': 'purple',
          'ICM': 'b',
          'RND': 'orange',
          'Oracle': 'g',
          'MAHRL(Hetero)' : 'b',
          'MAHRL(Homo)' : 'b',
          'MADDPG': 'r',
          'PPO': 'r',
         }
linestyle = {'SMiRL (ours)': '-',
          'ICM': '-',
          'RND': '--',
          'Oracle': '--',
          'SMiRL VAE (ours)': '--',
          'MAHRL(Hetero)' : '-',
          'MAHRL(Homo)' : '--',
          'MADDPG': '-.',
          'PPO': '--',
         }
def plotsns_smoothed(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    s = s.split('/')[-1]
    data = pd.DataFrame([(i//res*res, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False,  c=colors[label])

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

def plotsns(ax, s, df, label, title=None, ylabel=None, res=1):
    data = list(df[s])
    data = (np.cumsum(data)[res:]-np.cumsum(data)[:-res]) / res
    s = s.split('/')[-1]
    data = pd.DataFrame([(i, data[i]) for i in range(len(data))])
    data = data.rename(columns={1: s, 0: 'Episodes'})
    ax = sns.lineplot(x='Episodes', y=s, data=data, label=label, ax=ax, legend=False, c=colors[label])
    #print(ax.lines)
    ax.lines[-1].set_linestyle(linestyle[label])
    #print(\label\, label,linestyle[label] )
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(s)

    if ylabel is not None:
        ax.set_ylabel(ylabel)
        
    ax.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    #ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
def save(fname):
    plt.show()
    '''
    plt.savefig('{}.png'.format(fname))
    plt.clf()
    '''
        
if __name__ == '__main__':

    fig, (ax3) = plt.subplots(1, 1, figsize=(15,8))
    #*******************************************************************************
    #####################
    ##### Stability #####
    #####################
    #*******************************************************************************
    
    datadir = './safe2.csv'
    df = pd.read_csv(datadir)
    #ax3.set_title(' Mewan reward / Steps')
    
    
    # #####################
    # ##### w/ ICM ######
    # #####################
    
    biped_falls = []
    res = 5
    
    bf = list(df.iloc[92][1:])
    bf =  np.array([float(x) for x in bf])
    bf = (np.cumsum(bf)[res:] - np.cumsum(bf)[:-res])/res
    time = 32000
    for val in bf:
        biped_falls.append((time, float(val)))
        time = 32000

    
    bf = pd.DataFrame(biped_falls)
    bf = bf.rename(columns={1: 'Biped Falls', 0: 'Steps'})
    label='MAHRL(Hetero)'
    sns.boxplot(data=bf, x='Steps', y='Biped Falls', palette="Blues")
    ax3.lines[-1].set_linestyle(linestyle[label])
    

    #####################
    ###### Vanilla ######
    #####################
    
    biped_falls = []
    res = 5
    
    bf = list(df.iloc[94][1:])
    bf =   np.array([float(x) for x in bf])
    bf = (np.cumsum(bf)[res:] - np.cumsum(bf)[:-res])/res
    time = 16000
    for val in bf:
        biped_falls.append((time, float(val)))
        time = 16000
    
    bf = pd.DataFrame(biped_falls)
    bf = bf.rename(columns={1: 'Biped Falls', 0: 'Steps'})
    
    # Move colors (true reward should be 5)
    # sns.lineplot(data=[], x=None, y=None, ax=ax3)
    # sns.lineplot(data=[], x=None, y=None, ax=ax3)
    # sns.lineplot(data=[], x=None, y=None, ax=ax3)
    label='PPO'
    sns.boxplot(data=bf, x='Steps', y='Biped Falls')
    ax3.lines[-1].set_linestyle(linestyle[label])
    
    #ax3.ticklabel_format(axis= 'x', style='sci', scilimits=(0,3))
    
    
    ax3.set(ylabel=' Mean Reward ')
    ax3.set(xlabel=' Env Steps ' )
    
    h,l = ax3.get_legend_handles_labels()
    ax3.legend(h[:4],l[:4], bbox_to_anchor=(0.65, 0.5), loc=2)
    
    #ax3.legend()
    '''
    handles, labels = ax2.get_legend_handles_labels()
    plt.figlegend(handles, labels, ncol=5, mode='expand', bbox_to_anchor=(.37,.03,.3,.1))
    '''
    #plt.subplots_adjust(bottom=.25, wspace=.25)
    plt.show()
    fig.savefig("file2"+".svg")
    fig.savefig("file2"+".png")


