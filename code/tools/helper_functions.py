from collections import defaultdict
#from os import posix_fadvise
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cosine
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import math
from tqdm.notebook import tqdm

def complete_records(df: pd.DataFrame, target_col:str) -> pd.DataFrame:
    """function to add information to the press directories detaframe. we use the 
    NEWSPAPER_ID column to add missing information, for example if, for one year, a 
    newspaper doesn't have a POLITICS we insert one that is closest in time from a row with
    the same NEWSPAPER_ID"""
    df = df.copy()
    df[f'value_{target_col}_source_idx'] = None
    df_chain = df[~df.NEWSPAPER_ID.isnull()]
    chains = set(df_chain.NEWSPAPER_ID.unique())
    
    for chain_id in chains:
        # get elements for a specific chain_id
        chain_df = df[df.NEWSPAPER_ID==chain_id]
        # find those the have NaA values
        no_attr = list(np.where(chain_df[target_col].isnull())[0])

        # if there are empty but not all values are empty
        if no_attr and not (len(no_attr) == chain_df.shape[0]): 
            # look which cells have a value for this columns       
            has_attr = np.where(~chain_df[target_col].isnull())[0]
            # find indices for cells (that have content) closest to an empty cell
            replace_with = chain_df.iloc[[has_attr[np.argmin(abs(has_attr - i))] for i in no_attr]][target_col]
            zipped = list(zip(chain_df.iloc[no_attr].index,replace_with.index,replace_with.values))
            for target_idx, source_idx, cat in zipped:
                df.loc[target_idx,target_col] = cat
                df.loc[target_idx,f'value_{target_col}_source_idx'] = source_idx

    return df


def plot_comparison_selected_categories(selected,df,target_col,path):
    
    fig, ax = plt.subplots(figsize=(8,6))
    maximums = []
    for label,color in selected:
        prop_all = df.groupby([target_col,'YEAR'])['id'].count() / df.groupby(['YEAR'])['id'].count()
        
        prop_jisc_all = df[df['IN_JISC'] > 0].groupby([target_col,'YEAR'])['id'].count() / df[df['IN_JISC'] > 0].groupby(['YEAR'])['id'].count()
  
        if isinstance(label,str):
            prop = prop_all.loc[label,:].droplevel(target_col)
            prop_jisc = prop_jisc_all.loc[label,:].droplevel(target_col)
        elif isinstance(label,list):
            prop = prop_all.loc[label,:].groupby('YEAR').sum()
            prop_jisc = prop_jisc_all.loc[label,:].groupby('YEAR').sum()

        maximums.extend([prop_jisc.max(),prop.max()])
        
        ax.plot(prop.index, prop, color=color, marker="s", linestyle= '--',linewidth=2, markersize=8,alpha=0.75)
        ax.plot(prop_jisc.index, prop_jisc, color=color, marker="x", 
                                linestyle='-.', linewidth=2, markersize=8, alpha=0.75)
    ax.set_ylim(
            [0, np.max(maximums)+0.05]
        )
    #plt.savefig(figures_path/f'{selected[0][0]}_{selected[1][0]}_comparison.png',bbox_inches='tight')
    
    plt.savefig(path,bbox_inches='tight')
    plt.show()
    


def divergence(df,labels,measure='jensenshannon',target='POLITICS-2'):

    props_pop, props_sample = [], []

    for label in labels:
        prop_np_all = df.groupby([target,'YEAR'])['id'].count() / df.groupby(['YEAR'])['id'].count()
        
        prop_jisc_all = df[df['IN_JISC'] > 0].groupby([target,'YEAR'])['id'].count() / df[df['IN_JISC'] > 0].groupby(['YEAR'])['id'].count()

        if isinstance(label,str) or isinstance(label,float) :
            prop = prop_np_all.loc[label,:].droplevel(target)
            try:
                prop_jisc = prop_jisc_all.loc[label,:].droplevel(target)
            except:
                prop_jisc = pd.Series([.0]*len(df.YEAR.unique()),index=df.YEAR.unique())
            
         
        elif isinstance(label,list):
            prop = prop_np_all.loc[label,:].groupby('YEAR').sum()
            try:
                prop_jisc = prop_jisc_all.loc[label,:].groupby('YEAR').sum()
            except:
                prop_jisc = pd.Series([.0]*len(df.YEAR.unique()),index=df.YEAR.unique())
            
        props_pop.append(prop); props_sample.append(prop_jisc) 
    props_pop, props_sample = pd.concat(props_pop, axis=1).fillna(0),pd.concat(props_sample, axis=1).fillna(0)

    scores = {}
    for year in props_pop.index:
        exec(f'scores[year] = {measure}(props_pop.loc[year],props_sample.loc[year])')
    #return entropies
    return props_pop,props_sample,pd.DataFrame.from_dict(scores,orient='index')
    
#kl_p = lambda x: x[0] * np.log(2*x[0]/(x[0]+x[1]))

def kl_p(x):
    if x[0]+x[1] > .0:
        return x[0] * np.log(2*x[0]/(x[0]+x[1]))
    else:
        return .0

def js_p(x):
    m = (1/2) * (x[0] + x[1])
    if x[0] >.0:
        p = x[0]*np.log2(x[0])
    else:
        p = .0
    if x[1] > .0:
        q = x[1]*np.log2(x[1])
    else:
        q = .0
    return (-m * np.log2(m)) + (1/2) * (p + q)


def reduce_price(x,ceil=True):
    try:
        if float(x) <= 1.0:
            return float(x)
        else:
            if ceil:
                return float(math.ceil(x))
            else:
                return float(x)
    except Exception as e:
        #print(e)
        return None


