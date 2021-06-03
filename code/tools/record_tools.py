from collections import defaultdict
from os import posix_fadvise
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm.notebook import tqdm
### ------ Chaining code --------

def sort_pickled(pickled):
    """sort files with link predictions into ascending order
    """
    year2path = defaultdict(list)
    for p in pickled:
        year = int(p.stem.split("_")[1])
        # one year can be mapped to multiple paths
        # depending on how many "jumps" we have taken
        # during classification
        year2path[year].append(p)
    return [v[0] for k,v in sorted(year2path.items(),key=lambda x:x[0])]

class LinkChain(object):
    """basic chain object
    requires prediction in the following format
    {'source': {'idx': idx_t, 'year': year_t},
    'target': {'idx': idx_t+1, 'year': year_t+1}}
    starts with a root element, and adds subsequent ids
    """
    def __init__(self,prediction):

        self.root = prediction['source']['idx']
        self.chain = [self.root]
        #self.years = [int(self.root.split('_')[1])]
    
    def check_source(self,link_prediction):
        return link_prediction['source']['idx'] in self.chain
    
    def __len__(self):
        return len(self.chain)
        
    def add_object(self,link_prediction):
        self.chain.append(link_prediction['target']['idx'])
        #self.years.append(int(link_prediction['target']['idx'].split('_')[1]))

class LinkChainer(object):
    def __init__(self):
        self.chains = {}
        self.dataframe = None
        
    def initiate(self,first_predictions):
        """initiate new link chain with the items
        from the earliest/oldest predictions first
        """
        for p in first_predictions:
            # add not root 
            self.chains[p['source']['idx']] = LinkChain(p)
            # add new target for a given root
            self.chains[p['source']['idx']].add_object(p) 
        
    def add_objects(self,predictions):
        """after initiatializatin populate the link chaing object
        """
        # iterate over all predictions
        for p in predictions:
            # for new predictions, check if link can be
            # associated with an existing chaing or should
            # be added as a new root element
            new_root = True
            for idx,linkchain in self.chains.items():
                # if element can be associated with a existing chaing
                # add the target of the predictions
                if self.chains[linkchain.root].check_source(p):
                    self.chains[linkchain.root].add_object(p)
                    new_root = False
            # otherwise create new root
            if new_root:
                self.chains[p['source']['idx']] = LinkChain(p)
                self.chains[p['source']['idx']].add_object(p)
                
    def __sort__(self):
        return sorted(self.chains.items(),key =lambda x: len(x[1]), reverse=True)
        
    def __len__(self):
        return len(self.chains.keys())
        
    def __str__(self):
        return f'< LinkChainer Object with {self.__len__()} root elements. >'
    
    def load_dataframes(self,csv_dump):
        year2dataframes = {int(p.stem.split("_")[-1]) : pd.read_csv(p, index_col=0) for p in Path(csv_dump).glob("**/*.csv")}
        
        dataframes = []
        for y,d in sorted(year2dataframes.items(),key=lambda x:x[0]):
            d['year'] = y
            dataframes.append(d)
        
        self.dataframe = pd.concat(dataframes)
        self.dataframe.reset_index(inplace=True)

    def create_linked_dataset(self,csv_dump,out_folder):

        if self.dataframe is None:
            self.load_dataframes(csv_dump)

        self.dataframe['chain_id'] = ''

        for i,(idx,chain) in tqdm(enumerate(self.__sort__())):
            self.dataframe.loc[self.dataframe.id.isin(chain.chain),'chain_id'] = 'CID_' + str(i).zfill(6)
        
        min_year = self.dataframe.year.min()
        max_year = self.dataframe.year.max()

        out_folder.mkdir(exist_ok=True)
        
        self.dataframe.to_csv(out_folder / f'MPD_export_{min_year}_{max_year}.csv')
    
    
    def retrieve_by_chain_id(self,chain_id,csv_dump=Path('./csv_dump_final')):
        if self.dataframe is None:
            self.create_linked_dataset(csv_dump)
        
        return self.dataframe[self.dataframe.chain_id==chain_id]

    def print_chain(self,chain,csv_dump):
        if not hasattr(self,'dataframe'):
            self.load_dataframes(csv_dump)
        return self.dataframe.loc[self.dataframe.id.isin(chain)]

def insert_annotations(path_to_linked_csv,path_to_annotations,col):
    with open(path_to_annotations,'rb') as in_pickle:
        annotations = pickle.load(in_pickle)
    annotations = [w for w,l in annotations if l=='same']
    df = pd.read_csv(path_to_linked_csv,index_col=0)

    if not col in df:
        df[col] = ''


    for _,s_id,ndp_id,chain_id in annotations:
        df.loc[df.chain_id==chain_id,col] = s_id

    return df.to_csv(path_to_linked_csv)



def complete_records(df, target_col):
    
    df = df.copy()
    df[f'value_{target_col}_source_idx'] = None
    df_chain = df[~df.chain_id.isnull()]
    chains = set(df_chain.chain_id.unique())
    
    for chain_id in chains:
        # get elements for a specific chain_id
        chain_df = df[df.chain_id==chain_id]
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