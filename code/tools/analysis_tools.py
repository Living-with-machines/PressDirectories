from pathlib import Path
import pandas as pd
from tools.helpers import clean_corrected_csv



def load_all_csv(path='./csv_dump_final/'):
    paths = list(Path(path).glob('**/*.csv'))

    dfs = []
    for p in paths:
        df = pd.read_csv(p,index_col=0)
        df['YEAR'] = int(p.stem.split('_')[1])
        dfs.append(df)
    
    return pd.concat(dfs,axis=0)

