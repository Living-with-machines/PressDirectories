import random
import numpy as np
import pandas as pd
import re
from collections import defaultdict
from tqdm.notebook import tqdm
from tools.book_tools import NPD
from tools.annotation_env import AnnotationEnv
from tools.linking_tools import RecordLinker, Annotator

class Collection(object):
    """Allows to manipulate Book or NPD objects at the collection level
    """
    def __init__(self,editions,in_path,out_path):
        self.editions = editions
        self.in_path = in_path
        self.out_path = out_path
        
    def extract_structure(self,model_structure,assume_london=True,override=True,):
        for year, pages in self.editions.items():
            #for idx,page_range in enumerate(page_ranges):
            print(f'Processing {year}.')
            npd_anno = AnnotationEnv(year,self.in_path,self.out_path,pages)
            print('Parsing structure...')
            npd_anno.extract_structure(model_structure,assume_london=assume_london,override=override)
        
    def process(self,model_structure,model_lemma,assume_london=True,override=True):
        #idx=0 # try to get rid of this?
        for year, pages in self.editions.items():
            #for idx,page_range in enumerate(page_ranges):
            print(f'Processing {year}.')
            npd_anno = AnnotationEnv(year,self.in_path,self.out_path,pages)
            print('Parsing structure...')
            npd_anno.extract_structure(model_structure,assume_london=assume_london,override=override)
            print('Annotating lemmas...')
            npd_anno.annotate_lemmas(model_lemma)
            print('Saving...')
            npd_anno.save()
            print('Done!')
                
                
    def load(self):
        out_dict = defaultdict(dict)
        #idx = 0 # try to get rid of this?
        for year,pages in tqdm(self.editions.items()):
            #for idx,page_range in enumerate(page_ranges):
                
            npd = NPD(year,self.in_path,self.out_path,pages)
            out_dict[year] = npd.load_pickle()
        return out_dict

    def load_csv_from_path(self,path,suffix=''):
        dfs = []
        for year,pages in tqdm(self.editions.items()):
            csv_path = path / f'MPD_{year}{suffix}.csv'
            if csv_path.is_file():
                df = pd.read_csv(csv_path, index_col=0)
                df['YEAR'] = year
                dfs.append(df)
        self.dfs = pd.concat(dfs,axis=0)
        return self.dfs 


            
    
    def segment_annotation_export(self,model_path,prioritize=None,size=10):
        for year,pages in self.editions.items():
            #for pr in pranges:
            anno = AnnotationEnv(
                            year,
                            self.in_path,
                            self.out_path,
                            pages
                                ).segment_annotation_export(size,prioritize,model_path=model_path)
    
    def page_annotation_export(self,size=10,start_at=0,method='random'):
        for year,pages in self.editions.items():
            #for pr in pranges:
            start_at = np.random.randint(0, pages[-1] - pages[0] - size)
            #print(start_at)
            anno = AnnotationEnv(
                            year,
                            self.in_path,
                            self.out_path,
                            pages
                                ).page_annotation_export(size,start_at,method)
                
                
    def to_csv(self): 
        for year,pages in self.editions.items():
            #for pr in pranges:
            npd = NPD(year,self.in_path,
                        self.out_path,
                        pages,
                        verbosity=1)
            npd.to_csv()
                
    def to_excel(self): 
        for year,pages in self.editions.items():
            #for pr in pranges:
            npd = NPD(year,self.in_path,
                        self.out_path,
                        pages,
                        verbosity=1)
            npd.to_excel()  
                
    def link_records(self,model,path,fields,to_folder,jump=1):
        """jump (int): how many years to jump ahead for finding the matching record"""
        years = sorted(self.editions.keys())
        npd_collection = self.load_csv_from_path(path,suffix='')
        #npd_collection = self.load()
        
        recordlinker = RecordLinker(npd_collection)
        recordlinker.vectorize(fields)
        
        for i in range(len(years)-jump):
            
            print(years[i],years[i+jump])
            annotator = Annotator(years[i],years[i+jump],recordlinker)
            annotator.classify(model,to_folder=to_folder)
            
    def add_record_links(self):
        pass
        
    def create_csv_training_data(self,out_path,train_perc=0.7,
                                dev_perc=0.15,level='lemmas',recode=None,
                                return_dfs=False,clip_bioes=False): # ,
        dfs = []
        target = {'lemmas':'NPDtags','structure':'Structure'}[level]
        boundaries = False
        for year, pages in self.editions.items():
            
            #for pr in page_ranges:
                
            anno = AnnotationEnv(year,self.in_path,self.out_path,pages)
            
            if level == 'structure':
                boundaries = True
            dfs.extend(anno.df_export(level,recode,boundaries))
        
        random.seed(42)
        random.shuffle(dfs)
        
        size = len(dfs)
        
        train_idx = int(size*train_perc)
        dev_idx = train_idx + int(size*dev_perc)
        
        splits = {}
        splits["train"] = dfs[:train_idx]
        
        if dev_perc: 
            splits["dev"]  = dfs[train_idx:dev_idx]
        
        splits["test"] = dfs[dev_idx:]
    
        print(train_idx,dev_idx,size)
        print(level)
        
        out_path.mkdir(exist_ok=True)

        for split,df_split in splits.items():
            with open(out_path / f"{split}_{level}.csv",'w') as corpus_out:
                for lemma in splits[split]:
                    for i,line in lemma.iterrows():
                        target_tag = line[target]
                        if clip_bioes:
                            target_tag = re.sub("^[IEOBS]\-",'',target_tag)
                        corpus_out.write('{}\t{}'.format(line.token,target_tag) +'\n') # add line.value? for ocr
                    corpus_out.write('\n')
        
        if return_dfs:
            return splits["train"],splits["dev"],splits["test"]

    