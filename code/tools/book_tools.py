from tools.helpers import process_line, clean_df
from tools.document_tools import TXTProcessor, WebAnnoProcessor
from pathlib import Path
from itertools import product
from collections import defaultdict
import pandas as pd
from tqdm.notebook import tqdm
import re
import pickle
import json
import codecs

from collections import defaultdict

def parse_entry(district,district_descr,idx,entry):
    """returns entries as a mapping from tags to text
    """

    tag2texts = []

    prev_tag = ''
    # remove BIOES tags
    tag2text, tokens = defaultdict(str), []
    added_entries = 1
    for i,(token,bio_tag) in enumerate(entry):
        
        prev_comp = prev_tag.split('-')[-1]

        if bio_tag == "B-S-TITLE":
            #if not tag2text:
            #    #tag2text = defaultdict(str)
            #    tag2text['id'] = idx
            
            if tag2text and (prev_comp != 'TITLE'): 
                tag2text['TEXT'] = ' '.join(tokens)
                tag2texts.append(tag2text)
            
                tag2text, tokens = defaultdict(str), []
                tag2text['id'] = f"{idx}_{added_entries}"
                added_entries+=1
            
        tag2text['id'] = idx
        tokens.append(token)

        if bio_tag == 'O':
            continue
        
        
        elements = bio_tag.split('-')
        bioes,tag = elements[0],'-'.join(elements[1:])#bio_tag.split('-')
        
        if bioes == 'B':
            
            if tag2text[tag]:
                tag2text[tag] += ('<SEP>' + token)
            else:
                tag2text[tag] += token
        elif bioes != 'B':
            tag2text[tag] += ('<CON>' + token)
        
        prev_tag = bio_tag
            
    for tag, string in tag2text.items():
        tag2text[tag] = '<SEP>'.join([' '.join(e.split('<CON>')) for e in string.split('<SEP>')])
    
    # add district
    tag2text['DISTRICT'] = district
    tag2text['DISTRICT_DESCRIPTION'] = district_descr
    tag2text['TEXT'] = ' '.join(tokens)
    tag2texts.append(tag2text)
    #tag2text['TEXT'] = ' '.join([t for t,bt in entry])
    return tag2texts

class Book(object):
    """The Book Object records basic information on the level of the individual book
    It prime use is the sample annotation pages from a NPDs
    """
    def __init__(self,year,in_path,out_path,pages,verbosity=0):
        """Object to record the basic partition of a NDP object
        Args:
            london (tuple): indicates at which pages the London section starts and ends
            other (tuple): indicates at which pages the Other (Provincial, Wales, Scotland) section starts and ends
            seed (int): set random seed
        """
        
        self._vol = f'{year}' #f'MPD_{year}'
        self._path = in_path
        
        self._out_path = out_path / self._vol
        self._out_path.mkdir(exist_ok=True)
        
        pages = [self._generate_file_path(p_idx) for p_idx in pages] # list(range(*page_range))
        
        #print(pages)
        self.pages = [p for p in pages if p.is_file()]

        self.df = None
        
        self.status = {}
        self.verbosity = verbosity
        self.check_status()

    def _generate_file_path(self,p_idx,dig_format='Plain Text'):
        
        return self._path / self._vol / dig_format / (self._vol + f'_{p_idx:03}'+".txt")
    
    
    def _load_annotated(self,level,stage):
        
        folder = self._out_path / f"{stage}_{level}"
        pages = list(folder.glob("**/MPD*.*")) + list(folder.glob("**/seg_MPD*.*"))
        
        if level == 'structure':
            self.status[folder.name] = self.sort_pages_ascending(pages)
        
        elif level == 'lemmas':
            self.status[folder.name] = pages
        
    def _check_folders(self,folders):
        
        for folder in folders:
            folder.mkdir(exist_ok=True)
    
    def check_status(self,verbose=True):
        """helper that monitors the progress
        useful for tracking the number of annotated pages/size of training data etc.
        """
        folders = [self._out_path / f for f in ["to_annotate_structure",'annotated_structure', 
                    'to_annotate_lemmas','annotated_lemmas', # annotations for semantic annotation
                    "lemmas_raw"]] # TO DO: check if we need these folders
        
        self._check_folders(folders)
        
        for level,stage in list(product(["lemmas","structure"],['to_annotate','annotated'])):
            self._load_annotated(level, stage)
        
        # output of lowest level segments, is used for annotation export later
        lemmas_raw_path = self._out_path / 'lemmas_raw' / 'lemmas_raw.pickle'
        
        if (lemmas_raw_path).is_file():
            self.status['lemmas_raw'] = pickle.load(open(lemmas_raw_path,'rb'))
            
        structure_raw_path = self._out_path / 'lemmas_raw' / 'structure_raw.pickle'
        
        if (structure_raw_path).is_file():
            self.status['structure_raw'] = pickle.load(open(structure_raw_path,'rb'))
        
        if self.verbosity:
            print("_"*50+"\n")
            out_string = ''
            print(self._vol)
            for n,paths in self.status.items():
                print(n,len(paths))
            
        return True
    
    def __str__(self):
        return f"<Collection object with {len(self.pages)} pages>" 
    
    def __len__(self):
        return len(self.pages)
    
    @staticmethod            
    def sort_pages_ascending(paths,page_func=lambda p: re.compile('[0-9]{3}').findall(p.name)[-1]):
        page2path = {int(page_func(p)) : p for p in paths}
        return [page2path[k] for k in sorted(page2path.keys())]
        
    def load_pages(self): # changed from load NPD pages
        paths = self.sort_pages_ascending(self.pages)
        #print(paths)
        return [TXTProcessor.read(p) for p in paths if p.is_file()]
    
    def load_pickle(self):
        with open(self._out_path / (self._vol + ".pickle"), 'rb') as in_pickle:
            return pickle.load(in_pickle)
    
class NPD(Book):
    def __init__(self,year,in_path,out_path,pages,verbosity=0):
        year =  f'MPD_{year}'
        super().__init__(year,in_path,out_path,pages,verbosity)
        
        
    def __str__(self):
        return f"<NPD object with {len(self.pages)} pages>" 
    
    def to_df(self): # get rid of this index thingy later
        content = self.load_pickle()
        
        items_nested = [parse_entry(district,district_descr,idx,[(t.text,t.get_tag("lemmas").value) for t in entry])
                        for (district, district_descr), entries in content.items() for idx,entry in entries]

        items = [item for items in items_nested for item in items]

        # clean dataframe
        self.df = clean_df(pd.DataFrame(items))
    
    def to_csv(self):
        self.to_df()
        self.df.to_csv(self._out_path / f'{self._vol}.csv')

    def to_excel(self):
        self.to_df()
        self.df.to_excel(self._out_path / f'{self._vol}.xlsx')