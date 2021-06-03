from tools.helpers import process_line
from tools.document_tools import TXTProcessor, WebAnnoProcessor
from tools.book_tools import Book, NPD
from tools.crf_tools import *
#from tools.utils import *
from collections import defaultdict
from flair.models import SequenceTagger
from flair.data import Sentence
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm
import pycrfsuite
import re
import pickle
import random
import json
import codecs
import numpy as np

class AnnotationEnv(NPD): # NPD
    """Class for managing the manual annotation process.
    Has tools selecting and exporting segments to Inception for annotation.
    
    """
    def __init__(self,year,in_path,out_path,page_range=(0,999)):
        super().__init__(year,in_path,out_path,page_range)
        self.hierarchy_dict = None
        
    def page_annotation_export(self,size=10,start_at=0,method='random'): 
        """Select pages for annotation from a certain region
        Args:
            folder_template (str): template of the folder where data is stored
            size (int): the number of pages to sample
            start_at (int): start at first page?
        """
        out_folder = self._out_path / "to_annotate_structure"
        out_folder.mkdir(exist_ok=True)
        
        if method=='random':
            file_paths = self.pages[:]
            random.shuffle(file_paths)
            file_paths = file_paths[:size]
        elif method=='range':
            file_paths = self.pages[start_at:size+start_at] 
        elif method == 'all':
            file_paths = self.pages
            
        
        for path in file_paths:
            if path.is_file(): # skipping the gaps in the page numbering for now
                txt = TXTProcessor.read(path) 
                with open(out_folder / (txt._name+".tsv"), 'w') as out_tsv:
                    out_tsv.write(txt.to_webanno())

    def segment_annotation_export(self,size,
                                prioritize=None,
                                seed=1984,
                                model_path='../../../../../npd_experiments/Models/structure_crf-final.model'):
        """Selects lemmas for annotation in INCEPTION
        Arguments:
            size (int): number of lemmas to export
            seed (int): randomize order of the detected lemmas
            model_path (str): path to crf model to split pages into lemmas
        """
        
        random.seed(seed)
        if not self.status.get('lemmas_raw',{}):
            print('Document is not parsed yet. Processing collection')
            self.extract_structure(model_path)

        lemmas = [(i,l) for i,(idx,l) in enumerate(self.status['lemmas_raw'])]
        if prioritize:
            prioritize = int(prioritize*len(lemmas))
            #print(prioritize,len(lemmas))
            lemmas = lemmas[:prioritize]
        #lemmas = self.status['lemmas_raw']
        random.shuffle(lemmas)
        out_path = self._out_path / "to_annotate_lemmas"

        for i,l in lemmas[:size]:
            txt = TXTProcessor([l])
            webanno = txt.to_webanno()

            with open(out_path / ('seg_'+self._vol + "_" + str(i)+".tsv"), "w") as out_tsv:
                out_tsv.write(webanno)
                
    def df_export(self,level,recode=None,boundaries=False):
        """Export all annotated as tsv files as pandas.DataFrames
        """
        target_col = {'lemmas':'NPDtags','structure':'Structure'}[level]
        
        return  [WebAnnoProcessor(p).tsv2df(target_col,recode,boundaries) for p in self.status[f"annotated_{level}"]]
        
    def extract_structure(self, structure_model_path, 
                        level_1_tags = ["LOC","LOCDESCR"], # define hierarchy, the first tag should be the start element
                        level_2_tags = ["TITLE","NEWSPAPERDESCR"],
                        ignore_tags = ['O','HEADER'],
                        assume_london=True,
                        override=True): # override=False
        """
        extract structure of directories. it has newspapers nested in districts
        the first element of each set of level tags defines the start of a new element, i.e. level 1 is defined by LOC etc
        
        level 1 # item 1
            |__ level 2 # item 1
            |__ level 2 # item 2
            |__ etc.
            
        level 1 # item 2
            |__ level 2 # item 1
            |__ level 2 # item 2
            |__ etc.
            
        """
        
        pages = [Sentence(' '.join([w for l in page._lines for w in ['@']+ l])) # can prepend <S> tag here ``for w in ["<S>"] + l''
                                for page in self.load_pages()] # check if pages are ordered correctly
        
        #print(pages)
        structure_annotation_path = self._out_path / 'lemmas_raw' / 'structure_raw.pickle'
        
        if structure_annotation_path.is_file() and (not override):
            print('Structure already parsed, loading data...')
            with open(structure_annotation_path,'rb') as in_pickle:
                data = pickle.load(in_pickle)

            print('Done loading data.')
            
        else:
            print('Structure not parsed yet, creating data...')
            tagger = SequenceTagger.load(structure_model_path)

            if assume_london:
                data = [[0,'London','LOC'],[1,'London','LOCDESCR']]
                j = 2
            else:
                data = []
                j = 0
        
            for page in tqdm(pages):
                tagger.predict(page)
                data.extend([[i+j,token.text,token.get_tag('structure').value]
                                for i,token in enumerate(page)
                                ]
                        )
            with open(structure_annotation_path,'wb') as out_pickle:
                pickle.dump(data,out_pickle)
        
        df = pd.DataFrame(data, columns=["token_id","token","tag"])
        df_content = df[~df.tag.isin(ignore_tags)]
        df_content.reset_index(drop=True, inplace=True)
                
        offs_level_2 = np.where(df_content.tag.isin(level_2_tags))[0]
        # start of higher level entities
        # iterator over the offsets in the lower level entities
        # if the offset + one position is not at level two
        # add it as an offset for level 1
        # include the last element as final offset
        offs_level_1 = [0] + [o for o in offs_level_2
                                if o+1 not in offs_level_2] + [df_content.shape[0]]

        level_1_boundaries = [(offs_level_1[i],offs_level_1[i+1]) for i in range(len(offs_level_1)-1)]
        level_1_dfs = [df_content.iloc[s:e] for s,e in level_1_boundaries]

        # add a previous name variable
        # otherwise, if there is no LOC 
        # then there will be an empty string
        # for the level_1_name variable
        # in this case use the previously
        # encountered variable name
        previous_name = ''

        self.status['lemmas_raw'] = [] # make sure ingest is properly done
        hierarchy_dict = defaultdict(list)

        level_1_names = []
                
        for i,level_1_df in enumerate(level_1_dfs):
            
            level_1_name = ' '.join(level_1_df[level_1_df.tag==level_1_tags[0]].token)

            level_1_description = ' '.join(level_1_df[level_1_df.tag==level_1_tags[1]].token)
            
            if not level_1_name:
                level_1_name = previous_name + ' *'

            if not level_1_description:
                level_1_description = 'NA'
            
            df_lemma = level_1_df[level_1_df.tag.isin(level_2_tags)]
            
            offs_head = list(np.where(df_lemma.tag==level_2_tags[0])[0]) #+ [df_lemma.shape[0]] 
                    
            level_2_offs = [o for o in offs_head
                                        if o-1 not in offs_head] + [df_lemma.shape[0]]
                    
            level_2_boundaries = [(level_2_offs[i],level_2_offs[i+1]) for i in range(len(level_2_offs)-1)]
                    
            level_2_text = [[df_lemma.iloc[s].token_id,' '.join(df_lemma.iloc[s:e].token)] 
                                                for s,e in level_2_boundaries]
            
            # avoid collapsing newspaper from different 
            # locations in one key of the hierarchy_dict
            # add a whitespace if key already exists
            if level_1_name in level_1_names:
                level_1_name+= f' ##{i}'
            else:
                level_1_names.append(level_1_name)

            hierarchy_dict[(level_1_name,level_1_description)].extend(level_2_text)
            previous_name = level_1_name
                    
            self.status['lemmas_raw'].extend(level_2_text)
                
        with open(self._out_path / 'lemmas_raw' / 'lemmas_raw.pickle','wb') as out_pickle:
                pickle.dump(self.status['lemmas_raw'], out_pickle)
                
                self.hierarchy_dict = hierarchy_dict        
        
        
        
    def save(self):
        if self.hierarchy_dict:
            with open(self._out_path / (f"{self._vol}.pickle"),'wb') as out_pickle:
                pickle.dump(self.hierarchy_dict,out_pickle)
        else:
            print('No data processed. Run "extract_structure"')

    def annotate_lemmas(self,lemmas_model_path): # 'model/taggers/example-ocr/best-model.pt'
        
        if not self.hierarchy_dict:
            print('First perform page segmentation.')
            return False
        
        
        #npd_annotated = {}
        tagger = SequenceTagger.load(lemmas_model_path)
        
        ext = 0
        
        for district,newspapers in tqdm(self.hierarchy_dict.items()):
            annotated_newspapers = []
            for page_id,newspaper in newspapers:
                newspaper = Sentence(newspaper, use_tokenizer=False) 
                tagger.predict(newspaper)
                annotated_newspapers.append((f"{self._vol}_{ext}", newspaper))
                ext+=1
            
            self.hierarchy_dict[district] = annotated_newspapers

        
        

