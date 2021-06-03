from tools.helpers import *
from pathlib import Path
import pathlib
import pandas as pd
from tqdm.notebook import tqdm
from types import FunctionType
import re
import pickle
import json
import codecs

class TXTProcessor(object):
    """Class for manipulating .txt documents
    Main function is to read a file and preprocess it for 
    feature extraction and annotation
    """
    
    def __init__(self,lines,path=None):
        """
        Args:
            lines (list): a list which contains the individual lines
            path (str): a string that contains the (relative) path to the document
        """
        if lines is None:
            self._lines = [] 
        else:
            self._lines = [process_line(line) for line in lines]
        #self._lines = lines #[process_line(l) for l in lines if l] # discard empty lines with l.strip()
        self._path = path
        if self._path:
            self._name = path.stem
        
        
    @classmethod
    def read(cls, path):
        """Read lines from a txt document
        Args:
            path (str): a string that contains the (relative) path to the document
        """
        with codecs.open(path,"r",encoding="utf-8-sig") as input_doc:
            return cls([line for line in input_doc.readlines() if line.strip()],path) # discard empty lines with l.strip() # remove process_line here

    def to_webanno(self):
        """converts a txt document to a webanno formatted document"""
        webanno_string = """#FORMAT=WebAnno TSV 3.2\n#T_SP=de.tudarmstadt.ukp.dkpro.core.api.transform.type.SofaChangeAnnotation|operation|reason|value\n#T_SP=webanno.custom.NPDSchema|NPDtags\n#T_SP=webanno.custom.NPDstructure|Structure\n\n\n"""
        l_idx,ch_idx = 1,0 # line index and character index
        for line in self.__iter__():
            line_string = ' '.join(line)
            webanno_string+=f'#Text={line_string}\n'
            t_idx = 1 # term index
            for t in line: # for term in line
                empy_tags = "\t_"*5
                webanno_string+=f"{l_idx}-{t_idx}\t{ch_idx}-{ch_idx+len(t)}\t{t}{empy_tags}\n"
                ch_idx+=len(t)+1 # update character index
                t_idx+=1 # uppdate term index
            l_idx+=1 
            webanno_string+="\n"
        return webanno_string.strip()

    def __len__(self):
        return len(self._lines)
    
    def __str__(self):
        return "<TXTProcessor object with %s lines>" % self.__len__()
    
    def __iter__(self):
        for line in self._lines:
            yield line


class WebAnnoProcessor(object):
    
    def __init__(self,file):
        """processing webanno files
        reads tsv (str) files and converts it to a dataframe

        """
        self.file = file
        #if isinstance(self.file, pathlib.PosixPath):
        #    self.data = self.tsv2df()
            
        #if isinstance(self.file,pd.DataFrame): # do we need this functionality? not implemented
        #    self.data = self.df2tsv()
    
    
    def get_column_names(self):
        """get the column names of a webanno tsv document
        """
        colnames = []

        with open(self.file,'r') as in_tsv:
            for i,line in enumerate(in_tsv.readlines()):
                if i > 10: # !! TO DO check this, stops after tenth line 
                    break
                if line.startswith('#T_SP'):
                    colnames.extend(line.strip().split("|")[1:])
        

        return colnames
            
    def tsv2df(self,target_col,recode,boundaries=False):
        """converts a tsv document to pandas dataframe
        Adheres to BIOE format
        """
        rows = [] 
        columns = ['line_idx','char_idx',"token"] + self.get_column_names()
        
        target_missing = False
        if target_col not in columns:
            target_missing = True
            columns.append(target_col)
        #print(columns)
        
        with open(self.file,'r') as in_tsv:

            for line in in_tsv.readlines():
                #if (not line.strip()) or line.startswith("#"): # skip empty lines or lines starting with a hashtag
                if line.startswith("#"):
                    continue
                if not line.strip():
                    if boundaries:
                        rows.append(['@','@','@','O'])
                    continue

                line_split = line.strip().split("\t") # split by tab
                #print(line_split)
                if target_missing:
                    line_split.append('O')

                rows.append(line_split)
            
        df_lemma =  pd.DataFrame(rows,columns=columns)
        
        tagidx_count = {}
        tag_idx_pattern = re.compile('\[([0-9]{1,})\]')

        #head_tag_idx = {'NPDtags':-1,'Structure':0}[target_col] # this should be changed some inconsistency in the way the tags are constructed
        #print(df_lemma.columns,self.file)
        for i,row in df_lemma.iterrows():
            #if target_missing:
            #    row[target_col] = 'O'

            tag, tag_idx = get_tag_and_idx(row[target_col],tag_idx_pattern)
            if recode:
                if type(recode) == dict:
                    tag = recode.get(tag,tag)
                elif type(recode) == FunctionType:
                    tag = recode(tag)
                #tag = tag.split('-')[head_tag_idx]

            if tag in ['_','*']:
                    row[target_col] = 'O'
            
            elif tag_idx : 
                tag_idx = int(tag_idx[0])
                tagidx_count.setdefault(tag_idx,0)
                tagidx_count[tag_idx]+=1
        
                if tagidx_count[tag_idx] < 2:
                    row[target_col]  = 'B-' + tag
                
                elif tagidx_count[tag_idx] >= 2 and i < (df_lemma.shape[0]-1):
                    next_tag, next_tag_idx = get_tag_and_idx(df_lemma.loc[i+1,target_col],tag_idx_pattern)
                    
                    if next_tag_idx:
                        next_tag_idx = int(next_tag_idx[0])
                    
                    else:
                        next_tag_idx = -1
            
                    if next_tag_idx == tag_idx:
                        row[target_col]  ='I-' + tag
                    
                    elif next_tag_idx != tag_idx:
                        row[target_col]  ='E-' + tag
                
                else: # set last element to end tag
                    row[target_col]  = 'E-' + tag
            
            else:
                row[target_col]  = 'B-' + tag


        return df_lemma
        
    def df2tsv(self):
        """not implemented, check if this is needed
        """
        pass