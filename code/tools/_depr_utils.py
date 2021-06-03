import codecs
#from lxml import etree
from glob import glob
import pandas as pd
from tqdm import tqdm_notebook
from tools.helpers import *
from tools.crf_tools import SequenceVectorizer
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import defaultdict
import json
import random
import numpy as np
import pycrfsuite
import pickle


class TXTProcessor(object):
    """Class for manipulating .txt documents
    Main function is to read a file and preprocess it for 
    feature extraction and annotation
    """
    
    def __init__(self,lines,path):
        """
        Args:
            lines (list): a list which contains the individual lines
            path (str): a string that contains the (relative) path to the document
        """
        if lines is None:
            self._lines = [] 
        self._lines = lines #[process_line(l) for l in lines if l] # discard empty lines with l.strip()
        self._path = path
        self._name = self._path.stem
        
        
    @classmethod
    def read(cls, path):
        """Read lines from a txt document
        Args:
            path (str): a string that contains the (relative) path to the document
        """
        with codecs.open(path,"r",encoding="utf-8-sig") as input_doc:
            return cls([process_line(line) for line in input_doc.readlines() if line.strip()],path) # discard empty lines with l.strip()

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
        if isinstance(self.file, str):
            self.data = self.tsv2df()
            
        if isinstance(self.file,pd.DataFrame): # do we need this functionality? not implemented
            self.data = self.df2tsv()
    
    
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
            
    def tsv2df(self,remove_tagidx=True):
        """converts a tsv document to pandas dataframe
        Arguments:
            remove_tagidx (bool): removes the anno idx !! TO DO check and/or correct this
        """
        rows = [] 
        columns = ['line_idx','char_idx',"token"] + self.get_column_names()
        
        with open(self.file,'r') as in_tsv:
            for line in in_tsv.readlines():
                if (not line.strip()) or line.startswith("#"): # skip empty lines or lines starting with a hashtag
                    continue
                cells = line.strip().split("\t") # split by tab
                if remove_tagidx: # to do refine the handling of the tag idx
                    cells = [strip_anno_idx(c) if i > 2 else c for i,c in enumerate(cells)] # strip tag idx from third cell on
                rows.append(cells)
        return pd.DataFrame(rows,columns=columns)
        
    def df2tsv(self):
        """not implemented, check if this is needed
        """
        pass
    

            

    

