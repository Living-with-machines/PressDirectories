from annoy import AnnoyIndex
from pigeon import annotate
from functools import reduce
from scipy.sparse import hstack,vstack
from collections import defaultdict
from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from torch.optim.adam import Adam
from flair.data import Corpus, Sentence
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from flair.datasets import CSVClassificationCorpus
import numpy as np
import pandas as pd
import pickle
import random

### ------ Linking code --------

def filter_entry(district,entry):
    """returns entries as a mapping from tags to text
    """
    # remove BIOES tags
    clip_bioes = lambda x: (x[1].split('-')[-1],x[0])
    
    # convert entry to a list of (tag, token) tuples
    ttt = map(clip_bioes, # list of tag token tuple
                filter(lambda x: x[1] not in ['O'], entry))

        
    tag2text = defaultdict(str)
    
    # add tokens their respective keys
    for tag,token in ttt:
        tag2text[tag] += token + ' '

    # add district
    tag2text['DISTRICT'] = district
    return tag2text


def record_pair_string(l_series,r_series,fields=['S-TITLE','S-POL','D-PUB','D-EST','DISTRICT','COUNTY','E-ORG','E-PER']):
    
    l = ' || '.join(l_series[fields])
    r = ' || '.join(r_series[fields])
    return l + " [SEP] " + r

class RecordLinker(object):
    """Loads a NPD collection. Collects information from the individual lemmas by year.
    Arguments:
        npd_collection (dict):
        
    """
    
    def __init__(self,npd_collection):
        """Arguments:
            npd_collection (dict): a mapping for year to npd content 
        """

        self.collection_df = npd_collection
        self.collection_df.reset_index(inplace=True,drop=True)
        self.collection_df.fillna('None',inplace=True)
        
        self.npds = []
        self.years = []
        self.indices = []
        self.vectorized = False # flag to check if collection is vectorized
        
        
        
        print("number of books in the collections: ",len(self.collection_df.YEAR.unique()))

    def vectorize(self,fields): # to do: avoid this function being called all the time
        """Vectorization function. 
        Retrieves all entries as a list using the self.collect_information method.
        
        Creates:
                self.years: a numpy.array with all the years that were vectorized
                self.entries: a numpy.array with all the entries
                self.titles: a numpy.array with all the titles
                self.matrix: a sparse matrix that is a concatenation of the vectorized sparse matrices
                
        After vectorization it sets the vectorized flag to True.
        """
        
        print("number of entries in the collection: ",self.collection_df.shape[0])
        
        tag2vectorizer = {}
        
        for tag in fields:
            
            vectorizer = TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=10) 
            tag2vectorizer[tag] = vectorizer.fit_transform(list(self.collection_df[tag]))
    
        self.vectorized = True
        
        self.title_vectors = hstack([vec for targ,vec in tag2vectorizer.items()]).tocsr()
        
class LinkLoader(object):
    def __init__(self,recordlinker,fields=['S-TITLE', 'S-POL', 'DISTRICT','COUNTY', 'S-PRICE','D-EST', 'D-PUB', 'E-LOC', 'E-ORG', 'E-PER']):
        self.recordlinker = recordlinker
        if not self.recordlinker.vectorized: 
            print('Information not vectorized. Vectorizing...')
            self.recordlinker.vectorize(fields)
        print('Data vectorized.') 
        
class Index(LinkLoader):
    """Class that handles the indexing and search of entries in the NPD collection.
    
    """
    
    def __init__(self,recordlinker):
        """Instantiates an annoy index.
        Arguments:
            npd_collection (dict): mapping of years to npd content
        """ 
        super().__init__(recordlinker)
    
        self.title_vectors_year = None
        self.titles_year = None
        
    def create_index(self,select_year=None):
        """
        """
        # There is an issues here
        # The annoy index deviates from the global index
        # Temporary solution: map indices annoy2global?
        print(self.recordlinker.collection_df.shape)
        
        year_indices = self.recordlinker.collection_df[self.recordlinker.collection_df.YEAR == select_year].index
        self.title_vectors_year = self.recordlinker.title_vectors[year_indices]
        
        self.titles_year = self.recordlinker.collection_df[self.recordlinker.collection_df.YEAR == select_year]['S-TITLE']
        print(self.recordlinker.collection_df.shape[0],len(self.titles_year))
        self.index = AnnoyIndex(self.title_vectors_year.shape[1], metric='euclidean')
        self.index_to_titles = {i:t for i,t in enumerate(self.titles_year)}
        self.titles_to_index = {t:i for i,t in self.index_to_titles.items()}
        self.annoy_idx2global_idx = {i:j for i,j in enumerate(year_indices)}
        
        print("Building Index!")
        for _, i in self.titles_to_index.items():
            
            self.index.add_item(i, self.title_vectors_year[i].todense()[0].T)
        self.index.build(50)
        print("Finished!")
        
    def get_closest_title(self,vector,n=5):
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [(neighbor,self.index_to_titles[neighbor]) for neighbor in nn_indices]
    
def show_entries(x, exclude= ['TEXT','DISTRICT_DESCRIPTION']):
    
    tags = x[1].index
    for tag in tags:
        if tag in exclude: continue
        tag_line='{0:<10} | {1:>40} | {2:>40}'.format(tag,
                                            x[1][tag],
                                            x[2][tag])
        print(tag_line)
        print('_'*(len(tag_line)-1))

class Annotator(Index):
    """class for manual and automatic annotation"""
    
    def __init__(self,source,target,recordlinker):
        super().__init__(recordlinker)
        
        self.source = source
        self.target = target
        
        self.query_idx = recordlinker.collection_df[recordlinker.collection_df.YEAR==self.source].index
        
        self.create_index(self.target)
        self.labels = []
 

    def reset(self):
        self.labels = []
        
    def annotate_examples(self,sample): # =5
        
        query_idx_copy = list(self.query_idx.copy())
        random.shuffle(query_idx_copy)
        annotation_idx = query_idx_copy[:sample]
        self.query_idx = [i for i in query_idx_copy if i not in annotation_idx]
        annotation_examples = []
        
        for idx in annotation_idx:
            
            examples = [self.annoy_idx2global_idx[e] 
                            for e,ex in self.get_closest_title(self.recordlinker.title_vectors[idx].todense()[0].T)]


            for t_idx in examples: 
                annotation_examples.append(((idx,t_idx),
                                            self.recordlinker.collection_df.iloc[idx],
                                            self.recordlinker.collection_df.iloc[t_idx]
                                            ))

            
        
        
        self._annotations = annotate(annotation_examples, options=['same', 'different', "don't know"],
                                display_fn=show_entries)
    def add(self):
        self.labels.extend(self._annotations)

    @staticmethod
    def load_model(model_path):
        return TextClassifier.load(model_path)
    
    def classify(self,model,n_candidates=10,verbose=False, to_folder = Path('./link_dump/predictions/')):
        """
        Arguments:
            model (TextClassifier): FLAIR text classification model
            n_candidates (int): only consider the highest ranked n candidates for record linkage
        """
        
        links = []
        
        for idx in tqdm(self.query_idx):
            #self.recordlinker.titles[idx]
            candidates = [self.annoy_idx2global_idx[e] 
                            for e,ex in self.get_closest_title(self.recordlinker.title_vectors[idx].todense()[0].T,n_candidates)]
            for t_idx in candidates: 
                
                text = Sentence(record_pair_string(self.recordlinker.collection_df.iloc[idx],self.recordlinker.collection_df.iloc[t_idx]))
                
                model.predict(text)
                if text.labels[0].value=='same':
                    
                    if verbose:
                        print("-"*10)
                        print(self.recordlinker.collection_df.iloc[idx])
                        print("->")
                        print(self.recordlinker.collection_df.iloc[t_idx])
                        print("-"*10)
                        print("\n")
                    
                    links.append({"source": {"idx": self.recordlinker.collection_df.iloc[idx].id,"year":self.source},
                                "target": {"idx": self.recordlinker.collection_df.iloc[t_idx].id,"year":self.target}})
                    break # stop at first predicted link? change this?
        
        
        to_folder.mkdir(exist_ok=True)
        
        with open(to_folder / f'predictions_{self.source}_{self.target}.pickle','wb') as out_pickle:
            pickle.dump(links, out_pickle)

    def __str__(self):
        return f"< Annotators with {len(self.labels)} annotations >"

    def __len__(self):
        return len(self.labels)

    def save(self,link_dump_folder=Path('./link_dump')):
        
        link_dump_folder.mkdir(exist_ok=True)
        random_id = random.randint(100,1000000)
        save_to_path = link_dump_folder / f'link_dump_{random_id}.pickle'
        
        with open(save_to_path, 'wb') as out_pickle:
            pickle.dump(self.labels,out_pickle)

        print(f"Saved {len(self.labels)} annotation to {save_to_path}")
        print('Removed previous annotations')
        self.reset()
        

class RLModelTrainer(LinkLoader):
    """Train record linkage model
    """
    
    def __init__(self,recordlinker):
        super().__init__(recordlinker)
        
        self.annotations = []
        self.X,self.y = [],[]
    
    def load_annotations(self,annotations_dump):
        dumps = annotations_dump.glob('*.pickle')
        for d in dumps:

            with open(d,'rb') as in_pickle:
                
                self.annotations.extend(pickle.load(in_pickle))
        
        print(f'Loaded {len(self.annotations)} annotations')
        
    

    def export_csv(self,save_to="../link_dump/training"):
        """Export annotations to csv so it can be read into FLAIR.
        One line per annotations
        """
        if not self.annotations:
            self.load_annotations()
            
        lines = []
        
        for (anno,label) in self.annotations:
            _, l_dict, r_dict = anno
            lines.append([ record_pair_string(l_dict,r_dict), label ])
        
        df = pd.DataFrame(lines,columns=['text','label'])
        print(df.shape)
        df.drop_duplicates(inplace=True)
        print(df.shape)
        df["split"] = [np.random.choice(['train','dev','test'], p=[0.7, 0.1, 0.2]) for _ in range(df.shape[0])]
        for split in ['train','test','dev']:
            df[df.split==split].to_csv(save_to / f"{split}.csv",sep='\t')
            
    def train_with_transformer(self,save_to_path,data_folder = '../link_dump/training/'):

        column_name_map = {1: "text", 2: "label"}

        corpus: Corpus = CSVClassificationCorpus(data_folder,
                                        column_name_map,
                                        skip_header=True,
                                        delimiter='\t')

        label_dict = corpus.make_label_dictionary()
        
        document_embeddings = TransformerDocumentEmbeddings('bert-base-cased', fine_tune=False)

        classifier = TextClassifier(document_embeddings, 
                                    label_dictionary=label_dict,
                                    loss_weights={'same': 2.})

        
        trainer = ModelTrainer(classifier, corpus, optimizer=Adam)

    
        trainer.train(save_to_path / 'link_classifier',
                learning_rate=.05, 
                mini_batch_size=8,
                patience=2,
                mini_batch_chunk_size=4, 
                max_epochs=10, 
            )

    def train_with_svm(self,clf=LinearSVC(class_weight="balanced",C=10)):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.5, random_state=42)
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)
        print(classification_report(y_pred,y_test))



