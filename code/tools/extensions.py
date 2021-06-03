import pandas as pd
from pathlib import Path
import zipfile
import random
import re
from tools.document_tools import WebAnnoProcessor
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import *
from flair.models import SequenceTagger
from flair.data import Sentence
from flair.trainers import ModelTrainer
from tqdm import tqdm

tqdm.pandas()

class AnnotationTask(object):
    def __init__(self,df,task_name,target_field='TEXT'):

        self.df = df
        self.task_name = task_name
        self.target_field = target_field

        self.df[self.target_field].fillna('',inplace=True)

        self.annotations_folder = Path('annotations')
        self.task_folder = self.annotations_folder / self.task_name
        self.save_annotations = self.task_folder / 'to_annotate'
        self.load_annotations = self.task_folder / 'annotated'
        self.inception_export = self.task_folder / 'inception_export'
        self.training_data = self.task_folder / 'training_data'
        self.model_path = self.task_folder / 'model'
        
        self.annotations_folder.mkdir(exist_ok=True)
        self.task_folder.mkdir(exist_ok=True)
        self.save_annotations.mkdir(exist_ok=True)
        self.load_annotations.mkdir(exist_ok=True)
        self.inception_export.mkdir(exist_ok=True)
        self.training_data.mkdir(exist_ok=True)
        self.model_path.mkdir(exist_ok=True)


    @staticmethod
    def save_text(row,save_to,target_field):
        with open(save_to / (row.id + '.txt'), 'w') as out_doc:
            out_doc.write(row[target_field])
        return True

    def sample(self,**kwargs):
        self.sample = self.df[~self.df[self.target_field].isnull()].sample(**kwargs)

    def export_sample(self):
        self.sample.apply(self.save_text,
                            save_to=self.save_annotations,
                            target_field=self.target_field
                            ,axis=1)

    def load_annotated_files(self):
        files = list(self.load_annotations.glob('*.tsv'))
        return files

    def load_inception_export(self):
        zip_files = list(self.inception_export.glob('*.zip'))


        if not zip_files:
            print('No INCEpTION Export.')
        else:
            zip_file_path = zip_files[0]
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.inception_export)
        return [p for p in (self.inception_export / 'annotation').glob('**/*.tsv')]


    def create_training_data(self,target_col='NPDdescr',
                                train_perc=0.7,
                                dev_perc=0.15, 
                                clip_bioes=False):

        annotations = self.load_inception_export()
        webanno_dfs = [WebAnnoProcessor(a).tsv2df(target_col=target_col,recode=False) for a in annotations]

        random.seed(42)
        random.shuffle(webanno_dfs)
        
        size = len(webanno_dfs)
        
        train_idx = int(size*train_perc)
        dev_idx = train_idx + int(size*dev_perc)
        
        splits = {}
        splits["train"] = webanno_dfs[:train_idx]
        
        if dev_perc: 
            splits["dev"]  = webanno_dfs[train_idx:dev_idx]
        
        splits["test"] = webanno_dfs[dev_idx:]
    
        print(train_idx,dev_idx,size)
        
        for split,df_split in splits.items():
            with open(self.training_data / f"{split}.csv",'w') as corpus_out:
                #corpus_out.write('token\ttag\n')
                for lemma in df_split:
                    for _,line in lemma.iterrows():
                        try:
                            target_tag = line[target_col]
                        except:
                            return lemma
                        if clip_bioes:
                            target_tag = re.sub("^[IEOBS]\-",'',target_tag)
                        corpus_out.write('{}\t{}'.format(line.token,target_tag) +'\n') # add line.value? for ocr
                    corpus_out.write('\n')
    
    def train_model(self):

        columns = {0: 'text', 1: 'label'} 
        corpus = ColumnCorpus(self.training_data, columns,
                            train_file= 'train.csv',
                            test_file= 'test.csv',
                            dev_file= 'dev.csv')
        
        tag_dictionary = corpus.make_tag_dictionary(tag_type='label')
        
        print(corpus)
        print(tag_dictionary)

        embeddings = TransformerWordEmbeddings('bert-base-cased',fine_tune=True, allow_long_sentences=True,pooling_operation='mean',)

        tagger = SequenceTagger(hidden_size=128,
                        embeddings=embeddings,
                        tag_dictionary=tag_dictionary,
                        tag_type='label',
                        use_crf=True,
                        )

        trainer = ModelTrainer(tagger, corpus)

        trainer.train(self.model_path / 'tagger',
            learning_rate=0.05,
            mini_batch_size=2, # previously used value 5
            patience=2,
            anneal_factor=.5,
            max_epochs=10,
            embeddings_storage_mode='cpu',
            monitor_test=True,
            anneal_with_restarts=True,
            train_with_dev=False,
            )

    def apply_model(self, path):
        def predict(x):
            x = Sentence(x)
            tagger.predict(x)
            return x

        tagger = SequenceTagger.load(path)
        self.df[f'predictions_{self.task_name}'] = self.df[self.target_field].progress_apply(predict)

    def extract_tag(self, target_tag):
        def extract(x):

            if x is None: 
                return ''

            tokens = [t.text for t in x if t.get_tag('label').value == target_tag]
            if tokens:
                return ' '.join(tokens)
            return ''
        
        self.df[f'extract_{self.task_name}'] = self.df[f'predictions_{self.task_name}'].apply(extract)

    def save(self,path=Path('output_data'),
                    name = "MPD_export_1846_1920",
                    drop_predictions=True):
        if drop_predictions:
            self.df.drop(f'predictions_{self.task_name}', axis=1, inplace=True)
        self.df.to_csv(path / f"{name}_{self.task_name}.csv")