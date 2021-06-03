from tqdm import tqdm_notebook

class SequenceVectorizer(object):
    def __init__(self,context=10):
        self.context = context
        #self.data = None
        
    def transform(self,data):
        self.data = data
        return [self.vectorize(line) 
                    for line in tqdm_notebook(self.data)] #

    def vectorize(self,line):
        #example = self.data.loc[idx]
        #token = example.text
        corpus = []
        for idx,token in enumerate(line):
            
            #token = self.data[idx] #????
            features = ['bias',
                'token=%s'% token,
                'token.lower=%s' % token.lower(),
                'token.isupper=%s' % token.isupper(),
                'token.1.isupper=%s' % token[0].isupper(),
                'token.2.isupper=%s' % token[:2].isupper(),
                'token.istitle=%s' % token.istitle(),
                'token.isdigit=%s' % token.isdigit(),
                'token.end_2=%s' % token[-2:],
                'token.start_2=%s' % token[:2],
                #'token.shape=%s' % example.shape,
                #'token.is_stop=%s' % example.is_stop,
                'token.is_alpha=%s' % token.isalpha(),
                #'token.pos=%s' % example.pos,
                #'token.tag=%s' % example.tag,
                #'token.ent_type=%s' % example.ent_type,
                #'token.ent_iob=%s' % example.ent_iob,
                'token.length=%s' % len(token)
                
               ]
    
            for c in range(idx+1,idx+self.context+1):
                c = min(len(line)-1,c)
                if c < len(line)-1:
                    dist = abs(c-idx)
                    token1 = line[c]
            
                
                    features.extend([
                        '+{}.token={}'.format(dist,token1),
                        '+{}.token.lower='.format(dist,token1.lower()),
                        '+{}.token.istitle={}'.format(dist,token1.istitle()),
                        '+{}.token.isupper={}'.format(dist,token1.isupper()),
                        '+{}.token.1.isupper={}'.format(dist,token1[0].isupper()),
                        '+{}.token.2.isupper={}'.format(dist,token1[:2].isupper()),
                        '+{}.token.end_2={}'.format(dist,token1[-2:]),
                        '+{}.token.start_2={}'.format(dist,token1[:2]),
                        #'+{}.token.shape={}'.format(dist,example1.shape),
                        #'+{}.token.is_stop={}'.format(dist,example1.is_stop),
                        '+{}.token.is_alpha={}'.format(dist,token1.isalpha()),
                        #'+{}.token.pos={}'.format(dist,example1.pos),
                        #'+{}.token.tag={}'.format(dist,example1.tag),
                        #'+{}.token.ent_type={}'.format(dist,example1.ent_type),
                        #'+{}.token.ent_iob={}'.format(dist,example1.ent_iob),
                        '+{}.token.length={}'.format(dist,len(token1))
                    ])
                elif 'EOD' not in features:
                    features.append('EOD')
    
            for c in range(idx-self.context,idx):
        
                if c >= 0:
                    dist = abs(c-idx)
                    c = max(0,c)
            
                    token1 = line[c]
                    #token1 = example1.text
                
                    features.extend([
                        '-{}.token={}'.format(dist,token1),
                        '-{}.token.lower='.format(dist,token1.lower()),
                        '-{}.token.istitle={}'.format(dist,token1.istitle()),
                        '-{}.token.isupper={}'.format(dist,token1.isupper()),
                        '-{}.token.1.isupper={}'.format(dist,token1[0].isupper()),
                        '-{}.token.2.isupper={}'.format(dist,token1[:2].isupper()),
                        '-{}.token.end_2={}'.format(dist,token1[-2:]),
                        '-{}.token.start_2={}'.format(dist,token1[:2]),
                        #'-{}.token.shape={}'.format(dist,example1.shape),
                        #'-{}.token.is_stop={}'.format(dist,example1.is_stop),
                        '-{}.token.is_alpha={}'.format(dist,token1.isalpha()),
                        #'-{}.token.pos={}'.format(dist,example1.pos),
                        #'-{}.token.tag={}'.format(dist,example1.tag),
                        #'-{}.token.ent_type={}'.format(dist,example1.ent_type),
                        #'-{}.token.ent_iob={}'.format(dist,example1.ent_iob),
                        '-{}.token.length={}'.format(dist,len(token1))
                    ])
                elif 'BOD' not in features:
                    features.append('BOD')
            corpus.append(features)
        
        return corpus
