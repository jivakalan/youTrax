from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus.reader.util import *
from nltk.corpus.reader.plaintext import PlaintextCorpusReader as PCR
from nltk.stem.wordnet import WordNetLemmatizer as WNL
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim
import itertools
from wordcloud import WordCloud
from gensim import corpora
from gensim.models.wrappers import LdaVowpalWabbit
import string


class Contract_Reader(): 
    def __init__(self, config): 
        print('Filepath for texts = ', config.textpath)
        self.corpus = PCR(config.textpath, '.*\.txt', encoding='utf-16', para_block_reader=read_line_block)
        if config.clean_paragraphs=='yes':
           self.clean(config, mode='para')
        if config.clean_sentences=='yes':
           self.clean(config, mode='sent')
        #Corpus summaries
        self.corpus_info()
        self.LDA(config.num_topics, config.num_words)
        self.plot(config.num_words)
    def clean(self, config, mode='sent'):
        stop = set(stopwords.words('english'))
        exclude = set(string.punctuation)
        lemma = WNL()
        if mode=='para':
           #paragraphs are lists of sentences each of which is a list of tokens. Reducing to list of strings.
           self.para_list = [list(itertools.chain.from_iterable(para)) for para in self.corpus.paras()]
           for index,paragraph in enumerate(self.para_list):
                 paragraph = " ".join(paragraph)
                 stop_free = " ".join([i for i in paragraph.lower().split() if i not in stop])
                 punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
                 normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split()) 
                 self.para_list[index] = normalized
           print(self.para_list[0])
           self.para_list = [para.split() for para in self.para_list] 
           print(self.para_list[0])
        if mode=='sent':
           #Obtain list of strings each one a sentence rather than list of lists. 
           self.sents_list = [" ".join(sent) for sent in self.corpus.sents()]
           for index,sentence in enumerate(self.sents_list):
                 stop_free = " ".join([i for i in sentence.lower().split() if i not in stop])
                 punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
                 normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
                 self.sents_list[index] = normalized
           print(self.sents_list[0])
           self.sents_list = [sentence.split() for sentence in self.sents_list] 
           print(self.sents_list[0])
    def LDA(self, num_topics, num_words): 
        dictionary=corpora.Dictionary(self.para_list)
        doc_term_matrix = [dictionary.doc2bow(para) for para in self.para_list]
        path = '/mnt/APPDATA/Project_Mafia/omkhalil/vowpal_binaries/vw-7.20150623'
        self.ldamodel = LdaVowpalWabbit(path, doc_term_matrix, num_topics=num_topics, id2word = dictionary)
        self.ldamodel.save('model/lda_model')
        print(self.ldamodel.print_topics(num_topics=10, num_words=num_words))
            
    def plot(self, num_words): 
        for t in range(self.ldamodel.num_topics):
            plt.figure()
            tuples = [reversed(x) for x in self.ldamodel.show_topic(t,num_words)]
            plt.imshow(WordCloud().fit_words(dict(tuples)))
            plt.axis("off")
            plt.title("Topic #" + str(t))
            plt.savefig('plots/topic'+str(t))             
    def corpus_info(self):
        """
        Summary information about the status of a corpus.
        """
        fids   = len(self.corpus.fileids())
        paras  = len(self.corpus.paras())
        sents  = len(self.corpus.sents())
        sperp  = sum(len(para) for para in self.corpus.paras()) / float(paras)
        tokens = FreqDist(self.corpus.words())
        count  = sum(tokens.values())
        vocab  = len(tokens)
        lexdiv = float(count) / float(vocab)

        print((
            "Text corpus contains {} files\n"
            "Composed of {} paragraphs and {} sentences.\n"
            "{:0.3f} sentences per paragraph\n"
            "Word count of {} with a vocabulary of {}\n"
            "lexical diversity is {:0.3f}"
        ).format(
            fids, paras, sents, sperp, count, vocab, lexdiv
        ))
    

class Config(): 
   def __init__(self):
      self.textpath='/mnt/APPDATA/Project_Mafia/omkhalil/TEXT/0001'
      self.clean_paragraphs='yes'
      self.clean_sentences='no'
      self.num_words=10
      self.num_topics=10

if __name__=="__main__":  
   config = Config()
   reader = Contract_Reader(config)
   
  
