from gensim.corpora.wikicorpus import  WikiCorpus
from gensim.models.phrases import Phraser, Phrases
from gensim.models.word2vec import Word2Vec

wiki = WikiCorpus('ruwiki-20190720-pages-articles-multistream.xml.bz2', dictionary = False)
bigram  = Phrases(wiki.get_texts())
bigram_transformer = Phraser(bigram)

def text_generator_bigram():
    for text in wiki.get_texts():
        yield bigram_transformer[[word.encode('utf-8') for word in text]]

print(text_generator_bigram[1:5])
trigram = Phrases(text_generator_bigram())
trigram_transformer = Phrase(trigram)

def text_generator_trigram():
    for text in wiki.get_texts():
        yield trigram_transformer[bigram_transformer[ [word.decode('utf-8') for word in text] ] ]

model = Word2Vec(size = 100, window = 7, min_count = 10m workers = 10)
model.build_vocab(text_generator_trigram())
model.train(text_generator_trigram())