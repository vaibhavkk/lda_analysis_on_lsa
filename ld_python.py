

from os import walk
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora, models
import gensim
from nltk.stem import PorterStemmer
tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()
    
# create sample documents
#/home/administrator/Desktop/Link to doc study/Stage3/code/LSA_python/dataset_bbc
bbcdata = load_files("/home/administrator/doc_study/Stage3/code/LSA_python/20_newsgroups"#) 
		,                description = None, load_content = True,
                encoding='latin1', 
		decode_error='strict', shuffle=True,
                random_state=42)
#print ("BBC data loaded")

i = 0
#print (type(bbcdata))
#print (bbcdata)
#for file in bbcdata.filenames :
#	i = i+1
#print (i)

#doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
#doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
#doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
#doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
#doc_e = "Health professionals say that brocolli is good for your health." 
#print bbcdata.filenames
# compile sample documents into a list
#doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
doc_set = bbcdata.data
# list for tokenized documents in loop
texts = []

# loop through document list
for i in doc_set:
    
    # clean and tokenize document string
    raw = i.lower()
    print "lower case done"
    tokens = tokenizer.tokenize(raw)
    print "tokens created"
    # remove stop words from tokens
    stopped_tokens = [i for i in tokens if not i in en_stop]
    print "stop words removed"
    # stem tokenstype
    stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
    print "tokens stemmed"
    # add tokens to list
    texts.append(stemmed_tokens)
    print "tokens added to list"
# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)
print "dictionary created"
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text1) for text1 in texts]

print "The is courpus \n __________________________________\n"
#print (corpus)
stemmer = PorterStemmer()
class StemmedCountVectorizer(CountVectorizer):
		def build_analyzer(self):
			analyzer = super(StemmedCountVectorizer,self).build_analyzer()
			return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

count_vectorizer = StemmedCountVectorizer(min_df=3, analyzer="word", stop_words=text.ENGLISH_STOP_WORDS)

frequency_matrix = count_vectorizer.fit_transform(bbcdata.data)


#For TF-IDF calculation 
tfidf = TfidfTransformer()
print "tfidf variable created"
tfidf_matrix = tfidf.fit_transform(frequency_matrix)
print "TFIDF matrix created"
#print tfidf_matrix


# generate LDA model make changes in topic according to dataset
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=20, id2word = dictionary, passes=20)

# get probability distribution for a document 
print(ldamodel.print_topics())
print ldamodel.num_terms
print ldamodel.num_topics
print(ldamodel.print_topics(num_topics=3, num_words=3))

#write topics in file along with equation 
t = ldamodel.print_topics()
'''
for tup in t :
	for eq in tup :
		print eq
#or use this
with open('topics.csv','w') as fp :
    fp.write('\n'.join('%s,%s' % x for x in t))
fp.close()
'''
# or this

fp = open('topics.csv','w')
for tup in t :
    fp.write(str(tup))
fp.close()
