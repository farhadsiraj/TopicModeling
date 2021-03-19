#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
# Packages with tools for text processing.
from wordcloud import WordCloud
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Packages for getting data ready for and building a LDA model
import gensim
from gensim import corpora, models
from pprint import pprint
from gensim.models.coherencemodel import CoherenceModel
# Cosine similarity and clustering packages.
from sklearn.metrics.pairwise import cosine_similarity
# from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from gensim import matutils
# Network creation and visualization.
import networkx as nx
from pyvis.network import Network

# Other plotting tools.
import pyLDAvis
import pyLDAvis.gensim


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


# Set working directory.
main_dir = "C:\\Users\\Admin\\Documents\\Programming\\Python Code\\DSF\\NLP"
data_dir = main_dir + "\\project"
plot_dir = main_dir + "/plots"
os.chdir(data_dir)

# Check working directory.
print(os.getcwd())


# In[40]:


# Load corpus dataset
article_abstract = pd.read_csv("train.csv", encoding='utf8')
article_abstract.head(10)


# In[10]:


abstract = article_abstract['ABSTRACT']
print(abstract.head(10))
len(abstract)


# In[11]:


# Tokenize documents into individial words
abstract_tokenized = [word_tokenize(abstract[i]) for i in range(0,len(abstract))]

# Isolate single document to test cleaning process
abstract_words = abstract_tokenized[0]
print(abstract_words)


# In[12]:


# Convert to lower case.
abstract_words = [word.lower() for word in abstract_words]
print(abstract_words[:10])  


# In[13]:


# Get common English stop words.
stop_words = set(stopwords.words('english'))


# In[14]:


custom_word_list = ["a","about","above","after","again","against","ain","all","am","an","and","any","are","aren","aren't","as","at","be","because","been","before","being","below","between","both","but","by","can","couldn","couldn't","d","did","didn","didn't","do","does","doesn","doesn't","doing","don","don't","down","during","each","few","for","from","further","had","hadn","hadn't","has","hasn","hasn't","have","haven","haven't","having","he","her","here","hers","herself","him","himself","his","how","i","if","in","into","is","isn","isn't","it","it's","its","itself","just","ll","m","ma","me","mightn","mightn't","more","most","mustn","mustn't","my","myself","needn","needn't","no","nor","not","now","o","of","off","on","once","only","or","other","our","ours","ourselves","out","over","own","re","s","same","shan","shan't","she","she's","should","should've","shouldn","shouldn't","so","some","such","t","than","that","that'll","the","their","theirs","them","themselves","then","there","these","they","this","those","through","to","too","under","until","up","ve","very","was","wasn","wasn't","we","were","weren","weren't","what","when","where","which","while","who","whom","why","will","with","won","won't","wouldn","wouldn't","y","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","could","he'd","he'll","he's","here's","how's","i'd","i'll","i'm","i've","let's","ought","she'd","she'll","that's","there's","they'd","they'll","they're","they've","we'd","we'll","we're","we've","what's","when's","where's","who's","why's","would","able","abst","accordance","according","accordingly","across","act","actually","added","adj","affected","affecting","affects","afterwards","ah","almost","alone","along","already","also","although","always","among","amongst","announce","another","anybody","anyhow","anymore","anyone","anything","anyway","anyways","anywhere","apparently","approximately","arent","arise","around","aside","ask","asking","auth","available","away","awfully","b","back","became","become","becomes","becoming","beforehand","begin","beginning","beginnings","begins","behind","believe","beside","besides","beyond","biol","brief","briefly","c","ca","came","cannot","can't","cause","causes","certain","certainly","co","com","come","comes","contain","containing","contains","couldnt","date","different","done","downwards","due","e","ed","edu","effect","eg","eight","eighty","either","else","elsewhere","end","ending","enough","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","except","f","far","ff","fifth","first","five","fix","followed","following","follows","former","formerly","forth","found","four","furthermore","g","gave","get","gets","getting","give","given","gives","giving","go","goes","gone","got","gotten","h","happens","hardly","hed","hence","hereafter","hereby","herein","heres","hereupon","hes","hi","hid","hither","home","howbeit","however","hundred","id","ie","im","immediate","immediately","importance","important","inc","indeed","index","information","instead","invention","inward","itd","it'll","j","k","keep","keeps","kept","kg","km","know","known","knows","l","largely","last","lately","later","latter","latterly","least","less","lest","let","lets","like","liked","likely","line","little","'ll","look","looking","looks","ltd","made","mainly","make","makes","many","may","maybe","mean","means","meantime","meanwhile","merely","mg","might","million","miss","ml","moreover","mostly","mr","mrs","much","mug","must","n","na","name","namely","nay","nd","near","nearly","necessarily","necessary","need","needs","neither","never","nevertheless","new","next","nine","ninety","nobody","non","none","nonetheless","noone","normally","nos","noted","nothing","nowhere","obtain","obtained","obviously","often","oh","ok","okay","old","omitted","one","ones","onto","ord","others","otherwise","outside","overall","owing","p","page","pages","part","particular","particularly","past","per","perhaps","placed","please","plus","poorly","possible","possibly","potentially","pp","predominantly","present","previously","primarily","probably","promptly","proud","provides","put","q","que","quickly","quite","qv","r","ran","rather","rd","readily","really","recent","recently","ref","refs","regarding","regardless","regards","related","relatively","research","respectively","resulted","resulting","results","right","run","said","saw","say","saying","says","sec","section","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sent","seven","several","shall","shed","shes","show","showed","shown","showns","shows","significant","significantly","similar","similarly","since","six","slightly","somebody","somehow","someone","somethan","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specifically","specified","specify","specifying","still","stop","strongly","sub","substantially","successfully","sufficiently","suggest","sup","sure","take","taken","taking","tell","tends","th","thank","thanks","thanx","thats","that've","thence","thereafter","thereby","thered","therefore","therein","there'll","thereof","therere","theres","thereto","thereupon","there've","theyd","theyre","think","thou","though","thoughh","thousand","throug","throughout","thru","thus","til","tip","together","took","toward","towards","tried","tries","truly","try","trying","ts","twice","two","u","un","unfortunately","unless","unlike","unlikely","unto","upon","ups","us","use","used","useful","usefully","usefulness","uses","using","usually","v","value","various","'ve","via","viz","vol","vols","vs","w","want","wants","wasnt","way","wed","welcome","went","werent","whatever","what'll","whats","whence","whenever","whereafter","whereas","whereby","wherein","wheres","whereupon","wherever","whether","whim","whither","whod","whoever","whole","who'll","whomever","whos","whose","widely","willing","wish","within","without","wont","words","world","wouldnt","www","x","yes","yet","youd","youre","z","zero","a's","ain't","allow","allows","apart","appear","appreciate","appropriate","associated","best","better","c'mon","c's","cant","changes","clearly","concerning","consequently","consider","considering","corresponding","course","currently","definitely","described","despite","entirely","exactly","example","going","greetings","hello","help","hopefully","ignored","inasmuch","indicate","indicated","indicates","inner","insofar","it'd","keep","keeps","novel","presumably","reasonably","second","secondly","sensible","serious","seriously","sure","t's","third","thorough","thoroughly","three","well","wonder"]


# In[15]:


# Add additional words to stop words list.

for word in custom_word_list:
                if (word not in stop_words):
                    stop_words.add(word)
                else:
                        continue


# In[16]:


print(stop_words)


# In[17]:


# Remove stop words from text.
abstract_words = [word for word in abstract_words if not word in stop_words]
print(abstract_words[:10])


# In[18]:


# Remove punctuation and any non-alphabetical characters.
abstract_words = [word for word in abstract_words if word.isalpha()]
print(abstract_words[:10])


# In[19]:


# Stem words.
abstract_words = [PorterStemmer().stem(word) for word in abstract_words]
print(abstract_words[:10])


# In[20]:


# Create a list for clean documents.
abstract_clean = [None] * len(abstract_tokenized)
# Create a list of word counts for each clean document.
word_counts_per_abstract = [None] * len(abstract_tokenized)

# Process words in all documents.
for i in range(len(abstract_tokenized)):
# 1. Convert to lower case.
    abstract_clean[i] = [document.lower() for document in abstract_tokenized[i]]

# 2. Remove stopwords.
    abstract_clean[i] = [word for word in abstract_clean[i] if not word in stop_words]

# 3. Remove punctuation and any non-alphabetical characters.
    abstract_clean[i] = [word for word in abstract_clean[i] if word.isalpha()]

# 4. Stem words.
    abstract_clean[i] = [PorterStemmer().stem(word) for word in abstract_clean[i]]

# Record the word count per document.
    word_counts_per_abstract[i] = len(abstract_clean[i])
    


# In[21]:


len(abstract_clean)


# In[22]:


plt.hist(word_counts_per_abstract, bins = len(set(word_counts_per_abstract)))
plt.xlabel('Number of words per abstract')
plt.ylabel('Frequency')

#This shows how many words per abstract


# In[23]:


# Convert word counts list and document list to numpy arrays.
word_counts_array = np.array(word_counts_per_abstract)
abstract_array = np.array(abstract_clean)
print(len(abstract_array))


# In[24]:


project_abstract_list = [' '.join(document) for document in abstract_clean]
print(project_abstract_list[0])
print(len(project_abstract_list))


# In[25]:


# Save output file name to a variable.
out_filename = data_dir + "/clean_abstract.txt"

# Create a function that takes a list of character strings
# and a name of an output file and writes it into a txt file.
def write_lines(lines, filename):   #<- given lines to write and filename
    joined_lines = '\n'.join(lines) #<- join lines with line breaks
    file = open(out_filename, 'w', encoding="utf-8")  #<- open write only file
    file.write(joined_lines)        #<- write lines to file
    file.close()                    #<- close connection

# Write sequences to file.
write_lines(project_abstract_list, out_filename)


# In[26]:


# Initialize `CountVectorizer`.
vec = CountVectorizer()

# Transform the list of documents into DTM.
X = vec.fit_transform(project_abstract_list)
print(X.toarray()) #<- show output as a matrix


# In[27]:


# Convert the matrix into a pandas dataframe for easier manipulation.
DTM = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())
print(DTM.shape)
print(DTM.head(10))


# In[28]:


# Function that sorts and looks at first n-entries in the dictionary.
def HeadDict(dict_x, n):
    # Get items from the dictionary and sort them by
    # value key in descending (i.e. reverse) order
    sorted_x = sorted(dict_x.items(),
                reverse = True,
                key = lambda kv: kv[1])

    # Convert sorted dictionary to a list.
    dict_x_list = list(sorted_x)

    # Return the first `n` values from the dictionary only.
    return(dict(dict_x_list[:n]))


# In[29]:


# Sum frequencies of each word in all documents.
DTM.sum(axis = 0).head()

# Save series as a dictionary.
corpus_freq_dist = DTM.sum(axis = 0).to_dict()

# Look at the frequencies.
print(HeadDict(corpus_freq_dist, 10))


# In[30]:


# Save as a FreqDist object native to nltk.
corpus_freq_dist = nltk.FreqDist(corpus_freq_dist)
# Plot distribution for the entire corpus.
plt.figure(figsize = (16, 7))
corpus_freq_dist.plot(80)


# In[31]:


# Construct a word cloud from corpus.
wordcloud = WordCloud(max_font_size = 40, background_color = "black")
wordcloud = wordcloud.generate(' '.join(project_abstract_list))

# Plot the cloud using matplotlib.
plt.figure()
plt.imshow(wordcloud, interpolation = "bilinear")
plt.axis("off")
plt.show()


# In[32]:


pickle.dump(DTM, open('DTM.sav', 'wb'), protocol=4)
pickle.dump(X, open('DTM_matrix.sav', 'wb'))
pickle.dump(abstract_clean, open('abstract_clean.sav', 'wb'))
pickle.dump(project_abstract_list, open('project_abstract_list.sav', 'wb'))
pickle.dump(word_counts_array, open('word_counts_array.sav', 'wb'))


# In[33]:


processed_docs = pickle.load(open("abstract_clean.sav","rb"))  #<- the processed abstracts


# In[34]:


# Set the seed.
np.random.seed(1)

dictionary = gensim.corpora.Dictionary(processed_docs)

# Iterate through the first 10 items of the dictionary and prints out the key and value.
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 50:
        break


# In[35]:


# Use list comprehension to transform each doc within our processed_docs object.
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

# look at the 1st document.
print(bow_corpus[0])


# In[36]:


# Isolate the first document.
bow_doc_1 = bow_corpus[5]

# Iterate through each dictionary item using the index.
# Print out each actual word and how many times it appears.
for i in range(len(bow_doc_1)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_1[i][0],

        dictionary[bow_doc_1[i][0]],
        bow_doc_1[i][1]))


# In[37]:


# Initialize TF-IDF
tfidf = models.TfidfModel(bow_corpus)

# Apply to the entire corpus.
corpus_tfidf = tfidf[bow_corpus]

# Preview TF-IDF scores for the first document.
for doc in corpus_tfidf:
    pprint(doc)
    break


# In[43]:


# LDA (Latent Dirichlet allocation) model
lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics = 5, id2word = dictionary, workers = 4, passes = 25)
print(lda_model_tfidf)


# In[44]:


for idx, topic in lda_model_tfidf.print_topics(-1):
    print('Topic: {} Word: {}'.format(idx, topic))


# In[45]:


print(processed_docs[0])
for index, score in sorted(lda_model_tfidf[bow_corpus[0]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))


# In[46]:


# Compute Coherence Score using c_v.
coherence_model_lda = CoherenceModel(model = lda_model_tfidf, texts = processed_docs, dictionary = dictionary, coherence = 'c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[47]:


def compute_coherence_values(dictionary, corpus, texts, limit, start = 2, step = 3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.LdaMulticore(corpus = corpus, id2word = dictionary, num_topics = num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model = model, texts = texts, dictionary = dictionary, coherence = 'c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[48]:


model_list, coherence_values = compute_coherence_values(dictionary = dictionary, corpus = bow_corpus, texts = processed_docs, start = 2, limit = 40, step = 6)
print("Done")


# In[49]:


import matplotlib.pyplot as plt
limit = 40; start = 2; step = 6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc = 'best')
plt.show()


# In[50]:


pickle.dump(dictionary, open('dictionary.sav', 'wb')) 
pickle.dump(bow_corpus, open('bow_corpus.sav', 'wb')) 
pickle.dump(corpus_tfidf, open('corpus_tfidf.sav', 'wb')) 
pickle.dump(lda_model_tfidf, open('lda_model_tfidf.sav', 'wb'))


# In[51]:


# Load pickled data and models.
dictionary = pickle.load(open("dictionary.sav","rb"))
corpus_tfidf = pickle.load(open("corpus_tfidf.sav","rb"))
lda_model_tfidf =  pickle.load(open("lda_model_tfidf.sav","rb"))
DTM = pickle.load(open("DTM.sav","rb"))

# Load pre-saved word counts array 
word_counts_array = pickle.load(open("word_counts_array.sav","rb"))


# In[52]:


# Prepare LDA vis object by providing:
vis = pyLDAvis.gensim.prepare(lda_model_tfidf, #<- model object
                              corpus_tfidf, #<- corpus object
                              dictionary) #<- dictionary object

pyLDAvis.display(vis)


# In[53]:


pyLDAvis.save_html(vis, plot_dir+"/pyLDAvis.html")


# In[ ]:




