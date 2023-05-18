#Ngram

import nltk
from nltk.util import ngrams
from nltk.corpus import stopwords

def stop_word(text):
    stop_words = set(stopwords.words('english'))
    tokenized_text = nltk.word_tokenize(text.lower())
    filtered_text = [word for word in tokenized_text if word.isalnum() and word not in stop_words]
    return filtered_text
    


sent = "This is an example sentence to generate n-grams and remove stop words."
n = 2
sent = stop_word(sentence)

n_grams = ngrams(sent, n)
print(list(n_grams))

def generate_ngrams(text, n):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(tuple(text[i:i+n]))
    return ngrams
generate_ngrams(sent, n)

#ngram smoth
from collections import defaultdict
import math

td=['The quick brown fox jumps over the lazy dog',
    'The quick brown fox jumps over the lazy cat',
    'The quick brown fox jumps over the lazy mouse']

word=[]
for i in td:
    word+=i.split()
    
print(word)








def n_gram_count(data, n):
    n_grams = defaultdict(int)
    for i in range(len(data)-n+1):
        n_gram = tuple(data[i:i+n])
        n_grams[n_gram] += 1
    return n_grams

def add_k_smoothing(data, n, k):
    n_grams = n_gram_count(data, n)
    vocabulary_size = len(set(data))
    total_n_grams = sum(n_grams.values())
    for n_gram in n_grams:
        n_grams[n_gram] = (n_grams[n_gram] + k) / (total_n_grams + k * vocabulary_size ** n)
    return n_grams

def n_gram_smoothing(data, n, k):
    n_grams = add_k_smoothing(data, n, k)
    def probability(n_gram):
        return n_grams[n_gram]
    return probability
data = ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]

n = 2
k = 1

probability = n_gram_smoothing(word, n, k)

print(probability(("the")))
print(probability(("quick", "brown")))  
print(probability(("quick","fox"))) 


#bow fidf 

import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
pd.set_option('max_colwidth',100)
    
documents=["Gangs of Wasseypur is a great movie.", "The success of a movie depends on the performance of the actors", "There are new movie hits relaesing this week."]
print(documents)

def preprocess(document):
    'changes document to lower case and removes stopwords'
    document=document.lower()
    words=word_tokenize(document)
    words=[word for word in words if word not in stopwords.words("english")]
    document =" ".join(words)
    return document 
documents = [preprocess(document) for document in documents]
print(documents)

vectorizer=CountVectorizer()
bow_model=vectorizer.fit_transform(documents)
print(bow_model)

print(bow_model.toarray())

print(bow_model.shape)
print(vectorizer.get_feature_names())

spam=pd.read_csv("SMSSpamCollection.txt",sep="\t", names=["label","message"])
spam.head()

spam=spam.iloc[0:50,:]
print(spam)

messages=spam.message
print(messages)

messages= [message for message in messages]
print(messages)

messages=[preprocess(message)for message in messages]
print(messages)

vectorizer=CountVectorizer()
bow_model = vectorizer.fit_transform(messages)
print(bow_model.toarray())

print(bow_model.shape)
print(vectorizer.get_feature_names())

#lot of duplicate are visible
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer=PorterStemmer()
wordnet_lemmatizer=WordNetLemmatizer()

def preprocess(document,stem=True):
    document=document.lower()
    words=word_tokenize(document)
    words= [word for word in words if word not in stopwords.words("english")]
    if stem: 
        words=[stemmer.stem(word)for word in words]
    else:
        words=[wordnet_lemmatizer.lemmatize(word, pos='v')for word in words]
    document="".join(words)
    return document

#bow model on stemmer messages
messages=[preprocess(message, stem=True)for message in spam.message]
vectorizer=CountVectorizer()
bow_model=vectorizer.fit_transform(messages)
pd.DataFrame(bow_model.toarray(),columns=vectorizer.get_feature_names())
print(vectorizer.get_feature_names())

#lemmatizing the messages
messages=[preprocess(message, stem=False)for message in spam.message]
vectorizer=CountVectorizer()
bow_model=vectorizer.fit_transform(messages)
pd.DataFrame(bow_model.toarray(),columns=vectorizer.get_feature_names())
print(vectorizer.get_feature_names())

def preprocess(document):
    'changes document to lower case and removes stopwords'
    document=document.lower()
    words=word_tokenize(document)
    words=[word for word in words if word not in stopwords.words("english")]
    document =" ".join(words)
    return document 
documents = [preprocess(document) for document in documents]
print(documents)

vectorizer=TfidfVectorizer()
tfidf_model=vectorizer.fit_transform(documents)
print(tfidf_model)

print(tfidf_model.toarray())

pd.DataFrame(tfidf_model.toarray(), columns=vectorizer.get_feature_names())

#creating tfidf model for spam classification
spam=pd.read_csv("SMSSpamCollection.txt", sep="\t",names=["label","message"])
spam.head()

spam=spam.iloc[0:50,:]
print(spam)

#extract the mmessages from the dataframe
messages=[message for message in spam.message]
print(messages)

messages=[preprocess(message) for message in messages]
print(messages)

vectorizer=TfidfVectorizer()
tfidf_model=vectorizer.fit_transform(messages)
print(tfidf_model.toarray())

print(tfidf_model.shape)
print(vectorizer.get_feature_names())



#6
#import libraries
import nltk, re,pprint
import numpy as np
import pandas as pd
import requests 
import matplotlib.pyplot as plt
import seaborn as sns
import pprint, time
import random
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize

import nltk
nltk.download('treebank')

#reading the treebank tagged sentences
wsj=list(nltk.corpus.treebank.tagged_sents())

print(wsj[:40])

random.seed(1234)
train_set, test_set = train_test_split(wsj,test_size=0.3)
print(len(train_set))
print(len(test_set))
print(train_set[:40])

#getting list of tagged words
train_tagged_words=[tup for sent in train_set for tup in sent]
len(train_tagged_words)

#tokens
tokens=[pair[0] for pair in train_tagged_words]
tokens[:10]

#vocabulary
V=set(tokens)
print(len(V))

#number of tags
T=set([pair[1] for pair in train_tagged_words])
len(T)

print(T)

#emission probability
#computing P(w/t) amd starting in T x V matrix
t=len(T)
v=len(V)
w_given_t= np.zeros((t,v))

#compute word given tag: Emission Probablility 
def word_given_tag(word, tag, train_bag = train_tagged_words):
    tag_list =[pair for pair in train_bag if pair[1]==tag]
    count_tag=len(tag_list)
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)

#examples
#large
print("\n", "large")
print(word_given_tag('large', 'JJ'))
print(word_given_tag('large', 'VB'))
print(word_given_tag('large', 'NN'),"\n")
#will
print("\n", "will")
print(word_given_tag('will', 'MD'))
print(word_given_tag('will', 'NN'))
print(word_given_tag('will', 'VB'),"\n")
#BOOK
print("\n", "book")
print(word_given_tag('book', 'NN'))
print(word_given_tag('book', 'VB'))

#transition probability
#compute tag given tag: tag2(t2) given tag1(t1), i.e. Transition Probablity
def t2_given_t1(t2,t1, train_bag=train_tagged_words):
    tags=[pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 =0;
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1]==t2:
            count_t2_t1 +=1
    return (count_t2_t1, count_t1)

#examples
print(t2_given_t1(t2='NNP', t1='JJ'))
print(t2_given_t1('NN','JJ'))
print(t2_given_t1('NN','DT'))
print(t2_given_t1('NNP','VB'))
print(t2_given_t1(',','NNP'))
print(t2_given_t1('PRP','PRP'))
print(t2_given_t1('VBG','NNP'))

#PLEASE NOTE p(TAG/START) IS SAME AS P(TAG/'.')
print(t2_given_t1('DT','.'))
print(t2_given_t1('VBG','.'))
print(t2_given_t1('NN','.'))
print(t2_given_t1('NNP','.'))

#CREATING TXT transition matrix of tags
#each column is t2, each row is t1
# thus M(i,j) represents P(tj given ti)
tags_matrix = np.zeros((len(T),len(T)),dtype='float32')
for i, t1 in enumerate(list(T)):
    for j, t2 in enumerate(list(T)):
        tags_matrix[i,j]=t2_given_t1(t2,t1)[0]/t2_given_t1(t2,t1)[1]
        
tags_matrix

tags_df= pd.DataFrame(tags_matrix, columns=list(T), index=list(T))

tags_df

tags_df.loc['.', :]

#heatmap of tags matrix
#T(i,j) means P(tag j given tag i)
plt.figure(figsize=(18,12))
sns.heatmap(tags_df)
plt.show()

#frequent tags
#filter the df to get P(t2,t1)>0.5
tags_frequent = tags_df[tags_df>0.5]
plt.figure(figsize=(18,12))
sns.heatmap

#8
import nltk
nltk.download('averaged_perceptron_tagger')

import nltk
import spacy
from textblob import TextBlob

# Function to apply TextBlob POS tagging
def apply_textblob(text):
    blob = TextBlob(text)
    return blob.tags

# Function to apply spaCy POS tagging
def apply_spacy(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    return [(token.text, token.pos_) for token in doc]

# Function to apply NLTK POS tagging
def apply_nltk(text):
    tokens = nltk.word_tokenize(text)
    return nltk.pos_tag(tokens)

# Main program
user_input = input("Enter a sentence: ")

print("Select the POS tagger:")
print("1. TextBlob")
print("2. spaCy")
print("3. NLTK")

choice = input("Enter your choice (1, 2, or 3): ")

if choice == "1":
    tags = apply_textblob(user_input)
elif choice == "2":
    tags = apply_spacy(user_input)
elif choice == "3":
    tags = apply_nltk(user_input)
else:
    print("Invalid choice!")
    exit()

print("POS Tags:")
for token, tag in tags:
    print(f"{token}: {tag}")

    
#9
import nltk

# Function to implement chunking and generate a parse tree
def apply_chunking(text):
    # Define the grammar for chunking
    grammar = r"""
    NP: {<DT|JJ|NN.*>+}    # Chunk sequences of DT, JJ, NN
    PP: {<IN><NP>}         # Chunk prepositions followed by NP
    VP: {<VB.*><NP|PP>}    # Chunk verbs followed by NP or PP
    """

    # Create a parser based on the grammar
    parser = nltk.RegexpParser(grammar)

    # Tokenize the input text
    tokens = nltk.word_tokenize(text)

    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Apply chunking and generate the parse tree
    parse_tree = parser.parse(pos_tags)

    # Draw the parse tree
    parse_tree.draw()

# Main program
user_input = input("Enter a sentence: ")
apply_chunking(user_input)
