import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt

def remove_urls(dataset: str):
    # Regex pattern to match URLs (http, https, www)
    url_pattern = r'http\S+|www\.\S+'
    return re.sub(url_pattern, '', dataset)

def preprocess (dataset: str):
    def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
        def get_wordnet_pos(word):
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {"J": wordnet.ADJ,
                        "N": wordnet.NOUN,
                        "V": wordnet.VERB,
                        "R": wordnet.ADV}
            return tag_dict.get(tag, wordnet.NOUN)
        
        # lemmatize tokens
        lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
            
        # return lemmatized tokens
        return lemmas

    #Convert to lowercase
    preprocdataset = dataset.lower()
    #Remove URLs from dataset
    preprocdataset = remove_urls(preprocdataset)
    #Remove special chars and digits
    preprocdataset = re.sub(r'[^A-Za-z\s]', '', preprocdataset)
    #Tokenize string
    preprocdataset = word_tokenize(preprocdataset)
    #Remove Stopwords
    preprocdataset = [word for word in preprocdataset if word not in stop_words]
    #Lemmatization
    preprocdataset = lemmatize_tokens(preprocdataset)
    return preprocdataset

def newbowmatrix (docs: list):
    texts = [" ".join(doc) for doc in docs]
    vect = CountVectorizer()
    X = vect.fit_transform(texts)
    return vect, X

def newtfidfmatrix(docs: list):
    texts = [" ".join(doc) for doc in docs]
    vect = TfidfVectorizer()
    X = vect.fit_transform(texts)
    return vect, X

def ldatopicextraction (vect,matrix,texts,topiccount,relevanttopic):
    def print_topics(feature_names, n_top_words):
        for idx, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-n_top_words:]]
            print(f"Topic {idx}: {', '.join(top_words)}")
    lda = LatentDirichletAllocation(n_components=topiccount, random_state=42)
    lda.fit(matrix)
    feature_names = vect.get_feature_names_out()
    
    doc_topic_dist = lda.transform(X)
    for i, complaint in enumerate(texts):
        topic_id = doc_topic_dist[i].argmax()
        print(f"Complaint: '{complaint}' → Topic {topic_id}")


    print_topics(feature_names, relevanttopic)

def lsatopicextraction(vect, matrix, texts, topiccount=5, n_top_words=8):
    lsa = TruncatedSVD(n_components=topiccount, random_state=42)
    lsa.fit(matrix)
    feature_names = vect.get_feature_names_out()
    
    for idx, topic in enumerate(lsa.components_):
        top_words = [feature_names[i] for i in topic.argsort()[-n_top_words:][::-1]]
        print(f"Topic {idx}: {', '.join(top_words)}")

complaints = []
tokens = []
vectormethod = ""
topicmethod = ""
customstopwords = ("xxxx","xxxxxxxx")

print("Enter Path to input JSON")
jsonfile = input()
jsonfile = jsonfile.replace('"', '').replace("'", "")


while vectormethod != 1 and vectormethod != 2:
    print("Choose Vectorization method: 1=Bag of Words (Bow),2=Term Frequency-Inverse Document Frequency (TF-IDF)")
    vectormethod = int(input())

while topicmethod != 1 and topicmethod != 2:
    print("Choose topic extraction method: 1=Latent Semantic Analysis (LSA),2=Latent Dirichlet Allocation (LDA)")
    topicmethod = int(input())    

# Download NLTK Data and initialize
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger_eng')
stop_words = set(stopwords.words('english'))
for stopword in customstopwords:
    stop_words.add(stopword)

lemmatizer = WordNetLemmatizer()

with open(jsonfile, "r") as f:
    data = json.load(f)

for obj in data:
    if len(obj["complaint_what_happened"]) != 0:
        complaints.append(obj["complaint_what_happened"])

for complaint in complaints:
    tokens.append(preprocess(complaint))

if vectormethod == 1:
    vectorizer, X = newbowmatrix(tokens)
if vectormethod == 2:
    vectorizer, X = newtfidfmatrix(tokens)

if topicmethod == 1:
    lsatopicextraction(vectorizer,X,tokens,6,8)
if topicmethod == 2:
    ldatopicextraction(vectorizer,X,tokens,6,8)
