import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation,TruncatedSVD
import matplotlib.pyplot as plt
from gensim.models import LdaModel
from gensim.models.phrases import Phrases,Phraser
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
import random
import time
from multiprocessing import Pool, cpu_count, get_context

def preprocess_parallel(dataset, stop_words):
    # Create pool for all functions with parallel processing. All cores except 1
    pool = Pool(processes=cpu_count()-1)
    results = pool.starmap(preprocess, [(doc, stop_words) for doc in dataset])
    pool.close()
    pool.join()
    return results

def compute_optimal_topics_lda_parallel(tokens, dictionary, corpus, start, limit, step):
    # Create pool for all functions with parallel processing. All cores except 1
    pool = Pool(processes=cpu_count()-1)
    coherence_values = pool.starmap(compute_optimal_topics_lda,
        [(tokens, dictionary, corpus, num_topics) for num_topics in range(start, limit + 1, step)]
        )

    x = list(range(start, limit + 1, step))
    #Return the number of topics with the highest coherence value
    return x[coherence_values.index(max(coherence_values))]
    
def remove_urls(dataset: str):
    # Regex pattern to match URLs (http, https, www)
    url_pattern = r'http\S+|www\.\S+'
    return re.sub(url_pattern, '', dataset)

def preprocess (dataset,stop_words):
    #Initialize Lemmatizer
    lemmatizer = WordNetLemmatizer()
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
    #joins dicts to strings
    texts = [" ".join(doc) for doc in docs]
    #Creates a CountVectorizer object. Tokenizes the text and builds a vocabulary. Catch Bigrams via n-grams
    vect = CountVectorizer(ngram_range=(1,2), min_df=2)
    #Learns the vocabulary and converts the texts into a matrix
    X = vect.fit_transform(texts)
    return vect, X

def newgensimbowmatrix(docs: list, min_df: int = 2):
    # Detect bigrams (ngram_range=(1,2))
    bigram = Phrases(docs, min_count=min_df, delimiter='_')  # join tokens with "_"
    bigram_mod = Phraser(bigram)

    # Apply bigram model to all docs
    tokens_with_bigrams = [bigram_mod[doc] for doc in docs]

    # Build dictionary
    dictionary = Dictionary(tokens_with_bigrams)

    # Filter out tokens/bigrams that appear in fewer than `min_df` documents
    dictionary.filter_extremes(no_below=min_df, no_above=1.0)

    # Build corpus
    corpus = [dictionary.doc2bow(doc) for doc in tokens_with_bigrams]

    return dictionary, corpus, tokens_with_bigrams

def compute_optimal_topics_lda(tokens, dictionary, corpus, num_topics):
    """Compute coherence scores for Gensim LDA and return best topic number."""
    lda_model = LdaModel(corpus=corpus, id2word=dictionary,
                            num_topics=num_topics, random_state=42, passes=10)
    #processes=1 required so no damonic sub process exception occurs, becuase parallelization was implemented before
    coherencemodel = CoherenceModel(model=lda_model, texts=tokens,
                                    dictionary=dictionary, coherence='c_v',processes=1)
    coherence = coherencemodel.get_coherence()
    print(f"LDA | Topics = {num_topics}, Coherence = {coherence:.4f}")
    return(coherence)

def ldatopicextraction(texts, dictionary, corpus, topiccount, relevanttopic):
    # Train LDA model
    lda = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=topiccount,
        random_state=42,
        passes=10,
    )

    # Print top words per topic
    print("\n=== Top words per topic ===")
    for idx, terms in lda.show_topics(num_topics=topiccount, num_words=relevanttopic, formatted=False):
        words = [word for word, prob in terms]
        print(f"Topic {idx}: {', '.join(words)}")

    """
    # Print dominant topic for each document
    print("\n=== Dominant topic per document ===")
    for i, bow in enumerate(corpus):
        topic_dist = lda.get_document_topics(bow)
        topic_id, prob = max(topic_dist, key=lambda x: x[1])
        print(f"Document {i}: Dominant Topic {topic_id} → {' '.join(texts[i])}")
    """

def main():
    #variable initializationn
    complaints = []
    tokens = []
    vectormethod = ""
    topicmethod = ""
    #Words that appear often but do not add any value to the analysis
    customstopwords = ("xxxx","xxxxxxxx","account")

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


    start = time.perf_counter()
    with open(jsonfile, "r") as f:
        data = json.load(f)
    end = time.perf_counter()
    print(f"Read JSON took {end - start:.4f} seconds")

    #If the complaint_what_happened field has been filled out in the complaint add the complaint to the complaints list
    start = time.perf_counter()
    for obj in data:
        if len(obj["complaint_what_happened"]) != 0:
            complaints.append(obj["complaint_what_happened"])
    end = time.perf_counter()
    print(f"Extract complaints took {end - start:.4f} seconds")

    start = time.perf_counter()
    tokens = preprocess_parallel(complaints,stop_words)
    end = time.perf_counter()
    print(f"Preprocessing complaints took {end - start:.4f} seconds")

    start = time.perf_counter()
    if vectormethod == 1:
        if topicmethod == 2:
            dict, corp, tokens_with_bigrams = newgensimbowmatrix(tokens)
        else:
            vectorizer, X = newbowmatrix(tokens)
    end = time.perf_counter()
    print(f"Vectorization took {end - start:.4f} seconds")

    
    if topicmethod == 2:
        start = time.perf_counter()
        besttopiccount = compute_optimal_topics_lda_parallel(tokens_with_bigrams,dict,corp,1,23,1)
        end = time.perf_counter()
        print(f"compute_optimal_topics_lda took {end - start:.4f} seconds")
        start = time.perf_counter()
        ldatopicextraction(tokens_with_bigrams, dict, corp, besttopiccount, 10)
        end = time.perf_counter()
        print(f"LDA topic extraction took {end - start:.4f} seconds")
    

if __name__ == "__main__":
    main()