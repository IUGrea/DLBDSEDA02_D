import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

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

complaints = []
customstopwords = ("xxxx","xxxxxxxx")

print("Enter Path to input JSON")
jsonfile = input()
jsonfile = jsonfile.replace('"', '').replace("'", "")

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

for obj in complaints:
    print(preprocess(obj))
    print("")