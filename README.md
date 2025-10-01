Install the requirements: pip install -r requirements.txt

After execution the program will request the following input from you:
 - Path to input JSON
 - Choose Vectorization method: 1=Bag of Words (Bow),2=Term Frequency-Inverse Document Frequency (TF-IDF)
 - Choose topic extraction method: 1=Latent Semantic Analysis (LSA),2=Latent Dirichlet Allocation (LDA)
 - How many words should be shown per topic? (default = 10)
 - How many random complaints shall be extracted from the JSON document (default = All)

The code has been tested using Python 3.11.9 on Windows
