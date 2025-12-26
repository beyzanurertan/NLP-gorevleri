import nltk
nltk.download("wordnet")
from nltk.stem import PorterStemmer
stemmer= PorterStemmer()
words=["writing", "writter", "drink", "drunk", "eat", "ate"]
stems= [stemmer.stem(w) for w in words]
print(f"stems:{stems}")

# %% lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer= WordNetLemmatizer()
words=["writing", "writter", "drink", "drunk", "eat", "ate"]
lemmas=[lemmatizer.lemmatize(w, pos="v") for w in words]
print(f"lemmas:{lemmas}")