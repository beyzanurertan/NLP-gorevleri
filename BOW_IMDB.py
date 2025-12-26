import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords

# English stopword list
english_stop_words = stopwords.words('english')

df= pd.read_csv("IMDB Dataset.csv")

documents= df["review"]
labels= df["sentiment"]

def clean_text(text):
    
    text= text.lower() #küçük harf yapma
    
    text=re.sub(r"\d+", "", text) #sayıları temizlenmesi
    text=re.sub(r"[^\w\s]", "", text) #diğer karakterleri temizlenmesi
    text= " ".join([word for word in text.split() if len(word) > 2])
    
    return text

clean_doc = [clean_text(row) for row in documents]

vectorizer= CountVectorizer(stop_words=english_stop_words)
x = vectorizer.fit_transform(clean_doc[:75])
feature_names= vectorizer.get_feature_names_out()
vektor_temsili2= x.toarray()
print(f"vektör temsili: {vektor_temsili2}")
df_bow= pd.DataFrame(vektor_temsili2, columns=feature_names)
word_counts= x.sum(axis=0).A1
word_frequens= dict(zip(feature_names,word_counts))
most_common_5_words= Counter(word_frequens).most_common(5)