from sklearn.feature_extraction.text import CountVectorizer



documents= ["Bu bir bilgisayar yazılımıdır.",
            "Bu çalışma bir doğal dil işleme çalışmasıdır."]

vectorizer_unigram= CountVectorizer(ngram_range=(1,1))
vectorizer_bigram= CountVectorizer(ngram_range=(2,2))
vectorizer_trigram= CountVectorizer(ngram_range=(3,3))

x_unigram= vectorizer_unigram.fit_transform(documents)
unigram_features= vectorizer_unigram.get_feature_names_out()
 
x_bigram= vectorizer_bigram.fit_transform(documents)
bigram_features= vectorizer_bigram.get_feature_names_out()
 
x_trigram= vectorizer_trigram.fit_transform(documents)
trigram_features= vectorizer_trigram.get_feature_names_out()
 
print(f"uni: {unigram_features}")
print(f"bi: {bigram_features}")
print(f"tri: {trigram_features}")