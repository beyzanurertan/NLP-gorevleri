import spacy
nlp=spacy.load("en_core_web_sm")
word= "Natural Language Processing is a specialized field of artificial intelligence that enables computers to understand and generate human language. Modern systems often use deep learning architectures, such as Transformers, to analyze complex patterns in text data. This technology powers many applications we use daily, including machine translation, chatbots, and sentiment analysis. As algorithms improve, machines are becoming better at grasping the subtle nuances and context of human communication. "

#kelimeyi nlp işleminden geçir
doc=nlp(word)
for token in doc:
    print(f"text:{token.text}")
    print(f"lemma:{token.lemma_}")#kelimenin kökü
    print(f"pos:{token.pos_}")#dilbilgisel özelliği
    print(f"tag:{token.tag_}")#detaylı dilbilgisel özelliği
    print(f"dependency:{token.dep_}")#kelimenin cümledeki rolü özne yüklem vb
    print(f"shape:{token.shape_}")#kelimenin karakter sayısı
    print(f"is alpha:{token.is_alpha}")#kelimenin yalnızca alfabetik karakterlerden oluşup oluşmadığı
    print(f"is stop:{token.is_stop}")#kelimenin stopwords olup olmadığı
    print(f"morfoloji:{token.morph}")#kelimenin morfolojisi
    print(f"is plural:{'Number=Plur' in token.morph}")#kelimenin çoğul olup olmadığı
    print()