import pandas as pd
import spacy
nlp= spacy.load("en_core_web_sm")#spacy kütüphanesinin ingilizce dil modeli
content="we are live in İstanbul and I work at Amazon. we are so happy at 2 pm"
doc=nlp(content)
for ent in doc.ents:
    print(ent.text,ent.label_)
entities= [(ent.text,ent.label_,ent.lemma_) for ent in doc.ents]
df= pd.DataFrame(entities,columns=["text","type","lemma"])
