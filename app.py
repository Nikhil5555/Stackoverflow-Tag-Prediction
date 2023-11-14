import streamlit as st
import pickle
import os
path = os.getcwd()
os.chdir('c:\\Users\\nikhi\\Desktop\\project')
tfidf = pickle.load(open("voca.pickle", "rb"))
multilabel = pickle.load(open("labels.pickle", "rb"))
st.title("Stackoverflow Tag Prediction")

user_input = st.text_input("Enter Question")
filename = "finalized_model.sav"
loaded_model = pickle.load(open(filename, 'rb'))
x = [user_input]
xt = tfidf.transform(x)
# loaded_model.predict(xt)
ans = multilabel.inverse_transform(loaded_model.predict(xt))
res = ""
for i in ans:
    res = " #".join(i)

st.subheader('#'+res)