import streamlit as st

from annotated_text import annotated_text
from model.model_Ner1.database.connect_mongo import dataMongo
from model.model_Ner1.nermodel1 import text_to_token
import streamlit as st
import pandas as pd
from io import StringIO
import os

st.title('Medical Named Entity Recognition')
text = st.text_input('Nhập câu nói ', '', max_chars=3000)

path_atlas = 'mongodb+srv://cuoicungtui:6V0lb3R2MnFKH6op@cluster0.kte4zsw.mongodb.net/'
dataMongo = dataMongo(path_atlas)
dataMongo.insert(text)

y_predicts =  text_to_token([text])
y_predicts_list = []
for y_pre in y_predicts['taget']:
    y_predicts_list.extend(y_pre)
list1 = []
for i in range(len(y_predicts_list)):
    if(len(y_predicts_list[i].split()) > 1 ):
        if(y_predicts_list[i].split()[1] != 'O' ):
            list1.append((y_predicts_list[i].split()[0],y_predicts_list[i].split()[1]))
        else:
            list1.append(y_predicts_list[i].split()[0])
    else:
        list1.append(y_predicts_list[i].split()[0])
    list1.append(" ")

annotated_text(*list1)
