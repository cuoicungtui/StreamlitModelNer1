import streamlit as st

from annotated_text import annotated_text

from model.model_Ner1.nermodel1 import text_to_token
import streamlit as st
import pandas as pd
from io import StringIO
import os

# check = False
# file_list = os.listdir('./model/model_Ner1/modelsave/')
# if len(file_list) > 1:
#     check = True
#     file_name = file_list[1]
# st.title('Medical NER')
# if check == False:
uploaded_file = st.file_uploader("Load_model",type=["h5"])

if uploaded_file is not None :
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    file_name = uploaded_file.name
    save_path = os.path.join("./model/model_Ner1/modelsave/", file_name)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File '{file_name}' is saved successfully!")
    path = './model/model_Ner1/modelsave/'+file_name
    st.write('Path: ',path)
    text = st.text_input('Nhập văn bản', '', max_chars=3000)
    y_predicts =  text_to_token([text],path)
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
