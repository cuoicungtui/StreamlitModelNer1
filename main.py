import streamlit as st
from annotated_text import annotated_text
# from model.model_Ner1.nermodel1 import text_to_token
from model.model_Ner1.nermodel2 import text_to_token
from model.model_Ner1.database.connect_mongo import dataMongo
import streamlit as st
import pandas as pd
from io import StringIO
import os
from dotenv import dotenv_values

# Đọc các biến môi trường từ file .env
env_variables = dotenv_values('.env')


st.title('Medical Named Entity Recognition')
# text = st.text_input('Nhập câu nói ', '', max_chars=3000)
tab1, tab2 = st.tabs(['EXAMPLE', 'PREDICT'])

with tab1:
    st.write('B-Disease: Từ bắt đầu bệnh '+'I-Disease: Từ thuộc bệnh')
    st.write('B-DRUG: Từ bắt đầu thuốc '+'I-DRUG: Từ thuộc thuốc')
    st.write('Các câu ví dụ')
    st.write('Severe stomach upset after prolonged use.')
    list_ex = [
        ('Severe', 'B-Disease'),
        ('stomach', 'I-Disease'),
        ('upset', 'I-Disease'),
        'after',
        'prolonged',
        'use',
        '.'
    ]
    annotated_text(*list_ex)
    st.write('Bệnh được xác định: Severe stomach upset')

with tab2:
    text =  st.text_area('Enter your text', '', height=200)
    # path_atlas = env_variables['SRV_PY_MONGO']
    # datamongo = dataMongo(path_atlas)
    # if len(text) > 0:
    #     datamongo.insert(text)
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
