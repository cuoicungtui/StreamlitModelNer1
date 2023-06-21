from transformers import BertTokenizerFast
import tensorflow as tf
import pickle
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.losses import Loss
import tensorflow as tf
import numpy as np
from dotenv import dotenv_values

# Đọc các biến môi trường từ file .env
env_variables = dotenv_values('.env')

n_tags = int(env_variables['N_TAGs'])

class CustomCrossEntropy(Loss):
    def __init__(self,reduction=keras.losses.Reduction.AUTO ,name='custom_cross_entropy'):
        super(CustomCrossEntropy, self).__init__(reduction=reduction,name=name)

    @tf.function
    def call(self, y_true, y_pred):    
        sum_loss = 0.0
        for i in range(len(y_true)):
            y_true_clean = y_true[i][y_true[i] != -100]
            y_pred_clean = y_pred[i][y_true[i] != -100]
            y_true_one_hot = tf.one_hot(tf.cast(y_true_clean, tf.int32), depth=n_tags)
            loss =tf.reduce_sum(tf.keras.losses.categorical_crossentropy(y_true_one_hot, y_pred_clean)) 
            sum_loss+=loss
            
        return sum_loss/tf.cast(len(y_true), dtype=tf.float32) 


def word_id_pad(word_id):
    label_id = []

    for wordid in word_id:
        if wordid is None:
            label_id.append(-100)
        else:
            label_id.append(wordid)
            
    lb = tf.convert_to_tensor(np.array(label_id))
    return lb



def load_model_json(path_model_json,path_weight_h5):

    # load model json
    with open(path_model_json, 'r') as f:
        model_json = f.read()
        loaded_model = keras.models.model_from_json(model_json)

    # load weights into new model
    loaded_model.load_weights(path_weight_h5)

    return loaded_model


def text_to_token(text):


    path_model_json = env_variables['PATH_MODEL_JSON']
    path_weight_h5  = env_variables['PATH_WEIGHT_H5']

    model = load_model_json(path_model_json,path_weight_h5)

    path_tokenizer = env_variables['PATH_TOKENIZER']

    with open(path_tokenizer, 'rb') as pickle_file:
        tokenizer = pickle.load(pickle_file)

    texts = text[0].split('.')
    max_len = 100
    Sequences = ['<START> '+i+' <END>' for i in texts]     
    Sequences_id = tokenizer.texts_to_sequences(Sequences)
    vec_data = pad_sequences(Sequences_id,maxlen = max_len)

    tf_seq = tf.convert_to_tensor(vec_data)  
    y = model.predict(tf_seq)
    y_preds = tf.argmax(y, axis=-1)
    

    label_ids = []
    for i in range(len(y_preds)):
        label_id = [y_preds[i][100-len(Sequences_id[i])+1:-1]]
        label_ids.append(label_id)

    idx_taget = ['O','B-Disease','B-Drug','I-Disease','I-Drug']
    taglist  = []
    for i in range(len(label_ids)):
        taget = []
        seq = Sequences[i].split(' ')[1:-1]
        label = np.array(label_ids[i][0])
        for j in range(len(label)):
            taget.append(str(seq[j])+' '+ idx_taget[label[j]])
        taglist.append(taget)
    return{
        'taget':taglist
    }

