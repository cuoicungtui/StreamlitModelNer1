from transformers import BertTokenizerFast
import tensorflow as tf

from tensorflow import keras
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

# path_model_json = './model/model_Ner1/modelsave/modeljson1/model.json'
# path_weight_h5  = './model/model_Ner1/modelsave/modeljson1/model.h5'

def load_model_json(path_model_json,path_weight_h5):

    # load model json
    with open(path_model_json, 'r') as f:
        model_json = f.read()
        loaded_model = keras.models.model_from_json(model_json)

    # load weights into new model
    loaded_model.load_weights(path_weight_h5)

    return loaded_model


def text_to_token(text):

    # custom_objects = {'CustomCrossEntropy': CustomCrossEntropy}
    # with keras.utils.custom_object_scope(custom_objects):
    #     model = keras.models.load_model(path)
    # path_model_json = './model/model_Ner1/modelsave/modeljson1/model.json'
    # path_weight_h5  = './model/model_Ner1/modelsave/modeljson1/weights_ner.h5'

    path_model_json = env_variables['PATH_MODEL_JSON']
    path_weight_h5  = env_variables['PATH_WEIGHT_H5']

    model = load_model_json(path_model_json,path_weight_h5)

    path_tokenizer = env_variables['PATH_TOKENIZER']

    tokenizer = BertTokenizerFast.from_pretrained(path_tokenizer, do_lower_case=True)
    texts = text[0].split('.')
    # texts = [text+'.' for text in texts if text != '']

    max_len = 236
    texts_token = [tokenizer(seq, padding='max_length', max_length=max_len, truncation=True, return_tensors="pt") for seq in texts]
    word_ids = [token.word_ids() for token in texts_token] 
    word_ids_2 = [word_id_pad(word_id) for word_id in word_ids]
    
    input_ids = [token['input_ids'].numpy() for token in texts_token]
    tf_seq = tf.convert_to_tensor(input_ids)
    tf_seq =  tf.squeeze(tf.transpose(tf_seq, [0, 2, 1]), axis=-1) 
    
    y = model.predict(tf_seq)
    y_preds = tf.argmax(y, axis=-1)
    
    labels = []
    label_ids = []
    for i in range(len(y_preds)):
        y_lb = word_ids_2[i]
        y_pred = y_preds[i]
        y_p_clean =  y_pred[y_lb != -100]
        y_lb_clean = y_lb[y_lb != -100]
        labels.append(y_p_clean)
        label_ids.append(y_lb_clean)
    
    id_to_tokens = [ tokenizer.convert_ids_to_tokens(token['input_ids'][0])  for token in texts_token]
    idx_taget = ['O','B-Disease','B-Drug','I-Disease','I-Drug']
    taglist  = []
    for i in range(len(id_to_tokens)):
        taget = []
        id_to_token = id_to_tokens[i]
        y_pre = labels[i].numpy()
        text = " ".join(id_to_token).replace(" ##", "").split()
        index = 1
        for i in range(len(y_pre)):
            if id_to_token[i+1][0] != '#' :
                taget.append(text[index] + ' '+idx_taget[y_pre[i]] )
                index+=1
        taglist.append(taget)
            
    return{
        'taget':taglist
    }

