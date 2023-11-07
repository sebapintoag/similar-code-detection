# Import dependencies
import tokenize
from io import BytesIO
import pandas as pd

import numpy as np
import tensorflow as tf
from tensorflow import keras
import transformers
import keras
import keras.backend as K
import joblib
import sklearn

from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from keras.models import Model
from keras import layers

from time import time
import multiprocessing
from gensim.models import Word2Vec

# Read dataset and create a dataframe
df = pd.read_parquet('./dataset/train-00000-of-00009.parquet', engine='pyarrow')[["code1", "code2", "similar"]]

# Cut dataframe
df = df.iloc[:100,:]
print("RAW Dataframe:")
print(df)

##############################################################################################################################

# Create word embeddings

# Store all code fragments
code_fragments = []

# Tokenizes code
def apply_tokenization(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []

    for token in tokens:
        str_token = token.string
        result.append(str_token)
    code_fragments.append(result)
    return ' '.join(map(str, result))

# Apply tokenization to dataframe
df['code1'] = df['code1'].apply(apply_tokenization)
df['code2'] = df['code2'].apply(apply_tokenization)

cores = multiprocessing.cpu_count()

# Define Word2Vec model using fragments of code from dataframe
w2v_model = Word2Vec(min_count=20,
                     window=2,
                     vector_size=50,
                     sample=6e-5,
                     alpha=0.03,
                     min_alpha=0.0007,
                     negative=20,
                     workers=cores-1)

# Build vocabulary
t = time()

w2v_model.build_vocab(code_fragments, progress_per=10000)

print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

# Train model
t = time()

w2v_model.train(code_fragments, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

# Explore model
result = w2v_model.wv.most_similar(positive=["def"])
print(result)

weights = w2v_model.wv.vectors
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(w2v_model.wv.key_to_index)

##############################################################################################################################

# Create model
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers import Input, Concatenate, Conv2D, Flatten, Dense, Embedding, LSTM
from keras.models import Model
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D

# original source: https://github.com/prabhnoor0212/Siamese-Network-Text-Similarity/blob/master/quora_siamese.ipynb

# Create train, validation an test dataframes
X_temp, X_test, y_temp, y_test = train_test_split(df[['code1', 'code2']], df['similar'], test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)

X_train['text'] = X_train[['code1','code2']].apply(lambda x:str(x[0])+" "+str(x[1]), axis=1)

t = Tokenizer()
t.fit_on_texts(X_train['text'].values)

X_train['code1'] = X_train['code1'].astype(str)
X_train['code2'] = X_train['code2'].astype(str)

X_val['code1'] = X_val['code1'].astype(str)
X_val['code2'] = X_val['code2'].astype(str)

X_test['code1'] = X_test['code1'].astype(str)
X_test['code2'] = X_test['code2'].astype(str)

train_q1_seq = t.texts_to_sequences(X_train['code1'].values)
train_q2_seq = t.texts_to_sequences(X_train['code2'].values)
val_q1_seq = t.texts_to_sequences(X_val['code1'].values)
val_q2_seq = t.texts_to_sequences(X_val['code2'].values)
test_q1_seq = t.texts_to_sequences(X_test['code1'].values)
test_q2_seq = t.texts_to_sequences(X_test['code2'].values)

max_len = 30

train_q1_seq = pad_sequences(train_q1_seq, maxlen=max_len, padding='post')
train_q2_seq = pad_sequences(train_q2_seq, maxlen=max_len, padding='post')
val_q1_seq = pad_sequences(val_q1_seq, maxlen=max_len, padding='post')
val_q2_seq = pad_sequences(val_q2_seq, maxlen=max_len, padding='post')
test_q1_seq = pad_sequences(test_q1_seq, maxlen=max_len, padding='post')
test_q2_seq = pad_sequences(test_q2_seq, maxlen=max_len, padding='post')

#https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
#https://nlp.stanford.edu/projects/glove/
embeddings_index = {}
# Maps words and its vectors on a dictionary
for idx, key in enumerate(w2v_model.wv.key_to_index):
    embeddings_index[key] = w2v_model.wv[key]

print('Found %s word vectors.' % len(embeddings_index))

not_present_list = []
vocab_size = len(t.word_index) + 1
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, len(embeddings_index['def'])))

for word, i in t.word_index.items():
    if word in embeddings_index.keys():
        embedding_vector = embeddings_index.get(word)
    else:
        not_present_list.append(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros(300)

#######################################################################################################################

def cosine_distance(vests):
    x, y = vests
    x = K.l2_normalize(x, axis=-1)
    y = K.l2_normalize(y, axis=-1)
    return -K.mean(x * y, axis=-1, keepdims=True)

def cos_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0],1)

from sklearn.metrics import roc_auc_score

def auroc(y_true, y_pred):
    return tf.numpy_function(roc_auc_score, (y_true, y_pred), tf.double)

input_1 = Input(shape=(train_q1_seq.shape[1],))
input_2 = Input(shape=(train_q2_seq.shape[1],))

common_embed = Embedding(name="synopsis_embedd",input_dim =len(t.word_index)+1,
                       output_dim=len(embeddings_index['def']),weights=[embedding_matrix],
                       input_length=train_q1_seq.shape[1],trainable=False)

lstm_1 = common_embed(input_1)
lstm_2 = common_embed(input_2)

common_lstm = LSTM(64,return_sequences=True, activation="relu")
vector_1 = common_lstm(lstm_1)
vector_1 = Flatten()(vector_1)

vector_2 = common_lstm(lstm_2)
vector_2 = Flatten()(vector_2)

x3 = Subtract()([vector_1, vector_2])
x3 = Multiply()([x3, x3])

x1_ = Multiply()([vector_1, vector_1])
x2_ = Multiply()([vector_2, vector_2])
x4 = Subtract()([x1_, x2_])

#https://stackoverflow.com/a/51003359/10650182
x5 = Lambda(cosine_distance, output_shape=cos_dist_output_shape)([vector_1, vector_2])

conc = Concatenate(axis=-1)([x5,x4, x3])

x = Dense(100, activation="relu", name='conc_layer')(conc)
x = Dropout(0.01)(x)
out = Dense(1, activation="sigmoid", name = 'out')(x)

model = Model([input_1, input_2], out)

model.compile(loss="binary_crossentropy", metrics=['acc',auroc], optimizer=Adam(0.00001))

model.summary()

model.fit([train_q1_seq,train_q2_seq],y_train.values.reshape(-1,1), epochs = 5, batch_size=64,validation_data=([val_q1_seq, val_q2_seq],y_val.values.reshape(-1,1)))