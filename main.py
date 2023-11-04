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

import tqdm
import re
import string

# Trying to build a Word2Vec model
#from gensim.models import Word2Vec

# Define Python keywords translations

KEYWORDS_DICTIONARY = {
    "and": "and",
    "def": "definition",
    "global": "global",
    "or": "or",
    "False": "false",
    "as": "as",
    "del": "delete",
    "if": "if",
    "pass": "pass",
    "True": "true",
    "assert": "assert",
    "elif": "else if",
    "import": "import",
    "raise": "raise",
    "None": "none",
    "async": "async",
    "else": "else",
    "in": "in",
    "return": "return",
    "await": "await",
    "except": "except",
    "is": "is",
    "try": "try",
    "break": "break",
    "finally": "finally",
    "lambda": "lambda",
    "while": "while",
    "class": "class",
    "for": "for",
    "nonlocal": "non local",
    "with": "with",
    "continue": "continue",
    "from": "from",
    "not": "not",
    "yield": "yield",
    "(": "left parenthesis",
    ")": "right parenthesis",
    ":": "columns",
    "+": "plus",
    "-": "minus",
    "*": "times",
    "/": "split",
    "=": "equal"
}

df = pd.read_parquet('./dataset/train-00000-of-00009.parquet', engine='pyarrow')[["code1", "code2", "similar"]]
# CUT Dataframe
df = df.iloc[:100,:]
print("RAW Dataframe:")
print(df)

# Translate source codes
'''
def translate_code(code):
  tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
  result = []

  for token in tokens:
    if token.string in KEYWORDS_DICTIONARY:
      result.append(KEYWORDS_DICTIONARY[token.string])
  return ' '.join(map(str, result))

def translate_code2(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []

    for token in tokens:
        result.append(token.string)
    return ' '.join(map(str, result))

df['code1'] = df['code1'].apply(translate_code2)
df['code2'] = df['code2'].apply(translate_code2)

print("Translated Dataframe:")
print(df)
'''

##############################################################################################################################
'''
# Create word embeddings
# Source: https://www.tensorflow.org/text/tutorials/word2vec and https://jaketae.github.io/study/word2vec/

# Store all tokens
full_tokens = set()

# Set constants
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

# Tokenizes code
def apply_tokenization(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []

    for token in tokens:
        str_token = token.string
        result.append(str_token)
        full_tokens.add(str_token)
    return ' '.join(map(str, result))

# Maps every token to ID
def apply_mapping(tokens):
    vocab = {}
    inverse_vocab = {}

    for i, token in enumerate(set(tokens)):
        vocab[token] = i
        inverse_vocab[i] = token

    return vocab, inverse_vocab

# Apply tokenization to dataframe
df['code1'] = df['code1'].apply(apply_tokenization)
df['code2'] = df['code2'].apply(apply_tokenization)

print("Full tokens")
print(full_tokens)

print("Mapping")
# Map tokens into IDs
vocab, inverse_vocab = apply_mapping(full_tokens)
vocab_size = len(vocab)
print(vocab)

print("Translated Dataframe:")
print(df)

# Vectorize tokens
sequences = [vocab[word] for word in full_tokens]

# Generates skip-gram pairs with negative sampling for a list of sequences
# (int-encoded sentences) based on window size, number of negative samples
# and vocabulary size.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
  # Elements of each training example are appended to these lists.
  targets, contexts, labels = [], [], []

  # Build the sampling table for `vocab_size` tokens.
  sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

  # Iterate over all sequences (sentences) in the dataset.
  for sequence in tqdm.tqdm(sequences):

    # Generate positive skip-gram pairs for a sequence (sentence).
    positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
          sequence,
          vocabulary_size=vocab_size,
          sampling_table=sampling_table,
          window_size=window_size,
          negative_samples=0)

    # Iterate over each positive skip-gram pair to produce training examples
    # with a positive context word and negative samples.
    for target_word, context_word in positive_skip_grams:
      context_class = tf.expand_dims(
          tf.constant([context_word], dtype="int64"), 1)
      negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
          true_classes=context_class,
          num_true=1,
          num_sampled=num_ns,
          unique=True,
          range_max=vocab_size,
          seed=seed,
          name="negative_sampling")

      # Build context and label vectors (for one target word)
      context = tf.concat([tf.squeeze(context_class,1), negative_sampling_candidates], 0)
      label = tf.constant([1] + [0]*num_ns, dtype="int64")

      # Append each element from the training example to global lists.
      targets.append(target_word)
      contexts.append(context)
      labels.append(label)

  return targets, contexts, labels

# Standarizes code by converting letters to lowercase and remove punctuation
def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')

# Use the `TextVectorization` layer to normalize, split, and map strings to
# integers. Set the `output_sequence_length` length to pad all samples to the
# same length.
vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=10)

# Create dataset from list of tokens
text_ds = tf.data.experimental.from_list(list(full_tokens))

# Create vocabulary
vectorize_layer.adapt(text_ds.batch(1024))

# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()
print(inverse_vocab[:20])

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(vectorize_layer).unbatch()

# Obtain sequences
sequences = list(text_vector_ds.as_numpy_iterator())
print(len(sequences))
for seq in sequences[:50]:
  print(f"{seq} => {[inverse_vocab[i] for i in seq]}")

# Set constants
WINDOW_SIZE = 2
NUM_NS = 4

# Generate training examples from sequences
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=WINDOW_SIZE,
    num_ns=NUM_NS,
    vocab_size=vocab_size,
    seed=SEED)

targets = np.array(targets)
contexts = np.array(contexts)
labels = np.array(labels)

print('\n')
print(f"targets.shape: {targets.shape}")
print(f"contexts.shape: {contexts.shape}")
print(f"labels.shape: {labels.shape}")

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
print(dataset)

# Model and training
class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(vocab_size,
                                      embedding_dim,
                                      input_length=1,
                                      name="w2v_embedding")
    self.context_embedding = layers.Embedding(vocab_size,
                                       embedding_dim,
                                       input_length=NUM_NS+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots
  
# Define loss function
def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)

# Build model
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam',
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                 metrics=['accuracy'])

# Analyze model
#weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
weights = word2vec.get_layer('w2v_embedding').get_vocabulary()
vocab = vectorize_layer.get_vocabulary()
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(weights)
'''







# Create word embeddings
# Source: https://jaketae.github.io/study/word2vec/

# Store all tokens
full_tokens = set()

# Tokenizes code
def apply_tokenization(code):
    tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
    result = []

    for token in tokens:
        str_token = token.string
        result.append(str_token)
        full_tokens.add(str_token)
    return ' '.join(map(str, result))

# Maps every token to ID
def apply_mapping(tokens):
    token_to_id = {}
    id_to_token = {}

    for i, token in enumerate(set(tokens)):
        token_to_id[token] = i
        id_to_token[i] = token

    return token_to_id, id_to_token

# Apply tokenization to dataframe
df['code1'] = df['code1'].apply(apply_tokenization)
df['code2'] = df['code2'].apply(apply_tokenization)

print("Full tokens")
print(full_tokens)

print("Mapping")
# Map tokens into IDs
token_to_id, id_to_token = apply_mapping(full_tokens)
print(token_to_id)

print("Translated Dataframe:")
print(df)

# Generates a one-hot encodding for a single token
def one_hot_encode(id, token_size):
    result = [0] * token_size
    result[id] = 1
    return result

# Concatenates 2 range objects
def concatenate(*iterables):
    for iterable in iterables:
        yield from iterable

# Generates training data for tokens
def generate_training_data(tokens, token_to_id, window):
    x = []
    y = []
    num_tokens = len(tokens)

    for i in range(num_tokens):
        idx = concatenate(range(max(0, i - window), i), range(i, min(num_tokens, i + window + 1)))
        for j in idx:
            if i == j:
                continue
            x.append(one_hot_encode(token_to_id[list(tokens)[i]], len(token_to_id)))
            y.append(one_hot_encode(token_to_id[list(tokens)[j]], len(token_to_id)))
    return np.asarray(x), np.asarray(y)

# Generate training data prior to build an embedding model
embedding_train_x, embedding_train_y = generate_training_data(full_tokens, token_to_id, 2)

print(embedding_train_x.shape)
print(embedding_train_y.shape)

# Build neural network

# Inits neural network model
def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

model = init_network(len(token_to_id), 10)

def softmax(x):
    result = []
    for i in x:
        exp = np.exp(i)
        result.append(exp / exp.sum())
    return result

# Forward propagation
def forward(model, x, return_cache=True):
    cache = {}

    cache["a1"] = x @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])

    if not return_cache:
        return cache["z"]
    return cache

def cross_entropy(z, y):
    return - np.sum(np.log(z) * y)

def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

learning = one_hot_encode(token_to_id["def"], len(token_to_id))
result = forward(model, [learning], return_cache=False)[0]

for word in (id_to_token[id] for id in np.argsort(result)[::-1]):
    print(word)
# Get embedded
def get_embedding(model, word):
    try:
        idx = token_to_id[word]
    except KeyError:
        print("`word` not in corpus")
    one_hot = one_hot_encode(idx, len(token_to_id))
    return forward(model, one_hot)["a1"]

print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
print(get_embedding(model, "def"))
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print(model["w2"])

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
f = open('./embeddings/glove.6B/glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

not_present_list = []
vocab_size = len(t.word_index) + 1
print('Loaded %s word vectors.' % len(embeddings_index))
embedding_matrix = np.zeros((vocab_size, len(embeddings_index['no'])))
for word, i in t.word_index.items():
    if word in embeddings_index.keys():
        embedding_vector = embeddings_index.get(word)
    else:
        not_present_list.append(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
    else:
        embedding_matrix[i] = np.zeros(300)
##############################################################################################################################

# Create model
from keras.regularizers import l2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model

from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Concatenate
from keras.layers import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.layers import Input, Dense, Flatten, GlobalMaxPool2D, GlobalAvgPool2D, Concatenate, Multiply, Dropout, Subtract, Add, Conv2D

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
                       output_dim=len(embeddings_index['no']),weights=[embedding_matrix],
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