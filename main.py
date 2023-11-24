import dataframe
import embedding
import ann
import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Read dataset
df = dataframe.from_parquet(
    "./dataset/train-00000-of-00009.parquet", ["code1", "code2", "similar"]
)
df = dataframe.cut(df, 1000)

# TODO: Apply preprocessing to code
# TODO: Apply subsambling to code

# Prepare training and validation sets
# Source: https://github.com/prabhnoor0212/Siamese-Network-Text-Similarity/blob/master/quora_siamese.ipynb

# Create train, validation an test dataframes
X_temp, X_test, y_temp, y_test = train_test_split(
    df[["code1", "code2"]], df["similar"], test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=42
)

X_train["text"] = X_train[["code1", "code2"]].apply(
    lambda x: str(x[0]) + " " + str(x[1]), axis=1
)

t = Tokenizer()
t.fit_on_texts(X_train["text"].values)

X_train["code1"] = X_train["code1"].astype(str)
X_train["code2"] = X_train["code2"].astype(str)

X_val["code1"] = X_val["code1"].astype(str)
X_val["code2"] = X_val["code2"].astype(str)

X_test["code1"] = X_test["code1"].astype(str)
X_test["code2"] = X_test["code2"].astype(str)

train_q1_seq = t.texts_to_sequences(X_train["code1"].values)
train_q2_seq = t.texts_to_sequences(X_train["code2"].values)
val_q1_seq = t.texts_to_sequences(X_val["code1"].values)
val_q2_seq = t.texts_to_sequences(X_val["code2"].values)
test_q1_seq = t.texts_to_sequences(X_test["code1"].values)
test_q2_seq = t.texts_to_sequences(X_test["code2"].values)

max_len = 30

train_q1_seq = pad_sequences(train_q1_seq, maxlen=max_len, padding="post")
train_q2_seq = pad_sequences(train_q2_seq, maxlen=max_len, padding="post")
val_q1_seq = pad_sequences(val_q1_seq, maxlen=max_len, padding="post")
val_q2_seq = pad_sequences(val_q2_seq, maxlen=max_len, padding="post")
test_q1_seq = pad_sequences(test_q1_seq, maxlen=max_len, padding="post")
test_q2_seq = pad_sequences(test_q2_seq, maxlen=max_len, padding="post")

print(df)
# Get embedding matrix
embedded = embedding.Embedding(df, t, "keras")
embedded.prepare_dataframe()
embedded.build_model()

# Create model
neuralnet = ann.SiameseNeuralNetwork(
    [train_q1_seq, train_q2_seq], [val_q1_seq, val_q2_seq], y_train, y_val, embedded
)

neuralnet.build()
neuralnet.compile()
neuralnet.summary()
history = neuralnet.fit()

# Plot
metrics.plot_history(embedded.history, "Embedding model", False)
metrics.plot_history(history, "Siamese model")

# Evaluate model
loss, accuracy, _ = neuralnet.evaluate()
print("Loss: %f" % (loss * 100))
print("Accuracy: %f" % (accuracy * 100))
