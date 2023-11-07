import numpy as np
import tokenize
from io import BytesIO
from time import time
import multiprocessing
from gensim.models import Word2Vec

class Embedding:
    def __init__(self, dataframe, tkn, method = 'word2vec'):
        self.df = dataframe
        self.method = method
        self.tkn = tkn
        self.code_fragments = []
        self.model = None
        self.weights = None
        self.embeddings_index = {}
        self.embedding_matrix = None

        self.flat_code_fragments = []
        self.token_to_id = {}
        self.id_to_token = {}
        
    # Tokenizes code
    def apply_tokenization(self, code):
        tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
        result = []

        for token in tokens:
            str_token = token.string
            result.append(str_token)
        self.code_fragments.append(result)
        return ' '.join(map(str, result))
    
    # Maps every token to ID
    def apply_mapping(self):
        for i, token in enumerate(set(self.flat_code_fragments)):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

        return self.token_to_id, self.id_to_token

    def prepare_dataframe(self):
        # Apply tokenization to dataframe
        self.df['code1'] = self.df['code1'].apply(self.apply_tokenization)
        self.df['code2'] = self.df['code2'].apply(self.apply_tokenization)
        # Flatten code_fragments
        self.flat_code_fragments = flatten_list(self.code_fragments)
        # Map tokens into IDs
        self.apply_mapping()

    def get_vocab(self):
        if self.method == "word2vec":
            return self.model.key_to_index

    def get_embedding(self, word):
        if self.method == "word2vec":
            return self.model[word]

    def build_model(self):
        if self.method == "word2vec":
            self.model, self.weights = get_weights_using_word2vec(self.code_fragments)
        
        for _, key in enumerate(self.get_vocab()):
            self.embeddings_index[key] = self.get_embedding(key)

        print('Found %s word vectors.' % len(self.embeddings_index))

        not_present_list = []
        vocab_size = len(self.tkn.word_index) + 1
        print('Loaded %s word vectors.' % len(self.embeddings_index))
        self.embedding_matrix = np.zeros((vocab_size, len(self.embeddings_index['def'])))

        for word, i in self.tkn.word_index.items():
            if word in self.embeddings_index.keys():
                embedding_vector = self.embeddings_index.get(word)
            else:
                not_present_list.append(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector
            else:
                self.embedding_matrix[i] = np.zeros(300)

    def get_matrix(self):
        return self.embedding_matrix
    
    def get_index(self):
        return self.embeddings_index
    
    def get_word_index(self):
        return len(self.tkn.word_index)

def flatten_list(list_of_lists):
    result = []
    for list in list_of_lists:
        result = result + list
    return result
        
def get_weights_using_word2vec(code_fragments):
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
    # Return model and weights
    return w2v_model.wv, w2v_model.wv.vectors

def get_weights_using_raw_networks():
    return