from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Lambda, concatenate, Flatten, Dropout, Dense, Activation, Bidirectional, CuDNNLSTM, BatchNormalization
import keras.backend as K
from keras.optimizers import Adadelta, adam

from keras.models import load_model
import tensorflow as tf
import json
from modules import text_to_word_list
import sys, os, pickle

from imblearn.combine import SMOTETomek

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.Session(config = config)

# Project data
PROJECT = sys.argv[1]
EVALUATION_DATA_TYPE = sys.argv[2]
MODEL_NO =  sys.argv[3]
DATA_CSV = 'data/before_after_jit_data/' + PROJECT + '_' + EVALUATION_DATA_TYPE + '.csv'
TEXT_DATA = 'summary_'

# Word embedding vector
WORD_EMBEDDING_DIM = '100'
EMBEDDING_ALGO = 'sg'
WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/word2vec/bugreport-vectors-gensim-' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM +'dwin10.bin'

# HIN embedding vector
HIN_EMBEDDING_DIM = '128'
HIN_EMBEDDING_FILE = 'data/pretrained_embeddings/hin2vec/' + PROJECT + '_node_' + HIN_EMBEDDING_DIM + 'd_5n_4w_1280l.vec'
HIN_NODE_DICT = 'data/hin_node_dict/' + PROJECT + '_node.dict'

# Model Save
MODEL_SAVE_FILE = 'output/trained_model/HINDBR/' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA

# Model Test history record
EXP_TEST_HISTORY_FILE = 'output/training_history/' + 'test_result_' + PROJECT + '_' + EMBEDDING_ALGO + WORD_EMBEDDING_DIM + 'dwin10final_' + TEXT_DATA + EVALUATION_DATA_TYPE + '_' 


# Read data
# Load training and test set
data_df = pd.read_csv(DATA_CSV)
# Initialize hin features
data_df['hin1'] = 'nan'
data_df['hin2'] = 'nan'


# Preprocess the text information
stops = set(stopwords.words('english'))

##### Prepare word embedding -- Summary
vocabulary = dict()
inverse_vocabulary = ['<unk>']
word2vec = KeyedVectors.load_word2vec_format(WORD_EMBEDDING_FILE, binary=True)

summaries_cols = ['summary1', 'summary2']

# Iterate over the summaries
for index, row in data_df.iterrows():
    # Iterate through the text of both summaries of the row
    for summary in summaries_cols:
        s2n = []
        for word in text_to_word_list(row[summary]):
            # Check for unwanted words
            if word in stops and word not in word2vec.vocab:
                continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                s2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                s2n.append(vocabulary[word])

        # Replace summaries as word to summary as number representaion
        data_df.at[index, summary] = s2n

# Word embedding matrix settings
word_embedding_dim = int(WORD_EMBEDDING_DIM)
word_embeddings = 1 * np.random.randn(len(vocabulary) + 1, word_embedding_dim)
word_embeddings[0] = 0

# Build the word embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        word_embeddings[index] = word2vec.word_vec(word)


del word2vec


# Prepare hin embedding
hin_vocabulary = set()
hin_cols1 = ["bid1","pro1","com1","ver1","sev1","pri1"]
hin_cols2 = ["bid2","pro2","com2","ver2","sev2","pri2"]


with open(HIN_NODE_DICT, 'r') as f:
    hin_node_dict = json.load(f)

for index, row in data_df.iterrows():
    s2n = []
    for hin in hin_cols1:
        if str(row[hin]) != 'nan':
            hin_node_id = hin_node_dict[str(row[hin])][0]
            hin_vocabulary.add(hin_node_id)
            s2n.append(hin_node_id) 
        else:
            s2n.append(0)
    data_df.at[index,'hin1'] = s2n

    s2n = []
    for hin in hin_cols2:
        if str(row[hin]) != 'nan':
            hin_node_id = hin_node_dict[str(row[hin])][0]
            hin_vocabulary.add(hin_node_id)
            s2n.append(hin_node_id) 
        else:
            s2n.append(0)
    data_df.at[index,'hin2'] = s2n


# HIN embedding matrix settings
hin_embedding_dim = int(HIN_EMBEDDING_DIM)
hin_embeddings = 1 * np.random.randn(max(hin_vocabulary) + 1, hin_embedding_dim)
hin_embeddings[0] = 0

# Load hin node2vec
node2vec = {}
with open(HIN_EMBEDDING_FILE) as f:
    first = True
    for line in f:
        if first:
            first = False
            continue
        line = line.strip()
        tokens = line.split(' ')
        node2vec[tokens[0]] = np.array(tokens[1:],dtype=float)

# Build the hin embedding matrix
for hin_node_id in hin_vocabulary:
    hin_embeddings[hin_node_id] = node2vec[str(hin_node_id)]


# Prepare train test data
# Max length of summaries
max_seq_length = max(data_df.summary1.map(lambda x: len(x)).max(),
                     data_df.summary2.map(lambda x: len(x)).max())

hin_cols = ['hin1','hin2']
X = data_df[hin_cols + summaries_cols]
Y = data_df['is_duplicate']

##### Define 5-fold cross validation test harness
# Fix random seed for reproducibility
seed = 7
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

fold_count = 1
for train, test in kfold.split(X, Y):
    MODEL_NO = str(MODEL_NO)
    print('\n*********** ' + 'Running 5-fold validation: '  + str(fold_count) + ' ! ***********\n')


    # Split to dicts -- train
    X_text_train = {'left': X.iloc[train].summary1, 'right': X.iloc[train].summary2}
    X_hin_train = {'left': X.iloc[train].hin1, 'right': X.iloc[train].hin2}

    # Split to dicts -- test
    X_text_test = {'left': X.iloc[test].summary1, 'right': X.iloc[test].summary2}
    X_hin_test = {'left': X.iloc[test].hin1, 'right': X.iloc[test].hin2}

    # Convert labels to their numpy representations
    Y_train = Y.iloc[train].values
    Y_test = Y.iloc[test].values

   

    # Zero padding
    for dataset, side in itertools.product([X_text_train, X_text_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

    for dataset, side in itertools.product([X_hin_train, X_hin_test],['left','right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=6)

    ##### Performing oversampling in the training sets at each iteration of 5-fold cross-validation procedure #####
    # Concatenate X_hin_train_left, X_hin_train_right, X_text_train_left, X_text_train_right
    X_train_cat = np.concatenate([X_hin_train['left'],X_hin_train['right'],X_text_train['left'],X_text_train['right']],axis=-1)

    # Over-sampling using SMOTE and cleaning using Tomek links
    smt = SMOTETomek(random_state=42,n_jobs=14)
    X_train_cat_res, Y_train_res = smt.fit_resample(X_train_cat, Y_train)

    hin_features_num = np.shape(X_hin_train['left'])[1] 
    text_features_num = np.shape(X_text_train['left'])[1]

    X_hin_train_left_res = X_train_cat_res[:,0:hin_features_num]
    X_hin_train_right_res = X_train_cat_res[:,hin_features_num:2*hin_features_num]
    X_text_train_left_res = X_train_cat_res[:,2*hin_features_num:2*hin_features_num+text_features_num]
    X_text_train_right_res = X_train_cat_res[:,2*hin_features_num+text_features_num:]



    # Make sure everything is ok
    assert X_text_train_left_res.shape == X_text_train_right_res.shape
    assert len(X_text_train_left_res) == len(Y_train_res)


    # Load model HINDBR
    MODEL_NAME = 'TEXT_HIN_DENSE'
    model_text_hin_dense = load_model(MODEL_SAVE_FILE + MODEL_NAME + '_' + MODEL_NO + '.h5')
    model_text_hin_dense.summary()
    model_text_hin_dense.save_weights('tmp/my_model_weights.h5')
    

    # Model variables
    n_hidden_rnn = 100
    n_dense_hin = 32
    n_dense_fusion = 64
    gradient_clipping_norm = 1.25
    batch_size = 128
    n_epoch = 100 

    ## Text Information Representation ##
    # 1) Text Input Layer
    bug_text_left_input = Input(shape=(max_seq_length,), dtype='int32', name='text_left_input')
    bug_text_right_input = Input(shape=(max_seq_length,), dtype='int32', name='text_right_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(word_embeddings), 
                                output_dim = word_embedding_dim, 
                                weights=[word_embeddings], 
                                input_length=max_seq_length, 
                                trainable=False,
                                name='text_embedding')
    bug_text_embedding_left = embedding_layer(bug_text_left_input)
    bug_text_embedding_right = embedding_layer(bug_text_right_input)
    # 3) Shared Bi-LSTM 
    shared_bilstm = Bidirectional(CuDNNLSTM(n_hidden_rnn, return_sequences=False, name='shared_bilstm'))
    bug_text_left_bilstm = shared_bilstm(bug_text_embedding_left)
    bug_text_right_bilstm = shared_bilstm(bug_text_embedding_right)
    bug_text_left_repr = Dropout(0.25)(bug_text_left_bilstm)
    bug_text_right_repr = Dropout(0.25)(bug_text_right_bilstm)


    ## Hin Information Representation ##
    # 1) Hin Input Layer
    bug_hin_left_input = Input(shape=(6,), dtype='int32', name='hin_left_input')
    bug_hin_right_input = Input(shape=(6,), dtype='int32', name='hin_right_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(hin_embeddings),
                            output_dim = hin_embedding_dim,
                            weights=[hin_embeddings],
                            input_length=6,
                            trainable=False,
                            name='hin_embedding')
    bug_hin_embedding_left = embedding_layer(bug_hin_left_input)
    bug_hin_embedding_right = embedding_layer(bug_hin_right_input)
    bug_hin_embedding_left_flat = Flatten()(bug_hin_embedding_left)
    bug_hin_embedding_right_flat = Flatten()(bug_hin_embedding_right)
    dense_layer = Dense(n_dense_hin, activation='tanh')
    bug_hin_left_repr = dense_layer(bug_hin_embedding_left_flat)
    bug_hin_right_repr = dense_layer(bug_hin_embedding_right_flat)

    ## Bug Report Representation ##
    merge_bug_text_hin_left = concatenate([bug_text_left_repr, bug_hin_left_repr])
    merge_bug_text_hin_right = concatenate([bug_text_right_repr, bug_hin_right_repr])
    dense_layer = Dense(n_dense_fusion,activation='tanh',name='dense_bugrepr')
    bug_left_repr = dense_layer(merge_bug_text_hin_left)
    bug_right_repr = dense_layer(merge_bug_text_hin_right)

    ## Malstm Distance Layer ##
    mahanttan_layer = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0]-x[1]), axis=1, keepdims=True)))
    mahanttan_distance_text_hin_dense = mahanttan_layer([bug_left_repr, bug_right_repr])
        
    ## Build the model ##
    new_model_text_hin_dense = Model(inputs=[bug_text_left_input, bug_text_right_input, bug_hin_left_input, bug_hin_right_input], outputs=[mahanttan_distance_text_hin_dense])
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    new_model_text_hin_dense.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    ## Load weights
    new_model_text_hin_dense.load_weights('tmp/my_model_weights.h5', by_name=True, skip_mismatch=True) 
    new_model_text_hin_dense.summary()

    ## Train the model ##
    training_start_time = time()
    model_trained = new_model_text_hin_dense.fit([X_text_train_left_res, X_text_train_right_res, X_hin_train_left_res, X_hin_train_right_res], Y_train_res, batch_size=batch_size, epochs=n_epoch, validation_split=0.2, shuffle=True)

    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))


    ## Test ##
    Y_pred_text_hin_dense = new_model_text_hin_dense.predict([X_text_test['left'], X_text_test['right'], X_hin_test['left'], X_hin_test['right']])


    accuracy = accuracy_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5)
    precision = precision_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    recall = recall_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    f_measure = f1_score(Y_test == 1, Y_pred_text_hin_dense >= 0.5, average=None)
    
    print("model_text_hin_dense: test accuracy: {0}".format(accuracy))
    print("model_text_hin_dense: test precision p: {0}".format(precision[1]))
    print("model_text_hin_dense: test recall p: {0}".format(recall[1]))
    print("model_text_hin_dense: test f1 score p: {0}".format(f_measure[1]))
    print("model_text_hin_dense: test precision n: {0}".format(precision[0]))
    print("model_text_hin_dense: test recall n: {0}".format(recall[0]))
    print("model_text_hin_dense: test f1 score n: {0}".format(f_measure[0]))

    ## Record test result ## 
    with open(EXP_TEST_HISTORY_FILE + MODEL_NAME, 'a') as f:
        f.write(str(accuracy) + '\t')
        f.write(str(precision[1]) + '\t')
        f.write(str(recall[1]) + '\t')
        f.write(str(f_measure[1]) + '\t')
        f.write(str(precision[0]) + '\t')
        f.write(str(recall[0]) + '\t')
        f.write(str(f_measure[0]) + '\n')

    K.clear_session()
    fold_count += 1
