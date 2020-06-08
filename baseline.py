from time import time
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Lambda, concatenate, Reshape, Flatten, Dropout, Dense, Activation, Bidirectional, CuDNNLSTM,  CuDNNGRU, Conv1D, MaxPooling1D, merge, AveragePooling1D
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.models import load_model
import tensorflow as tf
import json
from modules import text_to_word_list
import sys, os, pickle
from keras.optimizers import Adam
from imblearn.combine import SMOTETomek

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True 
sess = tf.Session(config = config)

# Project data
PROJECT = sys.argv[1]
MODEL_NO = 1
DATA_CSV = 'data/model_training/' + PROJECT + '_all.csv'

# Word embedding vectorDense(hidden_dims)
WORD_EMBEDDING_DIM = '300'
WORD_EMBEDDING_FILE = 'data/pretrained_embeddings/glove/glove.42B.300d.txt'
if os.path.isfile(WORD_EMBEDDING_FILE) == 0:
    print("\nError: Glove word embeddings not exist. Download at http://nlp.stanford.edu/data/glove.42B.300d.zip")
    exit()
GLOVE_PICKLE_FILE = 'data/pretrained_embeddings/glove/glove.42B.300d.pkl'

# Model Save
MODEL_SAVE_FILE = 'output/trained_model/' + PROJECT + '_'

# Model Training history record
EXP_HISTORY_ACC_SAVE_FILE = 'output/training_history/' + 'acc_' + PROJECT + '_' 
EXP_HISTORY_VAL_ACC_SAVE_FILE = 'output/training_history/' + 'val_acc_'+ PROJECT + '_'
EXP_HISTORY_LOSS_SAVE_FILE = 'output/training_history/' + 'loss_' + PROJECT + '_' 
EXP_HISTORY_VAL_LOSS_SAVE_FILE = 'output/training_history/' + 'val_loss_' + PROJECT + '_' 

# Model Test history record
EXP_TEST_HISTORY_FILE = 'output/training_history/' + 'test_result_' + PROJECT + '_' 


# Preprocess the text information
stops = set(stopwords.words('english'))


# Read data
# Load training and test set
data_df = pd.read_csv(DATA_CSV)
# Initialize stru features
data_df['stru1'] = 'nan'
data_df['stru2'] = 'nan'


# Prepare stru embedding

vocabulary = dict()
inverse_vocabulary = ['<unk>']

stru_cols1 = ["pro1","com1","ver1","sev1","pri1","sts1"]
stru_cols2 = ["pro2","com2","ver2","sev2","pri2","sts2"]


for index, row in data_df.iterrows():
    s2n = []
    for stru in stru_cols1:
        if str(row[stru]) != 'nan':
            if str(row[stru]) not in vocabulary:
                vocabulary[str(row[stru])] = len(inverse_vocabulary)
                s2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(str(row[stru]))
            else:
                s2n.append(vocabulary[str(row[stru])])
        else:
            s2n.append(0)
    data_df.at[index,'stru1'] = s2n

    s2n = []
    for stru in stru_cols2:
        if str(row[stru]) != 'nan':
            if str(row[stru]) not in vocabulary:
                vocabulary[str(row[stru])] = len(inverse_vocabulary)
                s2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(str(row[stru]))
            else:
                s2n.append(vocabulary[str(row[stru])])
        else:
            s2n.append(0)
    data_df.at[index,'stru2'] = s2n

# Struture embedding matrix settings
stru_embedding_dim = int(len(vocabulary))
stru_embeddings = 1 * np.random.randn(len(vocabulary) + 1, stru_embedding_dim)
stru_embeddings[0] = 0
 
# Build the struture embedding matrix
for stru, index in vocabulary.items():
    one_hot = np.zeros(stru_embedding_dim)
    one_hot[vocabulary[stru]-1] = 1
    stru_embeddings[index] = one_hot 

# Load Glove Word Embedding
print("Loading Glove Model")
if os.path.isfile(GLOVE_PICKLE_FILE):
    with open(GLOVE_PICKLE_FILE, 'rb') as f:
        gloveModel = pickle.load(f)
else:
    with open(WORD_EMBEDDING_FILE,'r') as f:
        gloveModel = {}
        for line in f:
            splitLines = line.split()
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            gloveModel[word] = wordEmbedding
    with open(GLOVE_PICKLE_FILE, 'wb') as f:
        pickle.dump(gloveModel,f)

print(len(gloveModel)," words loaded!")

##### Prepare short word embedding -- Summary
vocabulary = dict()
inverse_vocabulary = ['<unk>']

summaries_cols = ['summary1', 'summary2']

# Iterate over the summaries
for index, row in data_df.iterrows():
    # Iterate through the text of both summaries of the row
    for summary in summaries_cols:
        s2n = []
        for word in text_to_word_list(row[summary]):
            # Check for unwanted words
            if word in stops and word not in gloveModel:
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
short_word_embeddings = 1 * np.random.randn(len(vocabulary) + 1, word_embedding_dim)
short_word_embeddings[0] = 0

# Build the short word embedding matrix
for word, index in vocabulary.items():
    if word in gloveModel:
        short_word_embeddings[index] = gloveModel[word]


##### Prepare long word embedding -- Description
vocabulary = dict()
inverse_vocabulary = ['<unk>']

descriptions_cols = ['description1', 'description2']

# Iterate over the descriptions
for index, row in data_df.iterrows():
    # Iterate through the text of both descriptions of the row
    for description in descriptions_cols:
        s2n = []
        for word in text_to_word_list(row[description]):
            # Check for unwanted words
            if word in stops and word not in gloveModel:
                continue

            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                s2n.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                s2n.append(vocabulary[word])

        # Replace descriptions as word to description as number representaion
        data_df.at[index, description] = s2n

# Word embedding matrix settings
word_embedding_dim = int(WORD_EMBEDDING_DIM)
long_word_embeddings = 1 * np.random.randn(len(vocabulary) + 1, word_embedding_dim)
long_word_embeddings[0] = 0.0

# Build the long word embedding matrix
for word, index in vocabulary.items():
    if word in gloveModel:
        long_word_embeddings[index] = gloveModel[word]



##### Prepare train test data
max_seq_length_stru = 6
max_seq_length_short = 100
max_seq_length_long = 300

stru_cols = ['stru1','stru2']

X = data_df[stru_cols + summaries_cols + descriptions_cols]
Y = data_df['is_duplicate']

##### Define 5-fold cross validation test harness
# Fix random seed for train-test data split
seed = 7

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cvscores_baseline = []

for train, test in kfold.split(X, Y):
    MODEL_NO = str(MODEL_NO)
   
    print('\n*********** ' + 'Running 5-fold validation: '  + MODEL_NO + ' ! ***********\n')

    # Split to dicts
    X_stru_train = {'left': X.iloc[train].stru1, 'right': X.iloc[train].stru2}
    X_stru_test = {'left': X.iloc[test].stru1, 'right': X.iloc[test].stru2}

    X_short_text_train = {'left': X.iloc[train].summary1, 'right': X.iloc[train].summary2}
    X_short_text_test = {'left': X.iloc[test].summary1, 'right': X.iloc[test].summary2}

    X_long_text_train = {'left': X.iloc[train].description1, 'right': X.iloc[train].description2}
    X_long_text_test = {'left': X.iloc[test].description1, 'right': X.iloc[test].description2}


    # Zero padding
    for dataset, side in itertools.product([X_stru_train, X_stru_test],['left','right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length_stru)

    for dataset, side in itertools.product([X_short_text_train, X_short_text_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length_short)

    for dataset, side in itertools.product([X_long_text_train, X_long_text_test], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length_long)


    # Convert labels to their numpy representations
    Y_train = Y.iloc[train].values
    Y_test = Y.iloc[test].values

    Y_test_new = np.empty([0,2])
    for label in Y_test:
        if label == 1:
            Y_test_new = np.append(Y_test_new,[[0,1]],axis=0)
        else:
            Y_test_new = np.append(Y_test_new,[[1,0]],axis=0)
    Y_test = Y_test_new.astype('int')

    ##### Performing oversampling in the training sets at each iteration of 5-fold cross-validation procedure #####
    # Concatenate X_stru_train_left, X_stru_train_right, X_short_text_train_left, X_short_text_train_right, X_long_text_train_left, X_long_text_train_right
    X_train_cat = np.concatenate([X_stru_train['left'],X_stru_train['right'],X_short_text_train['left'],X_short_text_train['right'],X_long_text_train['left'],X_long_text_train['right']],axis=-1)
    print(np.shape(X_train_cat))

    # Over-sampling using SMOTE and cleaning using Tomek links
    smt = SMOTETomek(random_state=42,n_jobs=14)
    X_train_cat_res, Y_train_res = smt.fit_resample(X_train_cat, Y_train)

    stru_features_num = np.shape(X_stru_train['left'])[1]
    short_text_features_num = np.shape(X_short_text_train['left'])[1]
    long_text_features_num = np.shape(X_long_text_train['left'])[1]

    X_stru_train_left_res = X_train_cat_res[:,0:stru_features_num]
    X_stru_train_right_res = X_train_cat_res[:,stru_features_num:2*stru_features_num]
    X_short_text_train_left_res = X_train_cat_res[:,2*stru_features_num:2*stru_features_num+short_text_features_num]
    X_short_text_train_right_res = X_train_cat_res[:,2*stru_features_num+short_text_features_num:2*stru_features_num+2*short_text_features_num]
    X_long_text_train_left_res = X_train_cat_res[:,2*stru_features_num+2*short_text_features_num:2*stru_features_num+2*short_text_features_num+long_text_features_num]
    X_long_text_train_right_res = X_train_cat_res[:,2*stru_features_num+2*short_text_features_num+long_text_features_num:]

    Y_train_new = np.empty([0,2])
    for label in Y_train_res:
        if label == 1:
            Y_train_new = np.append(Y_train_new,[[0,1]],axis=0)
        else:
            Y_train_new = np.append(Y_train_new,[[1,0]],axis=0)
    Y_train_res = Y_train_new.astype('int')

  
    # Make sure everything is ok
    assert X_short_text_train_left_res.shape == X_short_text_train_right_res.shape
    assert len(X_short_text_train_left_res) == len(Y_train_res)
    
    #################################
    ### Model BASELINE ###
    MODEL_NAME = 'DLDBR'
    K.clear_session()
    #################################
    
    # Model variables
    num_filters = 32 
    batch_size = 64
    n_epoch = 100

    ## Structure Information Representation ##
    # 1) Structure Input Layer
    bug_stru_left_input = Input(shape=(max_seq_length_stru,), dtype='float32', name='stru_left_input')
    bug_stru_right_input = Input(shape=(max_seq_length_stru,), dtype='float32', name='stru_right_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(stru_embeddings), 
                                output_dim = stru_embedding_dim, 
                                weights=[stru_embeddings], 
                                input_length=max_seq_length_stru, 
                                trainable=False,
                                name='stru_embedding')
    bug_stru_embedding_left = embedding_layer(bug_stru_left_input)
    bug_stru_embedding_right = embedding_layer(bug_stru_right_input)
    # 3) Single Neural Network
    bug_stru_left_repr = Dense(50,activation='tanh')(Flatten()(bug_stru_embedding_left))
    bug_stru_right_repr = Dense(50,activation='tanh')(Flatten()(bug_stru_embedding_left))


    ## Short Text Information Representation ##
    # 1) Short Text Input Layer
    bug_short_text_left_input = Input(shape=(max_seq_length_short,), dtype='int32', name='short_text_left_input')
    bug_short_text_right_input = Input(shape=(max_seq_length_short,), dtype='int32', name='short_text_right_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(short_word_embeddings), 
                                output_dim = word_embedding_dim, 
                                weights=[short_word_embeddings], 
                                input_length=max_seq_length_short, 
                                trainable=False,
                                name='short_text_embedding')
    bug_short_text_embedding_left = embedding_layer(bug_short_text_left_input)
    bug_short_text_embedding_right = embedding_layer(bug_short_text_right_input)
    # 3) Shared Bi-LSTM 
    shared_bilstm = Bidirectional(CuDNNGRU(50, return_sequences=True, name='shared_bilstm'))
    bug_short_text_left_bilstm = shared_bilstm(bug_short_text_embedding_left)
    bug_short_text_right_bilstm = shared_bilstm(bug_short_text_embedding_right)

    bug_short_text_left_repr = Flatten()(AveragePooling1D()(bug_short_text_left_bilstm))
    bug_short_text_right_repr = Flatten()(AveragePooling1D()(bug_short_text_right_bilstm))


    ## Long Text Information Representation ##
    # 1) Long Text Input Layer
    bug_long_text_left_input = Input(shape=(max_seq_length_long,), dtype='int32', name='long_text_left_input')
    bug_long_text_right_input = Input(shape=(max_seq_length_long,), dtype='int32', name='long_text_right_input')
    # 2) Embedding Layer
    embedding_layer = Embedding(input_dim = len(long_word_embeddings),
                                output_dim = word_embedding_dim, 
                                weights=[long_word_embeddings], 
                                input_length=max_seq_length_long, 
                                trainable=False,
                                name='long_text_embedding')
    bug_long_text_embedding_left = embedding_layer(bug_long_text_left_input)
    bug_long_text_embedding_right = embedding_layer(bug_long_text_right_input)
    # 3) Shared CNN 
    shared_cnn1d1 = Conv1D(num_filters,3,padding='same',activation='relu')
    bug_long_text_left_cnn1d1 = MaxPooling1D()(shared_cnn1d1(bug_long_text_embedding_left))
    bug_long_text_right_cnn1d1 = MaxPooling1D()(shared_cnn1d1(bug_long_text_embedding_right))

    shared_cnn1d2 = Conv1D(num_filters,4,padding='same',activation='relu')
    bug_long_text_left_cnn1d2 = MaxPooling1D()(shared_cnn1d2(bug_long_text_embedding_left))
    bug_long_text_right_cnn1d2 = MaxPooling1D()(shared_cnn1d2(bug_long_text_embedding_right))

    shared_cnn1d3 = Conv1D(num_filters,5,padding='same',activation='relu')
    bug_long_text_left_cnn1d3 = MaxPooling1D()(shared_cnn1d3(bug_long_text_embedding_left))
    bug_long_text_right_cnn1d3 = MaxPooling1D()(shared_cnn1d3(bug_long_text_embedding_right))

    bug_long_text_left_repr = Dropout(0.5)(Dense(50,activation='tanh')(Flatten()(concatenate([bug_long_text_left_cnn1d1,bug_long_text_left_cnn1d2,bug_long_text_left_cnn1d3],axis=1))))
    bug_long_text_right_repr = Dropout(0.5)(Dense(50,activation='tanh')(Flatten()(concatenate([bug_long_text_right_cnn1d1,bug_long_text_right_cnn1d2,bug_long_text_right_cnn1d3],axis=1))))


    ## Bug Report Representation ##
    merge_bug_left = concatenate([bug_stru_left_repr,bug_short_text_left_repr,bug_long_text_left_repr])
    merge_bug_right = concatenate([bug_stru_right_repr,bug_short_text_right_repr,bug_long_text_right_repr])
    bug_repr = concatenate([merge_bug_left,merge_bug_right])

    ## Output Layer ##
    output_layer = Dense(2,activation='softmax',name='output_layer')
    output = output_layer(bug_repr)

    ## Build the model ##
    model_baseline = Model(inputs=[bug_stru_left_input, bug_stru_right_input, bug_short_text_left_input, 
        bug_short_text_right_input, bug_long_text_left_input, bug_long_text_right_input], outputs=[output])
    model_baseline.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model_baseline.summary()
    
    ## Train the model ##
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    training_start_time = time()
    model_trained = model_baseline.fit([X_stru_train_left_res, X_stru_train_right_res, X_short_text_train_left_res, 
        X_short_text_train_right_res, X_long_text_train_left_res, X_long_text_train_right_res], Y_train_res, batch_size=batch_size, epochs=n_epoch, validation_split=0.2, shuffle=True,callbacks=[es])
 
    print("Training time finished.\n{} epochs in {}".format(n_epoch, datetime.timedelta(seconds=time()-training_start_time)))

    ## Test ##
    Y_pred = model_baseline.predict([X_stru_test['left'], X_stru_test['right'], X_short_text_test['left'], 
        X_short_text_test['right'], X_long_text_test['left'], X_long_text_test['right']])

    accuracy = accuracy_score(Y_test == 1, Y_pred >= 0.27)
    precision = precision_score(Y_test == 1, Y_pred >= 0.27, average=None)
    recall = recall_score(Y_test == 1, Y_pred >= 0.27, average=None)
    f_measure = f1_score(Y_test == 1, Y_pred >= 0.27, average=None)

    print("model_baseline: test accuracy: {0}".format(accuracy))
    print("model_baseline: test precision p: {0}".format(precision[1]))
    print("model_baseline: test recall p: {0}".format(recall[1]))
    print("model_baseline: test f1 score p: {0}".format(f_measure[1]))
    print("model_baseline: test precision n: {0}".format(precision[0]))
    print("model_baseline: test recall n: {0}".format(recall[0]))
    print("model_baseline: test f1 score n: {0}".format(f_measure[0]))

    cvscores_baseline.append(accuracy * 100)

    ## Save model ##
    model_baseline.save(MODEL_SAVE_FILE + MODEL_NAME + '_' + MODEL_NO + '.h5')


    ## Record test result ##
    with open(EXP_TEST_HISTORY_FILE + MODEL_NAME, 'a') as f:
        f.write(str(accuracy) + '\t')
        f.write(str(precision[1]) + '\t')
        f.write(str(recall[1]) + '\t')
        f.write(str(f_measure[1]) + '\t')
        f.write(str(precision[0]) + '\t')
        f.write(str(recall[0]) + '\t')
        f.write(str(f_measure[0]) + '\n')


    ## Record training history ##
    # Accuracy score
    with open(EXP_HISTORY_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['accuracy']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_ACC_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_accuracy']:
            f.write(str(i) + '\t') 
        f.write('\n')

    # Loss
    with open(EXP_HISTORY_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    with open(EXP_HISTORY_VAL_LOSS_SAVE_FILE + MODEL_NAME, 'a') as f:
        for i in model_trained.history['val_loss']:
            f.write(str(i) + '\t')
        f.write('\n')

    del model_baseline
    del model_trained


    MODEL_NO = int(MODEL_NO)
    MODEL_NO += 1

print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_baseline), np.std(cvscores_baseline)))
