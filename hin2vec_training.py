import os


'''Train the hin embeddings on bug report hin '''

########## Settings ###############
PROJECT = 'linux'
##################################


HIN_PATH = os.path.join(os.getcwd(),'data/bug_report_hin/') + PROJECT + '.hin'
DIM_d = 128
NEGATIVE_SAMPLE_RATE_n = 5
WINDOW_w = 4
WALK_LENGTH_l = 1280
NUM_PROCESSES = 4

NODE_VEC_SAVE_FILE = os.path.join(os.getcwd(),'data/pretrained_embeddings/hin2vec/') + PROJECT + "_node_" + str(DIM_d) + "d_" + str(NEGATIVE_SAMPLE_RATE_n) + "n_" + str(WINDOW_w) + "w_" + str(WALK_LENGTH_l) + "l.vec"
METAPATH_VEC_SAVE_FILE = os.path.join(os.getcwd(),'data/pretrained_embeddings/hin2vec/') + PROJECT + "_metapath_" + str(DIM_d) + "d_" + str(NEGATIVE_SAMPLE_RATE_n) + "n_" + str(WINDOW_w) + "w_" + str(WALK_LENGTH_l) + "l.vec"

os.system("cd hin2vec/model_c/src/; make")

#Note that hin2vec uses python2 environment.
os.system("cd hin2vec; python2 main.py %s %s %s -d %d -n %d -w %d -l %d -p %d" % (HIN_PATH,NODE_VEC_SAVE_FILE,METAPATH_VEC_SAVE_FILE,DIM_d,NEGATIVE_SAMPLE_RATE_n,WINDOW_w,WALK_LENGTH_l,NUM_PROCESSES)) 
