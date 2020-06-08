import os
import gensim
import logging

from gensim.test.utils import common_texts
from gensim.models import Word2Vec
from modules import extract_bug_corpus
import json


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("This may take a while...")

with open('data/xmlfile_path.setting') as f:
    xmlfile_path_dict = json.load(f)

xmlfile_paths = list()

for project in xmlfile_path_dict:
    xmlfile_paths.append(xmlfile_path_dict[project][0])

xmlfile_list = list()

for xmlfile_path in xmlfile_paths:
    logging.info("reading xmlfiles in path: {0}".format(xmlfile_path))
    xmlfile_path = xmlfile_path.strip()
    temp_list = os.listdir(xmlfile_path)
    for i in range(0,len(temp_list)):
        temp_list[i] = os.path.join(xmlfile_path, temp_list[i])
        xmlfile_list.append(temp_list[i])



#bug corpus
class bug_corpus:
    def __init__(self, filelist):
        self.filelist = filelist
 
    def __iter__(self):
        for fname in self.filelist:
            xmlfile = fname
            if xmlfile[-4:] == ".xml":
                documents = extract_bug_corpus(xmlfile, 1)
                if documents != None:
                    for line in documents:
                        yield line

logging.info ("Done reading data file")


#training the bug word2vec model
sentences = bug_corpus(xmlfile_list)

bug_w2c_model = Word2Vec(sentences, 
                         size=100, # Dimensionality 
                         window=10,# 5 for cbow, 10 for sg
                         min_count=5,
                         workers=10,
                         sg=1 # 0 for cbow, 1 for sg
                         )

bug_w2c_model.wv.save_word2vec_format('data/pretrained_embeddings/word2vec/bugreport-vectors-gensim-sg100dwin10.bin',binary=True)

logging.info("Training is done and the model is saved!")
