import os,sys
import logging
import json
import argparse
from modules import *



'''Generate bug report heterogeneous information network '''

########## Settings ###############
PROJECT = 'linux'
###################################



HINFILENAME = 'data/bug_report_hin/' + PROJECT + '.hin'

with open('data/xmlfile_path.setting') as f:
    XML_FILE_PATH_DICT = json.load(f)

XML_FILE_PATH = XML_FILE_PATH_DICT[PROJECT][0]
XML_FILE_PREFIX = XML_FILE_PATH_DICT[PROJECT][1]


#log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('Generate bug report hin for the ' + PROJECT + ' project!')
logging.info('loading xml files...')

#list xml files
xmlfile_list = os.listdir(XML_FILE_PATH)

with open(HINFILENAME,'w') as f:
    f.write('#source_node' + '\t' + 'source_class' + '\t' + 'dest_node' + '\t' + 'dest_class' + '\t' + 'edge_class' + '\n')

    for xmlfile in xmlfile_list:
        if xmlfile[-4:] == '.xml':

            if xmlfile_list.index(xmlfile) % 1000 == 0:
                logging.info("{0} xml files have been processed!".format(xmlfile_list.index(xmlfile)))

            #node generation
            node_result = nodeGeneration(os.path.join(XML_FILE_PATH, xmlfile))
            #edge generation
            edge_result = edgeGeneration(node_result[0],node_result[1],'default')
            for edge in edge_result:
                f.write(edge)


#store and output hin nodes' dictionary
node_dict = node_result[1]
js = json.dumps(node_dict)
with open('data/hin_node_dict/' + PROJECT + '_node.dict','w') as f:
    f.write(js)

# store and output hin nodes' classes
with open('data/hin_node_dict/' +PROJECT + '_node_class.txt', 'w') as f:
    f.write('node_id' + '\t' + 'node class (separated by tab)' + '\n')
    for node in node_dict:
        node_id = node_dict[node][0]
        node_class = node_dict[node][1]
        f.write(str(node_id) + '\t' + node_class + '\n')

logging.info('HIN generation done!')
