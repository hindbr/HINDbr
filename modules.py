import sys, os, re
from gensim.utils import simple_preprocess
import pickle
from itertools import combinations 
import json
import random

def extract_bug_corpus(xmlfile, description):
    """
	return a list of documents from a xml file
    1) extrac bug corpora of a given bug xmfile 
       - bug summary                                
       - bug description and comments               
    2) do simple preprocess                         
       - convert a document into a list of lowercase tokens                          
       - ignore too long or too short tokens
	
	description: 1 for using, 0 not using description
    """

    f = open(xmlfile, 'r')
    contents = f.read()
    f.close()
    
    document_only_summary = []
    #find bug summary <short_desc>(.*)</short_desc>
    short_desc = re.findall('<short_desc>(.*?)</short_desc>', contents)
    if len(short_desc) != 0:
        short_desc = simple_preprocess(short_desc[0])
        bug_summary = short_desc
        document_only_summary.append(bug_summary)

    document_both_summary_description = document_only_summary
    #find bug description and comments <thetext>(.*?)</thetext>
    long_desc = re.findall('<thetext>(.*?)</thetext>', contents, re.DOTALL)
    if len(long_desc) != 0:
        for text in long_desc:
            text = simple_preprocess(text)
            document_both_summary_description.append(text)
    if description == 0:
        return document_only_summary
    else:
        return document_both_summary_description


def generate_bug_pairs(bug_bucket_fname):
    """
    return a tuple of duplicate pair list and non-duplicate pair list

    """  

    # load bug bucket
    with open(bug_bucket_fname, 'rb') as f: 
        bug_bucket = pickle.load(f) 

    duplicate_pairs = list()

    non_duplicate_bugs = list()

    for master in bug_bucket:
        count = len(bug_bucket[master])
        if count != 0:
            duplicates = bug_bucket[master]
            duplicates.add(master)
            for pair in combinations(duplicates, 2):
                duplicate_pairs.append(pair)
        else:
            non_duplicate_bugs.append(master)

    number_duplicate_pairs = len(duplicate_pairs)

    # Number of non duplicate pair candidates
    number_non_duplicate_pairs = 4 * number_duplicate_pairs

    non_duplicate_pairs = [next(random_pair_generator(non_duplicate_bugs)) for i in range(number_non_duplicate_pairs)]

    return duplicate_pairs, non_duplicate_pairs

def generate_bug_pairs_before_after_jit(bug_bucket_fname, bid_jit_range):

    # load bug bucket
    with open(bug_bucket_fname, 'rb') as f:
        bug_bucket = pickle.load(f) 

    bid_jit_up = bid_jit_range[1]
    bid_jit_down = bid_jit_range[0]

    duplicate_pairs_before_jit = list()
    non_duplicate_bugs_before_jit = list()

    duplicate_pairs_after_jit = list()
    non_duplicate_bugs_after_jit = list()

    for master in bug_bucket:
        if master < bid_jit_down:
            count = len(bug_bucket[master])
            if count != 0:
                duplicates = bug_bucket[master]
                duplicates.add(master)
                for pair in combinations(duplicates, 2):
                    duplicate_pairs_before_jit.append(pair)
            else:
                non_duplicate_bugs_before_jit.append(master)
        elif master > bid_jit_up:
            count = len(bug_bucket[master])
            if count != 0:
                duplicates = bug_bucket[master]
                duplicates.add(master)
                for pair in combinations(duplicates, 2):
                    duplicate_pairs_after_jit.append(pair)
            else:
                non_duplicate_bugs_after_jit.append(master)


    # Number of non duplicate pair candidates
    number_non_duplicate_pairs_before_jit = 4 * len(duplicate_pairs_before_jit)
    number_non_duplicate_pairs_after_jit = 4 * len(duplicate_pairs_after_jit)

    non_duplicate_pairs = [next(random_pair_generator(non_duplicate_bugs_before_jit)) for i in range(number_non_duplicate_pairs_before_jit)] + [next(random_pair_generator(non_duplicate_bugs_after_jit)) for i in range(number_non_duplicate_pairs_after_jit)]

    duplicate_pairs = duplicate_pairs_before_jit + duplicate_pairs_after_jit

    return duplicate_pairs, non_duplicate_pairs



def random_pair_generator(number_list):
    """
    return an iterator of random pairs from a list of numbers
	"""
    used_pairs = set()
    while True:
         pair = random.sample(number_list, 2)
         pair = tuple(sorted(pair))
         if pair not in used_pairs:
             used_pairs.add(pair)
             yield pair

def bug_pair_generator(bug_pair_list,xml_file_path,xml_file_prefix):

    used_pairs = set()
    while True:
        pair = random.sample(bug_pair_list, 1)[0]
        if pair not in used_pairs:
            used_pairs.add(pair)
            # Bug 1
            bid1 = (int)(pair[0])
            information1 = extract_bug_report_information(xml_file_path + xml_file_prefix + str(bid1) + '.xml')
            summary1 = information1[0]
            description1 = information1[1]
            pro1 = information1[2]
            com1 = information1[3]
            ver1 = information1[4]
            sev1 = information1[5]
            pri1 = information1[6]
            # plt1 = information1[7]
            # os1  = information1[8]
            sts1 = information1[9]

            # Bug 2
            bid2 = (int)(pair[1])
            information2 = extract_bug_report_information(xml_file_path + xml_file_prefix + str(bid2) + '.xml')
            summary2 = information2[0]
            description2 = information2[1]
            pro2 = information2[2]
            com2 = information2[3]
            ver2 = information2[4]
            sev2 = information2[5]
            pri2 = information2[6]
            # plt2 = information2[7]
            # os2  = information2[8]
            sts2 =  information2[9]

            # Label
            is_duplicate = (int)(pair[2])

            yield [bid1, summary1, description1, pro1, com1, ver1, sev1, pri1, sts1, bid2, summary2, description2, pro2, com2, ver2, sev2, pri2, sts2, is_duplicate] 
     
def extract_bug_report_information(xmlfile):
    with open(xmlfile, 'r') as f:
        contents = f.read()

        # Summary
        short_desc = re.findall('<short_desc>(.*?)</short_desc>', contents)
        if len(short_desc) != 0:
            # Output the raw summary text
            bug_summary = short_desc[0]
        else:
            bug_summary = ''

        # Description
        long_desc = re.findall('<thetext>(.*?)</thetext>', contents, re.DOTALL)
        if len(long_desc) != 0:
            bug_description = ' '.join(simple_preprocess(long_desc[0]))
        else:
            bug_description = 'NaN'

        # Product
        bug_product = re.findall('<product>(.*)</product>',contents)
        if len(bug_product) != 0:
            bug_product = 'PRO_' + bug_product[0]
        else:
            bug_product = ''

        # Component
        bug_component = re.findall('<component>(.*)</component>',contents)
        if len(bug_component) != 0:
            bug_component = 'COM_' + bug_component[0]
        else:
            bug_component = ''

        # Version
        if 'linux' in xmlfile:
            bug_version = re.findall('<cf_kernel_version>(.*)</cf_kernel_version>',contents)
        else:
            bug_version = re.findall('<version>(.*)</version>',contents)
        if len(bug_version) != 0:
            bug_version = 'VER_' + bug_version[0]
        else:
            bug_version = ''

        # Severity
        bug_severity = re.findall('<bug_severity>(.*)</bug_severity>',contents)
        if len(bug_severity) != 0:
            bug_severity = 'SEV_' + bug_severity[0]
        else:
            bug_severity = ''

        # Priority
        bug_priority = re.findall('<priority>(.*)</priority>',contents)
        if len(bug_priority) != 0:
            bug_priority = 'PRI_' + bug_priority[0]
        else:
            bug_priority = ''
        
        # Platform
        bug_platform = re.findall('<rep_platform>(.*)</rep_platform>',contents)
        if len(bug_platform) != 0:
            bug_platform = 'PLT_' + bug_platform[0]
        else:
            bug_platform = ''

        # Operating System
        bug_os = re.findall('<op_sys>(.*)</op_sys>',contents)
        if len(bug_os) != 0:
            bug_os = 'OS_' + bug_os[0]
        else:
            bug_os = ''

        # Bug Status
        bug_status = re.findall('<bug_status>(.*)</bug_status>',contents)
        if len(bug_status) != 0:
            bug_status = 'STS_' + bug_status[0]
        else:
            bug_status = ''


        # Creation time
        bug_reported_time = re.findall('<creation_ts>(.*)</creation_ts>',contents)
        if len(bug_reported_time) != 0:
            bug_reported_time = bug_reported_time[0][0:4]
        else:
            bug_reported_time = ''

    return bug_summary, bug_description, bug_product, bug_component, bug_version, bug_severity, bug_priority, bug_platform, bug_os, bug_status, bug_reported_time



## Pre Process and convert texts to a list of words '''
def text_to_word_list(text):
    ''' Pre Process and convert texts to a list of words '''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text




# hin node dictionary - {xml_object: (node_id,node_class)}
node_dict = {}

#node extraction from bug report (.xml)
def nodeGeneration(xmlfile):
    f = open(xmlfile,'r')
    content = f.read()
    f.close()

    #Bug ID
    BID = re.findall('<bug_id>(.*)</bug_id>',content)[0]

    #Product
    PRO = re.findall('<product>(.*)</product>',content)
    if len(PRO) != 0:
        PRO = 'PRO_' + PRO[0]
    else:
        PRO = ''

    #Component
    COM = re.findall('<component>(.*)</component>',content)
    if len(COM) != 0:
        COM = 'COM_' + COM[0]
    else:
        COM = ''

    #Version
    if 'linux' in xmlfile:
        VER = re.findall('<cf_kernel_version>(.*)</cf_kernel_version>',content)
    else:
        
        VER = re.findall('<version>(.*)</version>',content)
    if len(VER) != 0:
        VER = 'VER_' + VER[0]
    else:
        VER = ''

    #Severity
    SEV = re.findall('<bug_severity>(.*)</bug_severity>',content)
    if len(SEV) != 0:
        SEV = 'SEV_' + SEV[0]
    else:
        SEV = ''

    #Priority
    PRI = re.findall('<priority>(.*)</priority>',content)
    if len(PRI) != 0:
        PRI = 'PRI_' + PRI[0]
    else:
        PRI = ''

    #TODO
    '''
    #Platform
    PLT = re.findall('<rep_platform>(.*)</rep_platform>',content)
    if len(PLT) != 0:
        PLT = 'PLT_' + PLT[0]
    else:
        PLT = ''

    #Operating System
    OS = re.findall('<op_sys>(.*)</op_sys>',content)
    if len(OS) != 0:
        OS = 'OS_' + OS[0]
    else:
        OS = ''
    '''
#    nodes = [('BID',BID),('PRO',PRO),('COM',COM),('VER',VER),('SEV',SEV),('PRI',PRI),('PLT',PLT),('OS',OS)]
    nodes = [('BID',BID),('PRO',PRO),('COM',COM),('VER',VER),('SEV',SEV),('PRI',PRI)]

    for node in nodes:
        if node[1] != '':
            if node[1] not in node_dict:
                #node_dict: (node_id, node_class)
                node_dict[node[1]] = (len(node_dict) + 1, node[0])
    
    return nodes, node_dict


#edge generation from nodes
def edgeGeneration(nodes,node_dict,option):

    #output hin format for hin2vec tool 
    if option == 'default':
        edges = []
        BID = nodes[0]
        PRO = nodes[1]
        COM = nodes[2]
        VER = nodes[3]
        SEV = nodes[4]
        PRI = nodes[5]
#        PLT = nodes[6]
#        OS  = nodes[7]

        BID_id = node_dict[BID[1]][0]
        BID_type = node_dict[BID[1]][1]
       
       #1 BID-COM
        if COM[1] != '':
            COM_id = node_dict[COM[1]][0]
            COM_type = node_dict[COM[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(COM_id) + '\t' + COM_type + '\t' + BID_type + '-' + COM_type + '\n'
            edges.append(edge)
            edge = str(COM_id) + '\t' + COM_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + COM_type + '-' + BID_type + '\n'
            edges.append(edge)

        #2 BID-SEV
        if SEV[1] != '':
            SEV_id = node_dict[SEV[1]][0]
            SEV_type = node_dict[SEV[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(SEV_id) + '\t' + SEV_type + '\t' + BID_type + '-' + SEV_type + '\n'
            edges.append(edge)
            edge = str(SEV_id) + '\t' + SEV_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + SEV_type + '-' + BID_type + '\n'
            edges.append(edge)

        #3 BID-PRI
        if PRI[1] != '':
            PRI_id = node_dict[PRI[1]][0]
            PRI_type = node_dict[PRI[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(PRI_id) + '\t' + PRI_type + '\t' + BID_type + '-' + PRI_type + '\n'
            edges.append(edge)
            edge = str(PRI_id) + '\t' + PRI_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + PRI_type + '-' + BID_type + '\n'
            edges.append(edge)

        #4 BID-VER
        if VER[1] != '':
            VER_id = node_dict[VER[1]][0]
            VER_type = node_dict[VER[1]][1]
            edge = str(BID_id) + '\t' + BID_type + '\t' + str(VER_id) + '\t' + VER_type + '\t' + BID_type + '-' + VER_type + '\n'
            edges.append(edge)
            edge = str(VER_id) + '\t' + VER_type + '\t' + str(BID_id) + '\t' + BID_type + '\t' + VER_type + '-' + BID_type + '\n'
            edges.append(edge)

        #5 COM-PRO
        if COM[1] != '' and PRO[1] != '':
            COM_id = node_dict[COM[1]][0]
            COM_type = node_dict[COM[1]][1]
            PRO_id = node_dict[PRO[1]][0]
            PRO_type = node_dict[PRO[1]][1]
            edge = str(COM_id) + '\t' + COM_type + '\t' + str(PRO_id) + '\t' + PRO_type + '\t' + COM_type + '-' + PRO_type + '\n'
            edges.append(edge)
            edge = str(PRO_id) + '\t' + PRO_type + '\t' + str(COM_id) + '\t' + COM_type + '\t' + PRO_type + '-' + COM_type + '\n'
            edges.append(edge)

        #TODO
        '''
        #6 VER-OS
        if VER[1] != '' and OS[1] != '':
            VER_id = node_dict[VER[1]][0]
            VER_type = node_dict[VER[1]][1]
            OS_id = node_dict[OS[1]][0]
            OS_type = node_dict[OS[1]][1]
            edge = str(VER_id) + '\t' + VER_type + '\t' + str(OS_id) + '\t' + OS_type + '\t' + VER_type + '-' + OS_type + '\n'
            edges.append(edge)
            edge = str(OS_id) + '\t' + OS_type + '\t' + str(VER_id) + '\t' + VER_type + '\t' + OS_type + '-' + VER_type + '\n'
            edges.append(edge)

        #7 OS-PLT
        if OS[1] != '' and PLT[1] != '':
            OS_id = node_dict[OS[1]][0]
            OS_type = node_dict[OS[1]][1]
            PLT_id = node_dict[PLT[1]][0]
            PLT_type = node_dict[PLT[1]][1]
            edge = str(OS_id) + '\t' + OS_type + '\t' + str(PLT_id) + '\t' + PLT_type + '\t' + OS_type + '-' + PLT_type + '\n'
            edges.append(edge)
            edge = str(PLT_id) + '\t' + PLT_type + '\t' + str(OS_id) + '\t' + OS_type + '\t' + PLT_type + '-' + OS_type + '\n'
            edges.append(edge)
        '''      
	#TODO
    '''	
    elif option == '--xgmml':
        pass
    elif option == '--sif':
        pass
    else:
        print('wong option!')

    '''
    return edges
