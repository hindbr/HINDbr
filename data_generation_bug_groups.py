import os, re
import logging
import networkx as nx
import pickle
import sys
import json

"""
Generate bug groups
identify non-duplicate bugs as master bug reports, then storing them in dictionary bug_group
- each item represents the same bug. 
- key: master bug report
- value: duplicate bug report(s) or none if the master bug report has no duplicates
"""
########## Settings ###############
PROJECT = 'openoffice'
###################################



logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info('Generating bug groups for the project: ' + PROJECT)

with open('data/xmlfile_path.setting') as f:
    XML_FILE_PATH_DICT = json.load(f)

XML_FILE_PATH = XML_FILE_PATH_DICT[PROJECT][0]
XML_FILE_PREFIX = XML_FILE_PATH_DICT[PROJECT][1]
BUG_GROUP_FNAME = 'data/bug_report_groups/' + PROJECT + '_all.pkl'

filelist = os.listdir(XML_FILE_PATH)

bug_graph = nx.DiGraph()

xmlfile_count = 0
for xmlfile in filelist:
    if xmlfile[-4:] == ".xml":
        xmlfile_path = os.path.join(XML_FILE_PATH,xmlfile)
    
        xmlfile_count += 1
        if xmlfile_count % 1000 == 0:
            logging.info('Processing {0} xml files'.format(xmlfile_count))
    
        f = open(xmlfile_path, 'r')
        contents = f.read()
        f.close()
        
        # get bug id
        bugid = int(re.findall('<bug_id>(.*?)</bug_id>', contents)[0])              
        
        # get resolution status
        resolution = re.findall('<resolution>(.*?)</resolution>', contents) 
    
        # get dupids if the resolution is duplicate
        if resolution == ['DUPLICATE']:
            dupids = re.findall('<dup_id>(.*?)</dup_id>', contents)  
            # dupid found
            if len(dupids) != 0:
                for dupid in dupids:
                    # the reported time of the dupid may beyond the selected data set
                    if os.path.exists(os.path.join(XML_FILE_PATH, re.sub(r'\d+', dupid, xmlfile))):                               
    					    # add edge (bug_id (resolution: DUPLICATE) -> bug_id (dup_id: master candidate)) to bug_graph
                        dupid = int(dupid)
                        bug_graph.add_edge(bugid,dupid)  

            # The dupids are missing in some bug reports, thus we have to discard these bug reports from the data set.
            else:
                pass
        else:
            bug_graph.add_node(bugid)

logging.info('Note that {0} bug report xmlfiles in the data dir {1}'.format(xmlfile_count, XML_FILE_PATH))
logging.info('Note that {0} valid bug reports in the bug group {1}'.format(len(bug_graph), BUG_GROUP_FNAME))

bug_group = dict()

for subgraph in nx.weakly_connected_component_subgraphs(bug_graph):
    # one bug report in the subgraph: non-duplicate
    if len(subgraph) == 1:  
        non_duplicate_bug_report = list(subgraph.nodes()).pop()
        bug_group[non_duplicate_bug_report] = set()
   
        # more than one bug report in the subgraph: dupliccates, including two cases: cycle or no cycle
    else:   
        master_candidates = list(subgraph.nodes())
        # check whether the duplicate subgraph has cycles
        try:  
            cycle_edges = nx.algorithms.cycles.find_cycle(subgraph)
    	    
        # subgraph has no cycles
        except:  
            for out_degree in bug_graph.out_degree(node for node in master_candidates):
                if out_degree[1] == 0:
                    # master bug report has no dupid (out edge), thus its out degree is 0
                     master_bug_report = out_degree[0]  
            # all duplicates without master bug report
            master_candidates.remove(master_bug_report)  
            bug_group[master_bug_report] = set()

            for duplicate in master_candidates:
                bug_group[master_bug_report].add(duplicate)

        # subgraph has a cycle (e.g., bugid (dupid) -> dupid (bugid)
        else: 
            # cycle_nodes: nodes in the cycle
            cycle_nodes = set()  
            for edge in cycle_edges:
                for node in edge:
                    cycle_nodes.add(node)
           
            # reported times dict for cycle nodes
            cycle_node_reported_times = dict()  

            for cycle_node in cycle_nodes:
                # cycle_node's xmlfile
                xmlfile = re.sub(r'\d+', str(cycle_node), xmlfile) 
                xmlfile = os.path.join(XML_FILE_PATH, xmlfile)
                f = open(xmlfile,'r')
                contents = f.read()
                f.close()

                # cycle_node's reported time
                cycle_node_reported_time = re.findall('<creation_ts>(.*?)</creation_ts>', contents)[0]  
                # all cycle_nodes' reported times
                cycle_node_reported_times[cycle_node_reported_time] = cycle_node  

            # the earliest reported time of cycle_nodes
            earliest_reported_time = min(cycle_node_reported_times)  
            # the earliest reported bug is the master bug report
            master_bug_report = cycle_node_reported_times[earliest_reported_time]  
           
    		    # all duplicates without master bug report
            master_candidates.remove(master_bug_report)  
            bug_group[master_bug_report] = set()
            for duplicate in master_candidates:
                bug_group[master_bug_report].add(duplicate)

f = open(BUG_GROUP_FNAME,'wb')
pickle.dump(bug_group,f)
f.close()

logging.info('Note that {0} master bug reports in the bug group {1}'.format(len(bug_group), BUG_GROUP_FNAME))
logging.info('Note that {0} duplicates in the bug group {1}'.format(sum(len(bug_group[key]) for key in bug_group), BUG_GROUP_FNAME))
