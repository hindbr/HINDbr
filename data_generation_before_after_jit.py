import pandas as pd
from modules import generate_bug_pairs_before_after_jit, bug_pair_generator
import random, re, os, sys, logging, json


''' Generate before-JIT and after-JIT dataset for evaluate HINDBR'''

########## Settings ###############
PROJECT = 'eclipse'
###################################


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
with open('data/xmlfile_path.setting') as f:
    XML_FILE_PATH_DICT = json.load(f)

XML_FILE_PATH = XML_FILE_PATH_DICT[PROJECT][0]
XML_FILE_PREFIX = XML_FILE_PATH_DICT[PROJECT][1]

BUG_GROUP = 'data/bug_report_groups/' + PROJECT + '_all.pkl'
SAVE_FILE_NAME = 'data/before_after_jit_data/' + PROJECT + '_' 

logging.info("Generating before-jit and after-jit samples for project: " + PROJECT)

# JIT year Bug ID exclusion range
jit_bid_range = dict()
jit_bid_range['eclipse'] = [333372,367682] #BIDs in Year 2011
jit_bid_range['gcc'] = [47139,51723] #BIDs in Year 2011
jit_bid_range['freedesktop'] = [32766,44361] #BIDs in Year 2011
jit_bid_range['kde'] = [290290,312447] #BIDs in Year 2012
jit_bid_range['openoffice'] = [118717,121563] #BIDs in Year 2012
jit_bid_range['linux'] = [42563,52131] #BIDs in Year 2012

# Number of Dups and Non-dups for each project
project_num_pairs = dict()
project_num_pairs['eclipse'] = [int(54742/10),int(54742/10) * 4]
project_num_pairs['freedesktop'] = [int(11316/10),int(11316/10) * 4]
project_num_pairs['gcc'] = [int(7819/10),int(7819/10) * 4]
project_num_pairs['kde'] = [int(41094/10),int(41094/10) * 4]
project_num_pairs['linux'] = [int(2998/10),int(2998/10) * 4]
project_num_pairs['openoffice'] = [int(12821/10),int(12821/10) * 4]

data_dict = dict()
project_jit_bid_range = jit_bid_range[PROJECT]
num_bug_pairs = project_num_pairs[PROJECT]

JIT_FLAG_BEFORE = range(0,project_jit_bid_range[0])
JIT_FLAG_AFTER = range(project_jit_bid_range[1],10000000)

# Generate duplicate pairs and non-duplicate pairs
bug_pairs = generate_bug_pairs_before_after_jit(BUG_GROUP,project_jit_bid_range)
duplicate_pairs = bug_pairs[0]
non_duplicate_pairs = bug_pairs[1]

# Generate labels
bug_pairs_with_label = []
for pair in duplicate_pairs:
    pair = pair + ('1',)
    bug_pairs_with_label.append(pair)

for pair in non_duplicate_pairs:
    pair = pair + ('0',)
    bug_pairs_with_label.append(pair)

number_pairs = len(bug_pairs_with_label)

# Prepare pandas dataframe
columns = ["bid1","summary1","description1","pro1","com1","ver1","sev1","pri1","sts1","bid2","summary2","description2","pro2","com2","ver2","sev2","pri2","sts2","is_duplicate"]
df_b = pd.DataFrame(columns=columns)
df_b[['bid1']] = df_b[['bid1']].astype('int')
df_b[['summary1']] = df_b[['summary1']].astype('object')
df_b[['description1']] = df_b[['description1']].astype('object')
df_b[['pro1']] = df_b[['pro1']].astype('object')
df_b[['com1']] = df_b[['com1']].astype('object')
df_b[['ver1']] = df_b[['ver1']].astype('object')
df_b[['sev1']] = df_b[['sev1']].astype('object')
df_b[['pri1']] = df_b[['pri1']].astype('object')
df_b[['sts1']] = df_b[['sts1']].astype('object')

df_b[['bid2']] = df_b[['bid2']].astype('int')
df_b[['summary2']] = df_b[['summary2']].astype('object')
df_b[['description2']] = df_b[['description2']].astype('object')
df_b[['pro2']] = df_b[['pro2']].astype('object')
df_b[['com2']] = df_b[['com2']].astype('object')
df_b[['ver2']] = df_b[['ver2']].astype('object')
df_b[['sev2']] = df_b[['sev2']].astype('object')
df_b[['pri2']] = df_b[['pri2']].astype('object')
df_b[['sts2']] = df_b[['sts2']].astype('object')
df_b[['is_duplicate']]= df_b[['is_duplicate']].astype('int')
df_b.to_csv(SAVE_FILE_NAME + 'before_jit.csv', mode='a', index=False, quoting=1)

df_a = pd.DataFrame(columns=columns)
df_a[['bid1']] = df_a[['bid1']].astype('int')
df_a[['summary1']] = df_a[['summary1']].astype('object')
df_a[['description1']] = df_a[['description1']].astype('object')
df_a[['pro1']] = df_a[['pro1']].astype('object')
df_a[['com1']] = df_a[['com1']].astype('object')
df_a[['ver1']] = df_a[['ver1']].astype('object')
df_a[['sev1']] = df_a[['sev1']].astype('object')
df_a[['pri1']] = df_a[['pri1']].astype('object')
df_a[['sts1']] = df_a[['sts1']].astype('object')

df_a[['bid2']] = df_a[['bid2']].astype('int')
df_a[['summary2']] = df_a[['summary2']].astype('object')
df_a[['description2']] = df_a[['description2']].astype('object')
df_a[['pro2']] = df_a[['pro2']].astype('object')
df_a[['com2']] = df_a[['com2']].astype('object')
df_a[['ver2']] = df_a[['ver2']].astype('object')
df_a[['sev2']] = df_a[['sev2']].astype('object')
df_a[['pri2']] = df_a[['pri2']].astype('object')
df_a[['sts2']] = df_a[['sts2']].astype('object')
df_a[['is_duplicate']]= df_a[['is_duplicate']].astype('int')
df_a.to_csv(SAVE_FILE_NAME + 'after_jit.csv', mode='a', index=False, quoting=1)

count_before_dup = 0
count_before_non_dup = 0
count_after_dup = 0 
count_after_non_dup = 0

for i in range(number_pairs):
    count_before = count_before_dup + count_before_non_dup
    count_after = count_after_dup + count_after_non_dup
    count = min(count_before,count_after)
    if count % 200 == 0:
         logging.info("Number of pairs generating: {0} ----> {1}% of total number of pairs ({2})".format(count, (int)((count/sum(num_bug_pairs)) * 100), sum(num_bug_pairs)))    
    elif count == sum(num_bug_pairs):
         logging.info("Number of pairs generating: {0} ----> {1}% of total number of pairs ({2})".format(count, (int)((count/sum(num_bug_pairs)) * 100), sum(num_bug_pairs)))    
         break

    # One sample
    bug_pair = next(bug_pair_generator(bug_pairs_with_label,XML_FILE_PATH,XML_FILE_PREFIX)) 
    if bug_pair[0] in JIT_FLAG_BEFORE and bug_pair[9] in JIT_FLAG_BEFORE:
        if bug_pair[-1] == 1:
            if count_before_dup < num_bug_pairs[0]:
                df_b.loc[1] = bug_pair
                df_b.to_csv(SAVE_FILE_NAME + 'before_jit.csv', mode='a', index=False, quoting=1, header=False)
                count_before_dup += 1
        else:
            if count_before_non_dup < num_bug_pairs[1]:
                df_b.loc[1] = bug_pair
                df_b.to_csv(SAVE_FILE_NAME + 'before_jit.csv', mode='a', index=False, quoting=1, header=False)
                count_before_non_dup += 1

    elif bug_pair[0] in JIT_FLAG_AFTER and bug_pair[9] in JIT_FLAG_AFTER:
        if bug_pair[-1] == 1:
            if count_after_dup < num_bug_pairs[0]:
                df_a.loc[1] = bug_pair
                df_a.to_csv(SAVE_FILE_NAME + 'after_jit.csv', mode='a', index=False, quoting=1, header=False)
                count_after_dup += 1
        else:
            if count_after_non_dup < num_bug_pairs[1]:
                df_a.loc[1] = bug_pair
                df_a.to_csv(SAVE_FILE_NAME + 'after_jit.csv', mode='a', index=False, quoting=1, header=False)
                count_after_non_dup += 1

logging.info("Done!")
