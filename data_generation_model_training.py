import pandas as pd
from modules import generate_bug_pairs, extract_bug_report_information, bug_pair_generator
import random, re, os, sys, json, logging

''' Generage model training data for training and test HINDBR'''


########## Settings ###############
PROJECT = 'linux'
###################################


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
print('loading...')

logging.info("Generating model training samples for project: " + PROJECT)

with open('data/xmlfile_path.setting') as f:
    XML_FILE_PATH_DICT = json.load(f)

XML_FILE_PATH = XML_FILE_PATH_DICT[PROJECT][0]
XML_FILE_PREFIX = XML_FILE_PATH_DICT[PROJECT][1]

BUG_GROUP = 'data/bug_report_groups/' + PROJECT + '_all.pkl'
SAVE_FILE_NAME = 'data/model_training/' + PROJECT + '_all.csv'

if os.path.isfile(SAVE_FILE_NAME):
    os.remove(SAVE_FILE_NAME)

# Generate duplicate pairs and non-duplicate pairs
bug_pairs = generate_bug_pairs(BUG_GROUP)
duplicate_pairs = bug_pairs[0]
non_duplicate_pairs = bug_pairs[1]

# Generate labels
dup_bug_pairs_with_label = [pair + ('1',) for pair in duplicate_pairs]
non_dup_bug_pairs_with_label = [pair + ('0',) for pair in non_duplicate_pairs]

if PROJECT == 'eclipse':
    # Eclipse 
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.4))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'freedesktop':
    # Freedesktop
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.3))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'gcc':
    # GCC
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.2))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'gnome':
    # GNOME
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.005))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'kde':
    # KDE
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.03))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'libreoffice':
    # LibreOffice
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.2))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'linux':
    # Linux kernel
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 1))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'llvm':
    # LLVM
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 1))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)
elif PROJECT == 'openoffice':
    # OpenOffice
    dup_bug_pairs_with_label = random.sample(dup_bug_pairs_with_label, int(len(dup_bug_pairs_with_label) * 0.15))
    non_dup_bug_pairs_with_label = random.sample(non_dup_bug_pairs_with_label, len(dup_bug_pairs_with_label) * 4)


bug_pairs_with_label = dup_bug_pairs_with_label + non_dup_bug_pairs_with_label
random.shuffle(bug_pairs_with_label)


logging.info('Total pairs: ' + str(len(bug_pairs_with_label))) 
# Prepare pandas dataframe
columns = ["id","bid1","summary1","description1","pro1","com1","ver1","sev1","pri1", "sts1", "bid2","summary2","description2","pro2","com2","ver2","sev2","pri2", "sts2", "is_duplicate"]
df = pd.DataFrame(columns=columns)
df[['id']] = df[['id']].astype('int')
df[['bid1']] = df[['bid1']].astype('int')
df[['summary1']] = df[['summary1']].astype('object')
df[['description1']] = df[['description1']].astype('object')
df[['pro1']] = df[['pro1']].astype('object')
df[['com1']] = df[['com1']].astype('object')
df[['ver1']] = df[['ver1']].astype('object')
df[['sev1']] = df[['sev1']].astype('object')
df[['pri1']] = df[['pri1']].astype('object')
df[['sts1']] = df[['sts1']].astype('object')

df[['bid2']] = df[['bid2']].astype('int')
df[['summary2']] = df[['summary2']].astype('object')
df[['description2']] = df[['description2']].astype('object')
df[['pro2']] = df[['pro2']].astype('object')
df[['com2']] = df[['com2']].astype('object')
df[['ver2']] = df[['ver2']].astype('object')
df[['sev2']] = df[['sev2']].astype('object')
df[['pri2']] = df[['pri2']].astype('object')
df[['sts2']] = df[['sts2']].astype('object')

df[['is_duplicate']]= df[['is_duplicate']].astype('int')
df.to_csv(SAVE_FILE_NAME, mode='a', index=False, quoting=1)

# Prepare sameple data
number_pairs = len(bug_pairs_with_label)
for i in range(number_pairs):
    if i % 1000 == 0:
        logging.info("Number of generated pairs: {0} ----> {1}% of total number of pairs ({2})".format(i, (int)((i/number_pairs) * 100), number_pairs))
    # One sample
    df.loc[1] = [i] + next(bug_pair_generator(bug_pairs_with_label,XML_FILE_PATH,XML_FILE_PREFIX))
    # Save training data
    df.to_csv(SAVE_FILE_NAME, mode='a', index=False, quoting=1, header=False)
#    print(sys.getsizeof(df))
#    break

logging.info("Saving data to {0}".format(SAVE_FILE_NAME))
#df.to_csv(SAVE_FILE_NAME, index=False, quoting=1)
logging.info("Done!")
