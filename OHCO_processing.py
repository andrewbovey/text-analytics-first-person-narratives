# =============================================================================
# Andrew Evans (ace8p)
# DS5559: Exploratory Text Analytics
# Final Project
# Corpus Processing to OHCO
# =============================================================================
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import re
#%matplotlib inline

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('tagsets')
nltk.download('wordnet')


#%%
OHCO = ['chap_num', 'para_num', 'sent_num', 'token_num']
CHAPS = OHCO[:1]
PARAS = OHCO[:2]
SENTS = OHCO[:3]

#%%
def text_to_tokens(
                   src_file,
                   body_start=0, 
                   body_end=-1, 
                   chap_pat=r'^\s*(Chapter)\s[0-9]', 
                   para_pat=r'\n+', 
                   sent_pat=r'([.;?!"“”]+)', 
                   token_pat=r'([\W_]+)'):

    # Text to lines
    lines = open(src_file, 'r', encoding='utf-8').readlines()
    lines = lines[body_start - 1 : body_end + 1]
    df = pd.DataFrame({'line_str':lines})
    df.index.name = 'line_id'
    #del(lines)
    
    # FIX CHARACTERS TO IMPROVE TOKENIZATION
    df.line_str = df.line_str.str.replace('—', ' — ')
    df.line_str = df.line_str.str.replace('-', ' - ')

    # Lines to Chapters
    mask = df.line_str.str.match(chap_pat)
    df.loc[mask, 'chap_id'] = df.apply(lambda x: x.name, 1)
    df.chap_id = df.chap_id.ffill().astype('int')
    chap_ids = df.chap_id.unique().tolist()
    df['chap_num'] = df.chap_id.apply(lambda x: chap_ids.index(x))
    chaps = df.groupby('chap_num')\
        .apply(lambda x: ''.join(x.line_str))\
        .to_frame()\
        .rename(columns={0:'chap_str'})
    #del(df)

    # Chapters to Paragraphs
    paras = chaps.chap_str.str.split(para_pat, expand=True)\
        .stack()\
        .to_frame()\
        .rename(columns={0:'para_str'})
    paras.index.names = PARAS
    paras.para_str = paras.para_str.str.strip()
    paras.para_str = paras.para_str.str.replace(r'\n', ' ')
    paras.para_str = paras.para_str.str.replace(r'\s+', ' ')
    paras = paras[~paras.para_str.str.match(r'^\s*$')]
    #del(chaps)

    # Paragraphs to Sentences
#     sents = paras.para_str.str.split(sent_pat, expand=True)\
    sents = paras.para_str\
        .apply(lambda x: pd.Series(nltk.sent_tokenize(x)))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'sent_str'})
    sents.index.names = SENTS
    #del(paras)

    # Sentences to Tokens
#     tokens = sents.sent_str.str.split(token_pat, expand=True)\
    tokens = sents.sent_str\
        .apply(lambda x: pd.Series(nltk.pos_tag(nltk.word_tokenize(x))))\
        .stack()\
        .to_frame()\
        .rename(columns={0:'pos_tuple'})
    tokens.index.names = OHCO
    del(sents)
    
    tokens['pos'] = tokens.pos_tuple.apply(lambda x: x[1])
    tokens['token_str'] = tokens.pos_tuple.apply(lambda x: x[0])
    tokens = tokens.drop('pos_tuple', 1)

    # Tag punctuation and numbers
    tokens['punc'] = tokens.token_str.str.match(r'^[\W_]*$').astype('int')
    tokens['num'] = tokens.token_str.str.match(r'^.*\d.*$').astype('int')
    
    # Extract vocab with minimal normalization
    WORDS = (tokens.punc == 0) & (tokens.num == 0)
    tokens.loc[WORDS, 'term_str'] = tokens.token_str.str.lower()\
        .str.replace(r'["_*.]', '')
    
    vocab = tokens[tokens.punc == 0].term_str.value_counts().to_frame()\
        .reset_index()\
        .rename(columns={'index':'term_str', 'term_str':'n'})
    vocab = vocab.sort_values('term_str').reset_index(drop=True)
    vocab.index.name = 'term_id'
    
    # Get priors for V
    vocab['p'] = vocab.n / vocab.n.sum()
    
    # Add stems
    stemmer = nltk.stem.porter.PorterStemmer()
    vocab['port_stem'] = vocab.term_str.apply(lambda x: stemmer.stem(x))
    
    # Define stopwords
    sw = pd.DataFrame({'x':1}, index=nltk.corpus.stopwords.words('english'))
    vocab['stop'] = vocab.term_str.map(sw.x).fillna(0).astype('int')
    del(sw)
            
    # Add term_ids to tokens 
    tokens['term_id'] = tokens['term_str'].map(vocab.reset_index()\
        .set_index('term_str').term_id).fillna(-1).astype('int')

    return tokens, vocab


#%%
import os
os.chdir('/Users/andrewevans/DSI/Spring2019/DS5559/project/first-person-narratives-american-south/data/texts')
docs = os.listdir(os.getcwd())
docs.sort()

#%%
# =============================================================================
# 
# Tokenization & Vocab for each Narrative
# 
# 
# =============================================================================
    
#%%
src_file_name = 'fpn-andrews-andrews.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 29,
    body_end = 628,
    chap_pat = '^\s*(?:December|Dec|January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|August|Aug|September|Sep|October|Oct|November|Nov).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ashby-ashby.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 707,
    chap_pat = r'^\s*(?:CHAPTER|CONCLUSION|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-aughey-aughey.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 40,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-avary-avary.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 1639,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-avirett-avirett.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 16,
    body_end = 556,
    chap_pat = r'^\s*(?:CHAPTER|EPILOGUE).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-bagby-bagby.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 43,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-balch-balch.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 155,
    chap_pat = r'^\s*(?:CHAPTER|LETTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ball-ball.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 13,
    body_end = 782,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-battle-lee.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 17,
    body_end = 1365,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-beard-beard.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 17,
    body_end = 1253,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-betts-betts.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 20,
    body_end = 1451,
    chap_pat = r'^\s*(?:EXPERIENCE|Monday|Tuesday|Wednesday|Thursday|Friday|January|Jan|February|Feb|March|Mar|April|Apr|May|June|Jun|July|Jul|September|Sep|October|Oct|November|Nov|December|Dec).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-biggs-biggs.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 150,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}|[0-9]st|[0-9]th|[0-9]rd).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-blackford-blackford.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 68,
    body_end = 438,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-boggs-boggs.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 53,
    body_end = 251,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-bokum-bokum.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 28,
    chap_pat = r'^\s*(?:A REFUGEE|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-boyd1-boyd1.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 170,
    body_end = 903,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-boyd2-boyd2.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 729,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-branch-branch.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 13,
    body_end = 454,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-brownd-dbrown.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 1554,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-brownw-brown.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 78,
    body_end = 382,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-bruce-bruce.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 17,
    body_end = 398,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-bryan-bryan.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 164,
    chap_pat = r'^\s*(?:LETTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-burge-lunt.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 140,
    chap_pat = r'^\s*(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER|SUNDAY|MONDAY|TUESDAY|WEDNESDAY|THURSDAY|FRIDAY|SATURDAY).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-burton-burton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 795,
    chap_pat = r'^\s*(?:[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-burtont-burton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 33,
    body_end = 384,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-burwell-burwell.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 706,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-caldwell-caldwell.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 122,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-carroll-carroll.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 134,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-chapter-chapter.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 280,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-clay-clay.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 1275,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-clinkscales-clinksc.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 14,
    body_end = 743,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-collis-collis.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 83,
    chap_pat = r'^\s*(?:(A WOMAN)|(A FEW)|(GENL)|LINCOLN).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-compton-compton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 13,
    body_end = 343,
    chap_pat = r'^\s*(?:INTRODUCTION|CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-copley-copley.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 26,
    body_end = 359,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-crumpton-crumpton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 25,
    body_end = 480,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-curry-curry.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 51,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-davisr-davis.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 789,
    chap_pat = r'^\s*(?:I|II|III|IV|V|VI|VII|VIII).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-dawson-dawson.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 38,
    body_end = 1026,
    chap_pat = r'^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December|Sunday|Monday|Tuesday|Wednesday|Thursay|Friday|Saturday).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-desaussure-desaussure.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 230,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-dimitry-dimitry.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 58,
    body_end = 360,
    chap_pat = r'^\s*(?:FREEDOM|(AN INCIDENT)|(MRS.)|(A CONFEDERATE)|(A WOMAN)|(A RAMBLING)|(DAVIDSON)|(A TRUE)|(PART II)|(JUNE 3)|(THE HALT)|(FOUR RICHMOND)|(THE LOUISIANA)|(JUDAH P)|(MEMMINGER)|(IN THE fall)|(ON THE 26th)|(THE BATTLE OF)).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-dupre-dupre.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 24,
    body_end = 1068,
    chap_pat = r'^\s*(?:CHAPTER|PREFACE).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-early-early.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 24,
    body_end = 1487,
    chap_pat = r'^\s*(?:CHAPTER|AUTOBIOGRAPHICAL).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-edmondson-edmondson.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 2,
    body_end = 615,
    chap_pat = r'^\s*(?:January|February|March|April|May|June|July|August|September|October|November|December).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-edwards-edwards.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 31,
    body_end = 430,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-eggleston-eggleston.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 12,
    body_end = 295,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-fearn-fearn.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 21,
    body_end = 283,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-fee-fee.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 28,
    body_end = 426,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-felton-felton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 43,
    body_end = 1337,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ferebee-ferebee.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 57,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ford-ford.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 328,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-gilman-gilman.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 1916,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-gordon-gordon.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 47,
    body_end = 1097,
    chap_pat = r'^\s*(?:CHAPTER|INTRODUCTION).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-grandy-grandy.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 94,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-green-green.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 27,
    body_end = 1156,
    chap_pat = r'^\s*(?:CHAPTER|January|February|March|April|May|June|July|August|September|October|November|December).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-greenhow-greenhow.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 796,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-grimball-grimball.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 3,
    body_end = 487,
    chap_pat = r'^\s*(?:January|Jan|February|Feb|March|Mar|Apr|April|May|Jun|June|Jul|July|August|Aug|September|Sep|October|Oct|November|Nov|December|Dec).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-grimes-grimes.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 22,
    body_end = 325,
    chap_pat = r'.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-hamill-hamill.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 141,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-harland-harland.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 16,
    body_end = 2662,
    chap_pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'

)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-harrison-harrison.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 1083,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-holden-holden.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 14,
    body_end = 529,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-hortonlife-horton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 1,
    body_end = 3187,
    chap_pat = r'^\s*\n+.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-hortonpoem-hortonpoem.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 3187,
    chap_pat = r'^\s*\n+.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-houghton-houghton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 65,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-howard-howard.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 2,
    body_end = 22,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-hughes-hughes.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 292,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-hughest-hughes.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 60,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-jackson-jackson.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 44,
    body_end = 517,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-jacobs-jacobs.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 839,
    chap_pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-janney-janney.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 12,
    body_end = 1460,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-jervey-jervey.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 3187,
    chap_pat = r'^\s*(?:Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|January|February|March|April|May|June|July|August|September|October|November).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-johnstond-johnston.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 22,
    body_end = 571,
    chap_pat = r'^\s*(?:Chapter|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-johnstonr-johnston.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 252,
    chap_pat = r'^\s*(?:INTRODUCTION|CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-jones-jones.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 25,
    body_end = 131,
    chap_pat = r'^\s*(?:CHAPTER|\n+).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-jonescharles-jones.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 479,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-kearney-kearney.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 16,
    body_end = 772,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-kell-kell.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 19,
    body_end = 655,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-lane-lane.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 27,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-leconte-leconte.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 24,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-leconteemma-leconte.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 3,
    body_end = 3187,
    chap_pat = r'^\s*(?:Dec|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Sat|Sun|Monday|Tues|Wed|Thu|Fri).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-leigh-leigh.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 188,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|NO.).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-leon-leon.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 378,
    chap_pat = r'^\s*(?:CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-lomax-lomax.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-macon-macon.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 208,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mallard-mallard.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 40,
    body_end = 432,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-malone-malone.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 68,
    body_end = 617,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mason-mason.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 12,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-maury-maury.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 668,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mckim-mckim.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{3,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mcleary-mcleary.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 3187,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-merrick-merrick.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 720,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mitchel-mitchel.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 3187,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-montgomery-montgom.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-morgan-morgan.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-morganjames-morgan.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 16,
    body_end = 1223,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-mosby-mosby.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 44,
    body_end = 1187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-moton-moton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 14,
    body_end = 375,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-negnurse-negnurse.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 15,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-negpeon-negpeon.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 17,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-northup-northup.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 33,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-oconnor-oconnor.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-olive-olive.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 40,
    body_end = 1092,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-patton-patton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 4,
    body_end = 58,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-paxton-paxton.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 23,
    body_end = 409,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-pickens-pickens.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 3187,
    chap_pat = r'^\s*(?:(I.)|(II.)|(III.)|(IV.)|(V.)|(VI.)|(VII.)|(VIII.)).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-pickett-pickett.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 99,
    body_end = 3187,
    chap_pat = r'^\s*(?:(PART ONE)|(PART TWO)|(PART THREE)|(PART FOUR)|(PART FIVE)).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-plake-plake.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 13,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-pringle-pringle.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 21,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|December|November|October|September|August|July|June|May|April|March|February|January|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-pryor-pryor.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 64,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-rankin-rankin.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 47,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ripley-ripley.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 12,
    body_end = 3187,
    chap_pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ripleyflag-ripley.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-robinson-robinson.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 22,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-robson-robson.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 9,
    body_end = 359,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-roper-roper.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 27,
    body_end = 103,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-royall-royall.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 5,
    body_end = 460,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-sawyer-sawyer.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 12,
    body_end = 132,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-saxon-saxon.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 59,
    body_end = 3187,
    chap_pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-shepherd-shepherd.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 31,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-sherrill-sherrill.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 21,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-smedes-smedes.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 49,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-steward-steward.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 22,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-stiles-stiles.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-stone-stone.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-taylor-taylor.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-thomas-thomas.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 3187,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-velazquez-velazquez.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 31,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-veney-veney.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 18,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-ward-ward.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 21,
    body_end = 41,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-washeducation-washing.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-washington-washing.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 19,
    body_end = 50,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-watterson1-watterson1.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 15,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-watterson2-watterson2.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 7,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-whitaker-whitaker.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 3187,
    chap_pat = r'^\s*(?:Finally|(When I was only six)|(At the age of ten)|(My mother was a Hogg)|(I am now sixteen years)|(On one trip to Cloris)).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-williams-williams.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 72,
    body_end = 178,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-wise-wise.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-wood-wood.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 27,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-worsham-worsham.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 10,
    body_end = 3187,
    chap_pat = r'^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-wright-wright.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 8,
    body_end = 725,
    chap_pat = r'^\s*(?:CHAPTER|Epilogue).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-wrightmarcus-wright.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 6,
    body_end = 95,
    chap_pat = r'^\s*.*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-wyeth-wyeth.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 17,
    body_end = 1574,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-young-young.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 61,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|INTRODUCTION).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
src_file_name = 'fpn-zettler-zettler.txt'
cfg = dict(
    src_file = src_file_name,
    body_start = 11,
    body_end = 3187,
    chap_pat = r'^\s*(?:CHAPTER|[A-Z]{2,}).*$'
)
nmK = src_file_name.split('-')[1] + 'K'
nmV = src_file_name.split('-')[1] + 'V'

globals()[nmK], globals()[nmV] = text_to_tokens(**cfg)

#%%
del(docs)
del(cfg)

del(chap_ids)
del(lines)
del(nmK)
del(nmV)
del(src_file)
del(src_file_name)

#%%
andrewsK['doc_name'] = 'andrewsK'
ashbyK['doc_name'] = 'ashbyK'
augheyK['doc_name'] = 'augheyK'
avaryK['doc_name'] = 'avaryK'
avirettK['doc_name'] = 'avirettK'
bagbyK['doc_name'] = 'bagbyK'
balchK['doc_name'] = 'balchK'
ballK['doc_name'] = 'ballK'
battleK['doc_name'] = 'battleK'
beardK['doc_name'] = 'beardK'
bettsK['doc_name'] = 'bettsK'
biggsK['doc_name'] = 'biggsK'
blackfordK['doc_name'] = 'blackfordK'
boggsK['doc_name'] = 'boggsK'
bokumK['doc_name'] = 'bokumK'
boyd1K['doc_name'] = 'boyd1K'
boyd2K['doc_name'] = 'boyd2K'
branchK['doc_name'] = 'branchK'
browndK['doc_name'] = 'browndK'
brownwK['doc_name'] = 'brownwK'
bruceK['doc_name'] = 'bruceK'
bryanK['doc_name'] = 'bryanK'
burgeK['doc_name'] = 'burgeK'
burtonK['doc_name'] = 'burtonK'
burtontK['doc_name'] = 'burtontK'
burwellK['doc_name'] = 'burwellK'
caldwellK['doc_name'] = 'caldwellK'
carrollK['doc_name'] = 'carrollK'
chapterK['doc_name'] = 'chapterK'
clayK['doc_name'] = 'clayK'
clinkscalesK['doc_name'] = 'clinkscalesK'
collisK['doc_name'] = 'collisK'
comptonK['doc_name'] = 'comptonK'
copleyK['doc_name'] = 'copleyK'
crumptonK['doc_name'] = 'crumptonK'
curryK['doc_name'] = 'curryK'
davisrK['doc_name'] = 'davisrK'
dawsonK['doc_name'] = 'dawsonK'
desaussureK['doc_name'] = 'desaussureK'
dimitryK['doc_name'] = 'dimitryK'
dupreK['doc_name'] = 'dupreK'
earlyK['doc_name'] = 'earlyK'
edmondsonK['doc_name'] = 'edmondsonK'
edwardsK['doc_name'] = 'edwardsK'
egglestonK['doc_name'] = 'egglestonK'
fearnK['doc_name'] = 'fearnK'
feeK['doc_name'] = 'feeK'
feltonK['doc_name'] = 'feltonK'
ferebeeK['doc_name'] = 'ferebeeK'
fordK['doc_name'] = 'fordK'
gilmanK['doc_name'] = 'gilmanK'
gordonK['doc_name'] = 'gordonK'
grandyK['doc_name'] = 'grandyK'
greenK['doc_name'] = 'greenK'
greenhowK['doc_name'] = 'greenhowK'
grimballK['doc_name'] = 'grimballK'
grimesK['doc_name'] = 'grimesK'
hamillK['doc_name'] = 'hamillK'
harlandK['doc_name'] = 'harlandK'
harrisonK['doc_name'] = 'harrisonK'
holdenK['doc_name'] = 'holdenK'
hortonlifeK['doc_name'] = 'hortonlifeK'
hortonpoemK['doc_name'] = 'hortonpoemK'
houghtonK['doc_name'] = 'houghtonK'
howardK['doc_name'] = 'howardK'
hughesK['doc_name'] = 'hughesK'
jacksonK['doc_name'] = 'jacksonK'
jacobsK['doc_name'] = 'jacobsK'
janneyK['doc_name'] = 'janneyK'
jerveyK['doc_name'] = 'jerveyK'
johnstondK['doc_name'] = 'johnstondK'
johnstonrK['doc_name'] = 'johnstonrK'
jonesK['doc_name'] = 'jonesK'
jonescharlesK['doc_name'] = 'jonescharlesK'
kearneyK['doc_name'] = 'kearneyK'
kellK['doc_name'] = 'kellK'
laneK['doc_name'] = 'laneK'
leconteK['doc_name'] = 'leconteK'
leconteemmaK['doc_name'] = 'leconteemmaK'
leighK['doc_name'] = 'leighK'
leonK['doc_name'] = 'leonK'
lomaxK['doc_name'] = 'lomaxK'
maconK['doc_name'] = 'maconK'
mallardK['doc_name'] = 'mallardK'
maloneK['doc_name'] = 'maloneK'
masonK['doc_name'] = 'masonK'
mauryK['doc_name'] = 'mauryK'
mckimK['doc_name'] = 'mckimK'
mclearyK['doc_name'] = 'mclearyK'
merrickK['doc_name'] = 'merrickK'
mitchelK['doc_name'] = 'mitchelK'
montgomeryK['doc_name'] = 'montgomeryK'
morganK['doc_name'] = 'morganK'
morganjamesK['doc_name'] = 'morganjamesK'
mosbyK['doc_name'] = 'mosbyK'
motonK['doc_name'] = 'motonK'
negnurseK['doc_name'] = 'negnurseK'
negpeonK['doc_name'] = 'negpeonK'
northupK['doc_name'] = 'northupK'
oconnorK['doc_name'] = 'oconnorK'
oliveK['doc_name'] = 'oliveK'
pattonK['doc_name'] = 'pattonK'
paxtonK['doc_name'] = 'paxtonK'
pickensK['doc_name'] = 'pickensK'
pickettK['doc_name'] = 'pickettK'
plakeK['doc_name'] = 'plakeK'
pringleK['doc_name'] = 'pringleK'
pryorK['doc_name'] = 'pryorK'
rankinK['doc_name'] = 'rankinK'
ripleyK['doc_name'] = 'ripleyK'
ripleyflagK['doc_name'] = 'ripleyflagK'
robinsonK['doc_name'] = 'robinsonK'
robsonK['doc_name'] = 'robsonK'
roperK['doc_name'] = 'roperK'
royallK['doc_name'] = 'royallK'
sawyerK['doc_name'] = 'sawyerK'
saxonK['doc_name'] = 'saxonK'
shepherdK['doc_name'] = 'shepherdK'
sherrillK['doc_name'] = 'sherrillK'
smedesK['doc_name'] = 'smedesK'
stewardK['doc_name'] = 'stewardK'
stilesK['doc_name'] = 'stilesK'
stoneK['doc_name'] = 'stoneK'
taylorK['doc_name'] = 'taylorK'
thomasK['doc_name'] = 'thomasK'
velazquezK['doc_name'] = 'velazquezK'
veneyK['doc_name'] = 'veneyK'
washeducationK['doc_name'] = 'washeducationK'
washingtonK['doc_name'] = 'washingtonK'
watterson1K['doc_name'] = 'watterson1K'
watterson2K['doc_name'] = 'watterson2K'
whitakerK['doc_name'] = 'whitakerK'
williamsK['doc_name'] = 'williamsK'
wiseK['doc_name'] = 'wiseK'
woodK['doc_name'] = 'woodK'
worshamK['doc_name'] = 'worshamK'
wrightK['doc_name'] = 'wrightK'
wrightmarcusK['doc_name'] = 'wrightmarcusK'
wyethK['doc_name'] = 'wyethK'
youngK['doc_name'] = 'youngK'
zettlerK['doc_name'] = 'zettlerK'

#%%
# =============================================================================
# alltext is a concatenation of every narrative's token dataframe
# =============================================================================


alltext = pd.concat([ashbyK	,
        augheyK	,
        avaryK	,
        avirettK	,
        bagbyK	,
        balchK	,
        ballK	,
        battleK	,
        beardK	,
        bettsK	,
        biggsK	,
        blackfordK	,
        boggsK	,
        bokumK	,
        boyd1K	,
        boyd2K	,
        branchK	,
        browndK	,
        brownwK	,
        bruceK	,
        bryanK	,
        burgeK	,
        burtonK	,
        burtontK	,
        burwellK	,
        caldwellK	,
        carrollK	,
        chapterK	,
        clayK	,
        clinkscalesK	,
        collisK	,
        comptonK	,
        copleyK	,
        crumptonK	,
        curryK	,
        davisrK	,
        dawsonK	,
        desaussureK	,
        dimitryK	,
        dupreK	,
        earlyK	,
        edmondsonK	,
        edwardsK	,
        egglestonK	,
        fearnK	,
        feeK	,
        feltonK	,
        ferebeeK	,
        fordK	,
        gilmanK	,
        gordonK	,
        grandyK	,
        greenK	,
        greenhowK	,
        grimballK	,
        grimesK	,
        hamillK	,
        harlandK	,
        harrisonK	,
        holdenK	,
        hortonlifeK	,
        hortonpoemK	,
        houghtonK	,
        howardK	,
        hughesK	,
        jacksonK	,
        jacobsK	,
        janneyK	,
        jerveyK	,
        johnstondK	,
        johnstonrK	,
        jonesK	,
        jonescharlesK	,
        kearneyK	,
        kellK	,
        laneK	,
        leconteK	,
        leconteemmaK	,
        leighK	,
        leonK	,
        lomaxK	,
        maconK	,
        mallardK	,
        maloneK	,
        masonK	,
        mauryK	,
        mckimK	,
        mclearyK	,
        merrickK	,
        mitchelK	,
        montgomeryK	,
        morganK	,
        morganjamesK	,
        mosbyK	,
        motonK	,
        negnurseK	,
        negpeonK	,
        northupK	,
        oconnorK	,
        oliveK	,
        pattonK	,
        paxtonK	,
        pickensK	,
        pickettK	,
        plakeK	,
        pringleK	,
        pryorK	,
        rankinK	,
        ripleyK	,
        ripleyflagK	,
        robinsonK	,
        robsonK	,
        roperK	,
        royallK	,
        sawyerK	,
        saxonK	,
        shepherdK	,
        sherrillK	,
        smedesK	,
        stewardK	,
        stilesK	,
        stoneK	,
        taylorK	,
        thomasK	,
        velazquezK	,
        veneyK	,
        washeducationK	,
        washingtonK	,
        watterson1K	,
        watterson2K	,
        whitakerK	,
        williamsK	,
        wiseK	,
        woodK	,
        worshamK	,
        wrightK	,
        wrightmarcusK	,
        wyethK	,
        youngK	,
        zettlerK])

#%%

alltext.reset_index(inplace=True)
alltext.head(10)

#%%

alltext.chap_num = alltext.chap_num.apply(str)


#%%

alltext['doc_id'] = alltext[['doc_name', 'chap_num']].apply(lambda x: ''.join(x), axis=1)

#%%
alltext.columns

#%%
# =============================================================================
# alltext has a doc_id which identifies which narrative+chapter combination it came from
# =============================================================================

alltext.set_index(['doc_id','chap_num','para_num','sent_num','token_num'], inplace = True)
alltext.head()

#%%
# =============================================================================
# Chunking Chapters
# =============================================================================

def gather_chunks(df, div_names, doc_str = 'token_str', sep=''):
    chunks = df.groupby(div_names)[doc_str].apply(lambda x: x.str.cat(sep=sep))
    chunks.columns = ['doc_content']
    return chunks.to_frame()


chaps = gather_chunks(alltext, ['doc_id'], sep=' ')

#%%

# =============================================================================
# Generating the Vocab dataframe for the entire corpus
# =============================================================================
vocab = alltext[alltext.punc == 0]
vocab = vocab[vocab.num == 0]

#%%
vocab2 = vocab.term_str.value_counts().to_frame()
vocab2 = vocab2.reset_index()
vocab2 = vocab2.rename(columns={'index':'term_str', 'term_str':'n'})

#%%
vocab2.index.rename('term_id', inplace=True)
vocab2.head()
#%%
vocab2['p'] = vocab2.n / vocab2.n.sum()
stemmer = nltk.stem.porter.PorterStemmer()
vocab2['port_stem'] = vocab2.term_str.apply(lambda x: stemmer.stem(x))

sw = pd.DataFrame({'x':1}, index=nltk.corpus.stopwords.words('english'))
vocab2['stop'] = vocab2.term_str.map(sw.x).fillna(0).astype('int')
#%%

vocab = vocab2[vocab2.stop == 0]

#%%
#Save Dataframes of Tokens, Vocab, and Chapters to CSV
alltext.to_csv("tokens.csv",index=True,header=True)
chaps.to_csv("chaps.csv",index=True,header=True)
vocab.to_csv("vocab.csv",index=True,header=True)