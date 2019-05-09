# =============================================================================
# Andrew Evans (ace8p)p
# DS5559: Exploratory Text Analytics
# Final Project
# Sentiment via VADER
# =============================================================================

#%%
# =============================================================================
# Non-tuned VADER sentimetn analyzer
# =============================================================================
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()


#%%
# =============================================================================
# Initialize a new df with sentiment columns
# =============================================================================
sent = chaps
sent['neg'] = 0
sent['pos'] = 0
sent['neu'] = 0
sent['compound'] = 0
sent.columns

#%%
# =============================================================================
# Add new column for pos, neg, neutral, and compound
# =============================================================================
for i in range(len(sent.token_str)):
    sent.neg.iloc[i] = analyzer.polarity_scores(str(sent.token_str.iloc[i]))['neg']
for i in range(len(sent.token_str)):
    sent.pos.iloc[i] = analyzer.polarity_scores(str(sent.token_str.iloc[i]))['pos']
for i in range(len(sent.token_str)):
    sent.neu.iloc[i] = analyzer.polarity_scores(str(sent.token_str.iloc[i]))['neu']
for i in range(len(sent.token_str)):
    sent['compound'].iloc[i] = analyzer.polarity_scores(str(sent.token_str.iloc[i]))['compound']
#%%
    
sent.to_csv("sentiment.csv",index=True,header=True)
