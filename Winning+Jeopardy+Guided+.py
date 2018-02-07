
# coding: utf-8

# In[3]:


import re
import string
import numpy as np
import scipy.stats as stats
import pandas as pd


# In[4]:


df = pd.read_csv('jeopardy.csv')
df.head()


# In[5]:


columns = []
pattern = re.compile(r'\s+')
for each in df.columns:
    sentence = re.sub(pattern, '', each)
    columns.append(sentence)
df.columns = columns
print(df.columns)


# In[6]:


def norm_text(col):
    col = col.lower()
    col = re.sub(r'[^\w\s]','',col)
    return col
df["clean_question"] = df["Question"].apply(norm_text) 


# In[7]:


df["clean_answer"] = df["Answer"].apply(norm_text)


# In[15]:


def norm_int(col):
    col = re.sub(r'[\w\s]',"",col)
    try:
        col = int(col)
    except Exception:
        col = 0
    return col
df["clean_value"] = df["Value"].apply(norm_int)


# In[16]:


air_date = df["AirDate"]
pd.to_datetime(air_date).head()


# In[17]:


def answer_in_ques(row):
    split_answer = row["clean_answer"].split(" ")
    split_question = row["clean_question"].split(" ")
    match_count = 0 
    if "the" in split_answer:
        split_answer.remove("the")
    if len(split_answer) == 0:
        return 0
    for each in split_answer:
        if each in split_question:
            match_count += 1
    res = match_count / len(split_answer)
    return res
df["answer_in_question"] = df.apply(answer_in_ques,axis=1)

df["answer_in_question"].mean()


# In[18]:


df.sort_values('AirDate', ascending=True, inplace=True)
df.head()


# In[19]:


question_overlap = []
terms_used = []
for i, row in df.iterrows():
  split_question = row['clean_question'].split(' ')
  split_question = [q for q in split_question if len(q) > 5]
  match_count = 0
  for q in split_question:
    if q in terms_used:
      match_count += 1
    else: terms_used.append(q)
  if len(split_question) > 0:
    question_overlap.append(match_count / len(split_question))
  else:
    question_overlap.append(0)
df['question_overlap'] = question_overlap
df['question_overlap'].mean()


# In[20]:


def value_800(row):
  value = 0
  if row['clean_value'] > 800:
    value = 1
  else:
    value = 0
  return(value)


# In[21]:


df['high_value'] = df.apply(lambda row: value_800(row), axis=1)
df['high_value'].head()


# In[22]:


def question_repeat(str):
  low_count = 0
  high_count = 0
  for i, row in df.iterrows():
    if str in row['clean_question'].split(' '):
      if row['high_value']:
        high_count += 1
      else:
        low_count += 1
  return(high_count, low_count)


# In[23]:


observed_expected = []
comparison_terms = terms_used[:10]
for term in comparison_terms:
  observed_expected.append(question_repeat(term))
observed_expected


# In[24]:


high_value_count = len(df[df['high_value'] == 1])
high_value_count


# In[26]:


low_value_count = len(df[df['high_value'] == 0])


# In[28]:


chi_squared = []
for e in observed_expected:
  total = sum(e)
  total_prob = total / df.shape[0]
  expected_high_count = total_prob * high_value_count
  expected_low_count = total_prob * low_value_count
  expected = np.array([expected_high_count, expected_low_count])
  observed = np.array([e[0], e[1]])
  print(observed, expected)
  chi_squared.append(stats.chisquare(e, expected))
#chi_squared

