#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


train = pd.read_csv('titanic/train.csv')
test = pd.read_csv('titanic/test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# In[6]:


train.shape


# In[7]:


test.shape


# In[8]:


train.info()


# In[9]:


test.info()


# In[10]:


train.isnull().sum()


# In[11]:


test.isnull().sum()


# In[12]:


train.describe()


# In[13]:


train[train['Age'] <1]


# In[14]:


train[(train['Age']>=1)&(train['Age']<=2)]


# In[15]:


get_ipython().system('pip install plotly chart_studio')


# In[16]:


get_ipython().system('pip install cufflinks')


# In[17]:


import chart_studio.plotly as py
import cufflinks as cf
cf.go_offline(connected=True)


# In[18]:


cf.help('heatmap')


# In[20]:


train.corr().iplot(kind='heatmap',colorscale='Blues')


# In[21]:


def create_dataframe_survived_rate(feature) :
    survived = train[train['Survived']==1][feature].value_counts()
    dead = train[train['Survived']==0][feature].value_counts()
    df_graph = pd.DataFrame([survived,dead])
    df_graph.index = ['Survived', 'Dead']
    
    return df_graph


# In[22]:


survived = train[train['Survived']==1]['Sex'].value_counts()
survived


# In[23]:


dead = train[train['Survived']==0]['Sex'].value_counts()
dead


# In[24]:


df = pd.DataFrame([survived, dead])
df


# In[26]:


df.index = ['Survived', 'Dead']
df


# In[27]:


df = create_dataframe_survived_rate('Sex')
df.iplot(kind='bar')


# ### 범주형 데이터인 Sex, Pclass, Embarked 데이터 분류
# 수준별로 데이터 갯수를 세어서 그래프 그리기를 진행함

# In[31]:


train['Sex'].value_counts()


# In[32]:


train['Sex'].isnull().sum()


# In[33]:


df = create_dataframe_survived_rate('Sex')


# In[34]:


df.iplot(kind='bar')


# In[36]:


df.iplot(kind='bar', barmode='stack')


# In[41]:


df['tag'] = df.index
#tag 칼럼 추가 pie 차트를 그릴 때 사용하기 위함!


# In[38]:


df


# In[40]:


df.iplot(kind='pie', labels = 'tag', values = 'female')


# In[42]:


df.iplot(kind='pie', labels='tag', values='male')


# In[43]:


train['Pclass'].value_counts()


# In[44]:


train['Pclass'].isnull().sum()


# In[45]:


df = create_dataframe_survived_rate('Pclass')


# In[46]:


df.iplot(kind='bar', barmode='stack')


# In[48]:


df_t = df.transpose()
#tramspose 함수를 활용하여 데이터 프레임 행과 열을 바꿈
#각 등급별 survived 비율을 확인해보기 위해서


# In[49]:


df


# In[50]:


df_t


# In[51]:


df_t.iplot(kind='bar', barmode='stack', dimensions=(800,600))


# In[57]:


#subplot을 활용하면 여러개의 그래프를 한 번에 볼 수 있음
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.offline as pyo
pyo.init_notebook_mode()

#domain = pie
fig = make_subplots(rows=1,cols=3,specs=[[{'type':'domain'},{'type':'domain'},{'type':'domain'}]], subplot_titles=['1 class', '2 class', '3 class'])
fig.add_trace(go.Pie(labels=df.index, values=df[1]), row=1, col=1)
fig.add_trace(go.Pie(labels=df.index, values=df[2]), row=1, col=2)
fig.add_trace(go.Pie(labels=df.index, values=df[3]), row=1, col=3)

fig.show()


# In[58]:


train['Embarked'].value_counts()


# In[59]:


train['Embarked'].isnull().sum()


# In[60]:


df = create_dataframe_survived_rate('Embarked')
df.iplot(kind='bar')


# In[61]:


df_t = df.transpose()


# In[62]:


df_t.iplot(kind='bar',barmode='stack')


# In[63]:


df


# In[64]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.offline as pyo
pyo.init_notebook_mode()

#domain = pie
fig = make_subplots(rows=1,cols=3,specs=[[{'type':'domain'},{'type':'domain'},{'type':'domain'}]], subplot_titles=['S', 'C', 'Q'])
fig.add_trace(go.Pie(labels=df.index, values=df['S']), row=1, col=1)
fig.add_trace(go.Pie(labels=df.index, values=df['C']), row=1, col=2)
fig.add_trace(go.Pie(labels=df.index, values=df['Q']), row=1, col=3)

fig.show()


# ### 수치형 데이터 분석

# In[66]:


train['Age'].describe()


# In[67]:


train['Age'].isnull().sum()


# In[68]:


cf.help()


# In[69]:


survived = train[train['Survived']==1]['Age']
dead = train[train['Survived']==0]['Age']


# In[70]:


df = pd.concat([survived,dead], axis=1, keys=['Survived', 'Dead'])


# In[71]:


df.iplot(kind='histogram', histfunc='count')


# In[73]:


train['SibSp'].describe()


# In[74]:


train['SibSp'].isnull().sum()


# In[76]:


survived=train[train['Survived']==1]['SibSp']
dead = train[train['Survived']==0]['SibSp']
df = pd.concat([survived, dead], axis=1, keys=['Survived', 'Dead'])


# In[78]:


df.iplot(kind='histogram', histfunc='count')


# In[81]:


train['Parch'].describe()


# In[82]:


train['Parch'].isnull().sum()


# In[83]:


survived = train[train['Survived']==1]['Parch']
dead = train[train['Survived']==0]['Parch']


# In[84]:


df = pd.concat([survived, dead], axis=1, keys=['Survived', 'Dead'])
df.iplot(kind='histogram', histfunc='count')


# In[88]:


train['FamilySize'] = train["SibSp"] + train['Parch']
#Sibsp, Parch 합쳤을 때 경향성이 보이기 때문에 합쳐서 다룸


# In[87]:


survived = train[train['Survived'] ==1]['FamilySize']
dead = train[train['Survived']==0]['FamilySize']
df = pd.concat([survived, dead], axis = 1, keys=['Survived', 'Dead'])
df.iplot(kind='histogram', bins=(0,20,1), histfunc='count', bargap=0.1)


# In[89]:


train['Fare'].describe()


# In[90]:


train['Fare'].isnull().sum()


# In[91]:


survived = train[train['Survived']==1]['Fare']
dead = train[train['Survived']==0]['Fare']
df = pd.concat([survived, dead], axis=1, keys=['Survived','Dead'])


# In[92]:


df.iplot(kind='histogram', bins=(0, 600,50))


# In[94]:


train['Title'] = train['Name'].str.extract('([A-Za-z]+)\.', expand=False)


# 범주형 데이터는 히스토그램이 주로 좋음

# In[95]:


train['Title'].value_counts()


# In[96]:


survived = train[train['Survived']==1]['Title']
dead = train[train['Survived']==0]['Title']
df = pd.concat([survived, dead], axis=1, keys=['Survived', 'Dead'])


# In[97]:


df.iplot(kind='histogram')

