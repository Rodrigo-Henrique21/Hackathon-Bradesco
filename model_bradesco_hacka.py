#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar pacotes necessários
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
pd.options.display.max_columns = None
#pd.options.display.max_rows = None
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import pandas_profiling
import warnings
warnings.filterwarnings(action="ignore")


# In[2]:


# Carregar o conjunto de dados históricos dos clientes
#df = pd.read_csv('dataset_all.csv').sample(100000, random_state=44)
df = pd.read_csv('dataset_all.csv')


# In[3]:


df.head()


# In[4]:


df.dtypes


# In[5]:


df.shape


# In[6]:


#missing data
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# In[7]:


df.drop(['ID da conta','tipo pessoa','é flexível no limite','data de vencimento - credito','é Prestação - credito','data de vencimento - financiamento','data de vencimento - emprestimo' ,'ID de consentimento','transações de valor - credito' ,'quantia - conta','Valor do pagamento - credito','data de pagamento - credito'], inplace=True, axis=1)


# In[8]:


df


# In[9]:


#verifica se é inadimplente
df['divida_total'] = df[['valor total da fatura - credito', 'saldo pendente do contrato - financiamento', 'saldo pendente do contrato - emprestimo']].sum(axis=1)
df['divida_atrasada'] = df[['Prestações devidas - emprestimo', 'Prestações devidas - financiamento']].sum(axis=1)
df['utilizacao_limite_cartao'] = df['Valor limite - credito'] / df['cheque especial Limite contratado']

df['utilizacao_limite_cartao'].replace([np.inf, -np.inf], -1, inplace=True)
df['utilizacao_limite_cartao'].replace([np.NaN], 0, inplace=True)

df['limite_negativo'] = 0

def verifica_limite(x):
    if x >= 0:
        return  0
    else:
        return  1

def verifica_inadimp(x,y):   
    if (x > 0) and (y == 1):
        return 1
    else:
       return 0


# In[10]:


#analise_2 = df[['Valor limite - credito','cheque especial Limite contratado']]


# In[11]:


#analise_3 = df[df['Valor limite - credito'] <= 0]


# In[12]:


df['limite_negativo'] = df['utilizacao_limite_cartao'].apply(verifica_limite)


# In[13]:


df['eh_inadimplente'] = np.vectorize(verifica_inadimp)(df['divida_atrasada'], df['limite_negativo'])


# In[14]:


df['limite_negativo'].value_counts()


# In[15]:


df['eh_inadimplente'].value_counts()


# In[16]:


'''
from imblearn.over_sampling import SMOTE
import numpy as np

SEED=42

smote = SMOTE(random_state=42)
'''


# In[17]:


count_majority_class, count_minority_class = df.eh_inadimplente.value_counts()


# In[18]:


count_majority_class


# In[19]:


df_majority_class = df[df['eh_inadimplente'] == 0]
df_minority_class = df[df['eh_inadimplente'] == 1]


# In[20]:


df_class_undersample = df_majority_class.sample(count_minority_class)


# In[21]:


df_balanced = pd.concat([df_class_undersample, df_minority_class], axis=0)


# In[22]:


print('Number of data samples after under-sampling:')
print(df_balanced.eh_inadimplente.value_counts())


# In[23]:


#smote_technique = SMOTE(sampling_strategy='minority')


# In[24]:


X = df_balanced.drop("eh_inadimplente", axis=1)
Y = df_balanced["eh_inadimplente"]


# In[25]:


#x_resampled, y_resampled = smote_technique.fit_resample(X, Y)


# In[26]:


#df_balanced = pd.concat([y_resampled, x_resampled], axis=1)
#df_balanced


# In[27]:


df_balanced["eh_inadimplente"].value_counts()


# In[28]:


#x = df_balanced.iloc[:, 1:].values
#y = df_balanced.iloc[:, 0].values


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42)


# In[30]:



from sklearn.preprocessing import StandardScaler

# Padronizar as variáveis numéricas
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[31]:


# Treinar o modelo de regressão logística
modelo = LogisticRegression()
#modelo = RandomForestClassifier(max_depth= 5, random_state = 42) 
modelo.fit(X_train, y_train)


# In[32]:


y_test


# In[33]:


acuracia = modelo.score(X_test, y_test)
print('Acurácia:', acuracia)


# In[34]:


# Gerar o score para cada cliente
#score = modelo.predict_proba(X_test)[:, 1].round()
score = modelo.predict_proba(X_test)[:, 1]


# In[35]:


#from sklearn import metrics
'''
print("Acurácia:",metrics.accuracy_score(y_test, score))
print("Precisão:",metrics.precision_score(y_test, score))
print("Recall:",metrics.recall_score(y_test, score)) 
print("F1:",metrics.f1_score(y_test, score))
'''


# In[36]:


score


# In[37]:


# Fornecer produtos financeiros com base no score gerado e nas variáveis adicionais
produto_oferecido = np.where((score >= 0.8),  'Empréstimo Pessoal',                                     np.where((score >= 0.5) , 'Cartão de Crédito',                                     np.where((score >= 0.2) , 'Conta Corrente', 'Sem Oferta')))


# In[38]:


produto_oferecido.shape


# In[39]:


produto_oferecido


# In[40]:


df_teste = pd.DataFrame(produto_oferecido, columns = ['produto_oferecido'])


# In[41]:


df_teste


# In[42]:


#produtos ofertados
df_teste['produto_oferecido'].value_counts()


# In[ ]:




