#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
import glob


# In[2]:


df = pd.read_csv("W_allAtomsTE0Atom5.csv")


# In[25]:


df.head()


# In[26]:


all_df_modified = df.iloc[:,0:5] 
all_df_modified.tail()


# In[27]:


df_norm = (all_df_modified - all_df_modified.mean()) / (all_df_modified.max() - all_df_modified.min())


# In[28]:


df_norm.to_csv("W_allAtomsTE0_NoTypeAtom5.csv")


# In[29]:


df = pd.read_csv("W_allAtomsTE0_NoTypeAtom5.csv")


# In[30]:


df.head()


# In[43]:


g = sns.pairplot(all_df_modified, palette="husl")


# In[32]:


g = sns.jointplot("X", "Y", df_norm, kind="kde", space=0, color="g")
g.savefig("XYAtom5.png")


# In[33]:


g = sns.jointplot("Y", "Z", df_norm, kind="kde", space=0, color="g")
g.savefig("YZAtom5.png")


# In[44]:


g = sns.jointplot("Y", "Z", df_norm, kind="kde", space=0, color="g")


# In[45]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df_norm.X, df.Y, ax=ax)
sns.rugplot(df_norm.X, color="g", ax=ax)
sns.rugplot(df_norm.Y, vertical=True, ax=ax)


# In[50]:


from scipy import stats
sns.distplot(df_norm.E0, kde=False, fit=stats.gamma);


# In[44]:


EK = pd.read_csv("EK.csv")


# In[45]:


EK.head()


# In[152]:


from scipy import stats
pE_dist = pd.read_csv('pE_dist.csv')
#pE_dist.shape
sns.distplot(pE_dist,kde=True,color="b",bins=30)
g.figure.savefig('PE4.png')


# In[187]:


# from scipy import stats
# Temp_W_dist = pd.read_csv('Temp_W_dist.csv')
#pE_dist.shape
#sns.distplot(pE_dist,kde=True,color="b",bins=15)
# g = sns.lineplot(data=Temp_W_dist,color="b")
# g.set(ylim=(0, 30))
# g.figure.savefig('tempW.png')


import pandas as pd
from sklearn.preprocessing import MinMaxScaler

Temp_W_dist = pd.read_csv('Temp_W_dist.csv')

scaler = MinMaxScaler()



Temp_W_dist[['temp'
]] = scaler.fit_transform(Temp_W_dist[['temp']])

# EK = EK.melt('time', var_name='M', value_name='EK')

Temp_W_dist[['temp']] = scaler.fit_transform(Temp_W_dist[['temp']])

g = sns.distplot(Temp_W_dist.temp, bins=12, kde = False)
# g.set(ylim=(-1, 1))
# Put the legend out of the figure
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('temp4.png')



# In[201]:


PE_dist = pd.read_csv("PE_dist.csv")
ax = sns.lineplot(x='time',y='PE', data=PE_dist)
ax.figure.savefig('PE7.png')


# In[93]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
#sns.set(rc={'figure.figsize':(11.7,8.27)})

EK = pd.read_csv("EK.csv")
EK = EK.melt('time', var_name='M', value_name='EK')

g = sns.lineplot(x="time", y="EK", hue='M', data=EK)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('EK2.png')


# In[79]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
#sns.set(rc={'figure.figsize':(11.7,8.27)})

PE = pd.read_csv("PE.csv")
PE_modified = df.iloc[:,1:] 
PE_modified.tail()

PE_norm = (PE_modified - PE_modified.mean()) / (PE_modified.max() - PE_modified.min())

PE = PE.melt('time', var_name='M', value_name='PE')

g = sns.lineplot(x="time", y="PE", hue='M', data=PE_norm)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('PE2.png')


# In[112]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

EK = pd.read_csv("EK.csv")


EK[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb'
]] = scaler.fit_transform(EK[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb']])

EK = EK.melt('time', var_name='M', value_name='EK')

EK[['EK']] = scaler.fit_transform(EK[['EK']])

g = sns.lineplot(x="time", y="EK", hue='M', data=EK)

# Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('EK3.png')


# In[110]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

PE = pd.read_csv("PE.csv")

PE[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb'
]] = scaler.fit_transform(PE[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb']])


PE = PE.melt('time', var_name='M', value_name='PE')



g = sns.lineplot(x="time", y="PE", hue='M', data=PE)

# # Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('PE3.png')


# In[184]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

PE = pd.read_csv("PE.csv")

PE[['W'
]] = scaler.fit_transform(PE[['W']])


# PE = PE.melt('time', var_name='M', value_name='PE')



g = sns.lineplot(x="time", y="PE", hue='M', data=PE)

# # Put the legend out of the figure
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

g.figure.savefig('PE6.png')


# In[188]:


from scipy import stats
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler()

PE = pd.read_csv("PE.csv")

PE[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb'
]] = scaler.fit_transform(PE[['W','Ta','Sn','Rh','Pt','Pb','Ni','Mo','Mn','V','Ti','Ru','Ir','Cr','Co','Nb']])

g = sns.lineplot(x='time',y=PE.W)

g.figure.savefig('PE4.png')


# In[189]:


PE.head()


# In[2]:


import matplotlib.pylab
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.pylab import plt

get_ipython().run_line_magic('pylab', 'inline')
pylab.rcParams['figure.figsize'] = (10, 6)

import pandas as pd
import numpy as np
from statsmodels.tsa import stattools
from statsmodels.tsa.stattools import acf, pacf


# In[33]:


df = pd.read_csv("train200lstm.csv")


# In[34]:


df.head()


# In[35]:


time = df.iloc[:,0:1]
X_true_train = df.iloc[:,1:2] 

X_pred_train = df.iloc[:,2:3] 

Y_true_train = df.iloc[:,3:4] 
Y_pred_train = df.iloc[:,4:5] 

Z_true_train = df.iloc[:,5:6] 
Z_pred_train = df.iloc[:,6:7] 

E0_true_train = df.iloc[:,7:8] 
E0_pred_train = df.iloc[:,8:9] 


# In[10]:


plt.plot(time, E0_true_train)

plt.savefig('potential_E0.png')


# In[28]:


pacf_result_X = stattools.pacf(X_true_train, nlags = 100)
pacf_result_Y = stattools.pacf(Y_true_train, nlags = 400)
acf_result_Z = stattools.acf(Z_true_train)
pacf_result_E0 = stattools.pacf(E0_true_train, nlags = 200)
pacf_result_E0_pred = stattools.pacf(E0_pred_train, nlags = 200, method="ols")
plt.rcParams.update({'font.size': 16})
plt.plot(pacf_result_E0)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(E0_true_train)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(E0_true_train)),linestyle='--')
plt.ylabel('Partial Autocorrelation')
plt.xlabel('Lag')

plt.savefig('PACF_True.png')


# In[36]:


plt.plot(time, E0_pred_train)

# plt.savefig('potential_E0.png')


# In[37]:


pacf_result_X = stattools.pacf(X_true_train, nlags = 100)
pacf_result_Y = stattools.pacf(Y_true_train, nlags = 400)
acf_result_Z = stattools.acf(Z_true_train)
pacf_result_E0 = stattools.pacf(E0_true_train, nlags = 200)
pacf_result_E0_pred = stattools.pacf(E0_pred_train, nlags = 200, method="ols")
plt.rcParams.update({'font.size': 16})
plt.plot(pacf_result_E0_pred)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(E0_pred_train)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(E0_pred_train)),linestyle='--')
plt.ylabel('Partial Autocorrelation')
plt.xlabel('Lag')

plt.savefig('PACF_LSTM.png')


# In[32]:


X_true_train.interpolate(inplace = True)
decomposition = seasonal_decompose(X_true_train, model = 'additive')
# trend = decomposition.trend
# seasonal = decomposition.seasonal
# residual = decomposition.resid


# In[48]:



from matplotlib.pyplot import figure
plt.rcParams.update({'font.size': 16})

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

ax1 = plt.subplot(4,1,1)
plt.plot(time, X_true_train, '-b' ,label='AIMD')

plt.plot(time, X_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))


plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X (Å)')

ax2 = plt.subplot(4,1,2)
plt.plot(time, Y_true_train, '-b' ,label='true')
plt.plot(time, Y_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y (Å)')

ax3 = plt.subplot(4,1,3)
plt.plot(time,Z_true_train, '-b' ,label='AIMD')
plt.plot(time, Z_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel('Z (Å)')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(time,E0_true_train, '-b' ,label='AIMD')
plt.plot(time, E0_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(0.95, 0.95))



plt.xlabel('time (fs)')
plt.ylabel('energy (eV)')


plt.savefig('LSTM_200_prediction.png')


# In[49]:


time = df.iloc[:,0:1]
ErrorX = df.iloc[:,9:10] 

ErrorY = df.iloc[:,10:11] 

ErrorZ = df.iloc[:,11:12] 

ErrorE0 = df.iloc[:,12:13] 


# In[66]:


ErrorZ.describe()


# In[22]:



figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

ax1 = plt.subplot(4,1,1)
plt.plot(time, ErrorX, '-b')
plt.title('LSTM point-by-point relative percent error')

plt.legend(bbox_to_anchor=(1.1, 1.1))


plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X')

ax2 = plt.subplot(4,1,2)
plt.plot(time, ErrorY, '-b' )
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y')

ax3 = plt.subplot(4,1,3)
plt.plot(time, ErrorZ, '-b' )
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel('Z')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(time, ErrorE0, '-b')
plt.legend(bbox_to_anchor=(0.95, 0.95))



plt.xlabel('time (fs)')
plt.ylabel('energy')


plt.savefig('LSTM_200_Error.png')


# In[30]:


df = pd.read_csv("train200GRU.csv")


# In[31]:


time = df.iloc[:,0:1]
X_true_train = df.iloc[:,1:2] 

X_pred_train = df.iloc[:,2:3] 

Y_true_train = df.iloc[:,3:4] 
Y_pred_train = df.iloc[:,4:5] 

Z_true_train = df.iloc[:,5:6] 
Z_pred_train = df.iloc[:,6:7] 

E0_true_train = df.iloc[:,7:8] 
E0_pred_train = df.iloc[:,8:9] 


# In[32]:


plt.plot(time, E0_pred_train)

# plt.savefig('potential_E0.png')


# In[26]:


pacf_result_X = stattools.pacf(X_true_train, nlags = 200)
pacf_result_Y = stattools.pacf(Y_true_train, nlags = 400)
acf_result_Z = stattools.acf(Z_true_train)
pacf_result_E0 = stattools.pacf(E0_true_train, nlags = 200)
pacf_result_E0_pred = stattools.pacf(E0_pred_train, nlags = 200, method="ols")
plt.rcParams.update({'font.size': 16})
plt.plot(pacf_result_E0_pred)
plt.axhline(y=0,linestyle='--')
plt.axhline(y=-1.96/np.sqrt(len(E0_pred_train)),linestyle='--')
plt.axhline(y=1.96/np.sqrt(len(E0_pred_train)),linestyle='--')
plt.ylabel('Partial Autocorrelation')
plt.xlabel('Lag')

plt.savefig('PACF_GRU.png')


# In[ ]:





# In[25]:



from matplotlib.pyplot import figure
plt.rcParams.update({'font.size': 16})

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

ax1 = plt.subplot(4,1,1)
plt.plot(time, X_true_train, '-b' ,label='AIMD')

plt.plot(time, X_pred_train, '-r' ,label='GRU')
plt.legend(bbox_to_anchor=(1.1, 1.1))


plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X (Å)')

ax2 = plt.subplot(4,1,2)
plt.plot(time, Y_true_train, '-b' ,label='AIMD')
plt.plot(time, Y_pred_train, '-r' ,label='GRU')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y (Å)')

ax3 = plt.subplot(4,1,3)
plt.plot(time,Z_true_train, '-b' ,label='AIMD')
plt.plot(time, Z_pred_train, '-r' ,label='GRU')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel('Z (Å)')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(time,E0_true_train, '-b' ,label='AIMD')
plt.plot(time, E0_pred_train, '-r' ,label='GRU')
plt.legend(bbox_to_anchor=(0.95, 0.95))



plt.xlabel('time (fs)')
plt.ylabel('energy (eV)')


plt.savefig('GRU_200_prediction.png')


# In[40]:


time = df.iloc[:,0:1]
ErrorX = df.iloc[:,9:10] 

ErrorY = df.iloc[:,10:11] 

ErrorZ = df.iloc[:,11:12] 

ErrorE0 = df.iloc[:,12:13] 


# In[44]:


ErrorE0.describe()


# In[27]:



figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

ax1 = plt.subplot(4,1,1)
plt.plot(time, ErrorX, '-b')
plt.title('GRU point-by-point relative percent error')

plt.legend(bbox_to_anchor=(1.1, 1.1))


plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X')

ax2 = plt.subplot(4,1,2)
plt.plot(time, ErrorY, '-b' )
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y')

ax3 = plt.subplot(4,1,3)
plt.plot(time, ErrorZ, '-b' )
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel('Z')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(time, ErrorE0, '-b')
plt.legend(bbox_to_anchor=(0.95, 0.95))



plt.xlabel('time (fs)')
plt.ylabel('energy')


plt.savefig('GRU_200_Error.png')


# In[31]:


df = pd.read_csv("lstm_train_atom25.csv")


# In[34]:


df.head()


# In[32]:


time = df.iloc[:,0:1]
X_true_train = df.iloc[:,1:2] 

X_pred_train = df.iloc[:,2:3] 

Y_true_train = df.iloc[:,3:4] 
Y_pred_train = df.iloc[:,4:5] 

Z_true_train = df.iloc[:,5:6] 
Z_pred_train = df.iloc[:,6:7] 

E0_true_train = df.iloc[:,7:8] 
E0_pred_train = df.iloc[:,8:9] 


# In[33]:



from matplotlib.pyplot import figure
plt.rcParams.update({'font.size': 16})

figure(num=None, figsize=(12, 10), dpi=80, facecolor='w', edgecolor='k')

ax1 = plt.subplot(4,1,1)
plt.plot(time, X_true_train, '-b' ,label='AIMD')

plt.plot(time, X_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))


plt.setp(ax1.get_xticklabels(), visible=False)
plt.ylabel('X (Å)')

ax2 = plt.subplot(4,1,2)
plt.plot(time, Y_true_train, '-b' ,label='AIMD')
plt.plot(time, Y_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax2.get_xticklabels(), visible=False)
plt.ylabel('Y (Å)')

ax3 = plt.subplot(4,1,3)
plt.plot(time,Z_true_train, '-b' ,label='AIMD')
plt.plot(time, Z_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(1.1, 1.1))

plt.setp(ax3.get_xticklabels(), visible=False)
plt.ylabel('Z (Å)')

ax4 = plt.subplot(4,1,4, sharex=ax1)
plt.plot(time,E0_true_train, '-b' ,label='AIMD')
plt.plot(time, E0_pred_train, '-r' ,label='LSTM')
plt.legend(bbox_to_anchor=(0.95, 0.95))



plt.xlabel('time (fs)')
plt.ylabel('energy (eV)')


plt.savefig('LSTM_prediction_O.png')




