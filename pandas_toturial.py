from numpy.random import randn
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


obj = Series([4, 7, -5, 3])

obj


# In[ ]:


obj.values
obj.index


# In[ ]:


obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
obj2


# In[ ]:


obj2.index


# In[ ]:


obj2['a']


# In[ ]:


obj2['d'] = 6
obj2[['c', 'a', 'd']]


# In[ ]:


obj2[obj2 > 0]


# In[ ]:


obj2 * 2


# In[ ]:


np.exp(obj2)


# In[ ]:


'b' in obj2


# In[ ]:


'e' in obj2


# In[ ]:


sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
obj3


# In[ ]:


states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
obj4

pd.isnull(obj4)


# In[ ]:


pd.notnull(obj4)


# In[ ]:


obj4.isnull()


# In[ ]:


obj3


# In[ ]:


obj4


# In[ ]:


obj3 + obj4


# In[ ]:


obj4.name = 'population'
obj4.index.name = 'state'
obj4


# In[ ]:


obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
obj

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

DataFrame(data, columns=['year', 'state', 'pop'])


# In[ ]:


frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
frame2

frame2.columns


# In[ ]:


frame2['state']


# In[ ]:


frame2.year


# In[ ]:


# frame2.ix['three']
frame2.loc['three']

# In[ ]:


frame2['debt'] = 16.5
frame2


# In[ ]:


frame2['debt'] = np.arange(5.)
frame2

val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
frame2

frame2['eastern'] = frame2.state == 'Ohio'
frame2

del frame2['eastern']
frame2.columns


# In[ ]:


pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}


# In[ ]:


frame3 = DataFrame(pop)
frame3


# In[ ]:


frame3.T


# In[ ]:


DataFrame(pop, index=[2001, 2002, 2003])


# In[ ]:


pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
DataFrame(pdata)


# In[ ]:


frame3.index.name = 'year'; frame3.columns.name = 'state'
frame3


# In[ ]:


frame3.values


# In[ ]:


frame2.values


# ### Index objects

# In[ ]:


obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
index


# In[ ]:


index[1:]


# In[ ]:


index[1] = 'd'


# In[ ]:


index = pd.Index(np.arange(3))
obj2 = Series([1.5, -2.5, 0], index=index)
obj2.index is index


# In[ ]:


frame3


# In[ ]:


'Ohio' in frame3.columns


# In[ ]:


2003 in frame3.index


# ## Essential functionality

# ### Reindexing

# In[ ]:


obj = Series([4.5, 7.2, -5.3, 3.6], index=['d', 'b', 'a', 'c'])
obj


# In[ ]:


obj2 = obj.reindex(['a', 'b', 'c', 'd', 'e'])
obj2


# In[ ]:


obj.reindex(['a', 'b', 'c', 'd', 'e'], fill_value=0)


# In[ ]:


obj3 = Series(['blue', 'purple', 'yellow'], index=[0, 2, 4])
obj3.reindex(range(6), method='ffill')