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