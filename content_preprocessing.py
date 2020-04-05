#%%
import json
#%%
import os
dirpath = os.getcwd()
dirpath

#%%
with open('path_to_file/person.json') as f:
  data = json.load(f)

# %%
