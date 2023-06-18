import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pdb
import json
import os
import pandas as pd
from PIL import Image as PI
import cv2
from fnmatch import fnmatch
from datasets import Dataset, Features,Image, Value,Sequence


root    = "./image_pairs"
file_n  = os.listdir(root)

input_image, edited_image, edit_prompt, path_ori= [],[],[],[]

for file in file_n:
    path    = os.path.join(root, file)
    data_n  = os.listdir(path)
    
    for data in data_n:
        data_p = os.path.join(path, data)
        path_ori = data
        
        if fnmatch(data, "*_0.jpg"):
            image = PI.open(data_p)
            if hasattr(image, "filename"):
                image.filename = ""
            image_decode = Image(decode=True, id=None)
            _image = image_decode.encode_example(value=image)
            input_image.append(_image)
            
        if fnmatch(data, "*_1.jpg"):
            image = PI.open(data_p)
            if hasattr(image, "filename"):
                image.filename = ""
            image_decode = Image(decode=True, id=None)
            _image = image_decode.encode_example(value=image)
            
            edited_image.append(_image)
        
        if fnmatch(data, "bbox.json"):
            continue
        
        if fnmatch(data, "*json*"):
            with open(data_p, encoding = 'utf-8') as fp:
                data_load = json.load(fp)
                edit_prompt_  = data_load['edit']
                edit_prompt.append(edit_prompt_)

# features = Features({
#         'edit_prompt': Value(dtype='string', id=None),
#         'input_image': Image(decode=True, id=None),
#         'edited_image': Image(decode=True, id=None),
#     })

df = pd.DataFrame({
    "input_image": input_image,
    "edit_prompt": edit_prompt,
    "edited_image": edited_image,
})

# tb = pa.Table.from_pandas(df)
df.to_parquet("./data_convert/InstructCV-data.parquet")
