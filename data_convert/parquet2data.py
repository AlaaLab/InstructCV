import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import pdb
import cv2

# root = '/Users/yulu/Downloads/train-00000-of-00001-476d66d124561578.parquet'
root = './data_convert/InstructCV-data.parquet'
table = pq.read_table(root)
df = table.to_pandas()
input_image  = df["input_image"] #<class 'pandas.core.series.Series'> type(input_image[0]):dict; keys:'bytes', 'path'; type(input_image[0]['path']):str
edit_prompt  = df["edit_prompt"] #<class 'pandas.core.series.Series'> type(edit_prompt[0]):str
edited_image = df["edited_image"] #<class 'pandas.core.series.Series'> type(edited_image[0]):dict; keys:'bytes', 'path'
pdb.set_trace()
file_buff_bytes_input  = input_image[2]['bytes']
edit_prompt_           = edit_prompt[2]
file_buff_bytes_output = edited_image[2]['bytes']

img1 = cv2.imdecode(np.frombuffer(file_buff_bytes_input, np.uint8), cv2.IMREAD_COLOR)
print(edit_prompt_)
img2 = cv2.imdecode(np.frombuffer(file_buff_bytes_output, np.uint8), cv2.IMREAD_COLOR)

## show image
# cv2.imshow("img", img2)
# cv2.waitKey(5000)

cv2.imwrite("123.jpg", img1)

print(df)