import os
from fnmatch import fnmatch
import json


path        = "./image_pairs_rephrasing_data"


file_list   = os.listdir(path)

n           = 0
seeds       = []

for file_name in file_list:
    
    if fnmatch(file_name,"seeds.json"):
        
        continue

    seed    = [file_name, [file_name]]
    seeds.append(seed)
    n      += 1 
    if n % 1000 == 0:
        print('About {} images processed!'.format(n))

seed_file = open(os.path.join(path, 'seeds.json'), 'w')
seed_file.write(json.dumps(seeds))
seed_file.close()