import os
import pdb



def split(dataset):
    '''
    split test data into various parts
    '''
    if dataset == "ADE": # split into 10 parts
        
        img_list        = os.listdir(os.path.join(ADE_path, 'images/validation'))
        data = [i+'\n' for i in img_list]
        
        for i in range(0,10):
            data_part   = data[i*200:(i+1)*200]

            with open (os.path.join(ADE_path,'test_part{}.txt'.format(i)), 'w') as b:
                b.writelines(data_part)
                b.close()
    
    if dataset == "coco": # split into 10 parts
        
        img_list        = os.listdir(os.path.join(coco_path, 'val2017'))
        data = [i+'\n' for i in img_list]
        
        for i in range(0,10):
            data_part   = data[i*500:(i+1)*500]
            
            with open (os.path.join(coco_path,'test_part{}.txt'.format(i)), 'w') as b:
                b.writelines(data_part)
                b.close()

    return


if __name__ == "__main__":
    
    ADE_path            = './data/ADEChallengeData2016/'
    coco_path           = './data/coco'
    
    
    split(dataset="coco")
    

