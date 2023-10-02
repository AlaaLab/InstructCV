import os
import pdb


def split(dataset):
    '''
    split test data into various parts
    '''
    if dataset == "ADE": # split into 10 parts
        
        img_list            = os.listdir(os.path.join(ADE_path, 'images/validation'))
        data = [i+'\n' for i in img_list]
        
        for i in range(0,10):
            data_part       = data[i*200:(i+1)*200]

            with open (os.path.join(ADE_path,'test_part{}.txt'.format(i)), 'w') as b:
                b.writelines(data_part)
                b.close()
    
    if dataset == "coco": # split into 10 parts
        
        img_list            = os.listdir(os.path.join(coco_path, 'val2017'))
        data                = [i+'\n' for i in img_list]
        
        for i in range(0,10):
            data_part       = data[i*500:(i+1)*500]
            
            with open (os.path.join(coco_path,'test_part{}.txt'.format(i)), 'w') as b:
                b.writelines(data_part)
                b.close()
    
    if dataset == "fs_1000": #split into 10 parts
        
        img_list            = os.listdir(fs1000_path)
        num                 = len(img_list)
        
        if num != 1000:
            assert "The number of files must be 1000"
            
        data_img, data_gt   = [],[]
        
        for n in range(1,11):
            
            data1           = [i+'/{}.jpg\n'.format(n) for i in img_list]
            data2           = [i+'/{}.png\n'.format(n) for i in img_list]
            data_img        = data1 + data_img# i:ab_wheel/1.jpg
            data_gt         = data2 + data_gt # i:ab_wheel/1.png

        
        print("len:", len(data_img))
        print("len:", len(data_gt))
        
        for i in range(0,10):
            data_part_img   = data_img[i*1000:(i+1)*1000]
            data_part_gt    = data_gt[i*1000:(i+1)*1000]
            
            with open (os.path.join(fs1000_path,'test_part{}.txt'.format(i)), 'w') as b:
                b.writelines(data_part_img + data_part_gt)
                b.close()
    
    if dataset == "voc": #split into 9 parts
        
        voc_list = []
        for line in open(os.path.join(voc_path, 'ImageSets/Segmentation/val.txt')):
            line = line.strip()
            voc_list.append(line)
        
        print("len(voc_list)",len(voc_list))
        data_img                = ["JPEGImages/" + m + ".jpg"  for m in voc_list]
        data_gt                 = ["SegmentationClass/" + m + ".png" + '\n' for m in voc_list]
        data_final              = []

        for p in range(len(data_img)):
            tmp = data_img[p] + " " + data_gt[p]
            data_final.append(tmp)
            
        for i in range(0,9):
            tmp       = data_final[i*161:(i+1)*161]
            with open (os.path.join(voc_path,'val_part{}.txt'.format(i)), 'w') as b:
                b.writelines(tmp)
                b.close()
            
        
        


    return


if __name__ == "__main__":
    
    ADE_path            = './data/ADEChallengeData2016/'
    coco_path           = './data/coco'
    fs1000_path         = './data/fewshot_data/fewshot_data'
    voc_path            = './data/VOCdevkit/VOC2012/'
    
    
    split(dataset="voc")
    

