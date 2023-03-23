import os
import pdb



def split(dataset):
    '''
    split test data into various parts
    '''
    if dataset == "ADE": # split into 5 parts
        
        img_list        = os.listdir(os.path.join(ADE_path, "images/validation"))
        data = [i+'\n' for i in img_list]
        
        for i in range(0,5):
            data_part   = data[i*400:(i+1)*400]

            with open (os.path.join(ADE_path,"test_part{}.txt".format(i)), 'w') as b:
                b.writelines(data_part)
                b.close()

    return


if __name__ == "__main__":
    
    ADE_path = "./data/ADEChallengeData2016/"
    split(dataset="ADE")
    

