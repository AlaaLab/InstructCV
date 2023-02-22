# define the mae utils

import sys
import os
import requests

import torch
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import models_mae
import config
import yaml
from comet_ml import Experiment

from promptification import promptify
#from util.data_processors import *
import util.lr_decay as lrd

# remove these

# imagenet_mean = np.array([0.485, 0.456, 0.406])
# imagenet_std  = np.array([0.229, 0.224, 0.225])

imagenet_mean = torch.Tensor([0.485, 0.456, 0.406])
imagenet_std  = torch.Tensor([0.229, 0.224, 0.225])


def get_cuda_devices():
    
    if torch.cuda.is_available():
    
        # print("GPU(s) available: ", torch.cuda.get_device_name())
    
        # remove any device which doesn't exists
        deviceIds = [int(d) for d in range(torch.cuda.device_count())] 
    
        # set args.deviceIds[0] (the master node) as the current device
        torch.cuda.set_device(deviceIds[0])
    
        device    = torch.device("cuda")

    else:
    
        device    = torch.device('cpu')
        deviceIds = None

    return device
 
device        = get_cuda_devices()

def stitch_visual_collage(train_batch, visual_cue, batch_size, multi_task, out_shape="nchw"):
    '''
    Arg:
        train_batch - (dataset_loader, language_dataset_loader)
        visual_cue - VisualCue(args.image_size)
    
    Return:
        train: concat databatch & visual_cue (see shape)
    '''
    
    if type(train_batch) is list:
        
        concat_train_batch      = torch.cat([train_batch[0], train_batch[1]], dim=2)
        
        print('concat_train_batch:{}'.format(concat_train_batch.shape))
        
        train_batch             = [concat_train_batch]
    
    if multi_task == False:
        #print('visual_cue.shape[0]=1, train_batch:{}'.format(train_batch[0].shape))
    
        train                   = torch.cat([torch.einsum("nchw->nhwc", train_batch[0]), visual_cue[0, :, :, :].unsqueeze(0).repeat(batch_size, 1, 1, 1)], dim=2) 
        #print('after cat, train:{}'.format(train.shape))
    
    else:

        data_batch, lan_batch   = train_batch
  
        visual_cue              = visual_cue(lan_batch)
        device                  = get_cuda_devices()
        data_batch[0]           = data_batch[0].to(device)
        train                   = torch.cat([torch.einsum("nchw->nhwc", data_batch[0]), visual_cue], dim=2) 
    
    
    if out_shape=="nchw":
        
        train = torch.einsum("nhwc->nchw", (train - torch.tensor(imagenet_mean).to(device)) / torch.tensor(imagenet_std).to(device)).float()
        #print('out_shape is "nchw", train:{}'.format(train.shape))
    
    elif out_shape=="nhwc":
        
        train = ((train - torch.tensor(imagenet_mean)) / torch.tensor(imagenet_std)).float()
    
    return train


def create_dir(PATH):

    if not os.path.exists(PATH):
    
        os.makedirs(PATH)


def get_out_dir(dataset_name, task, model_type, finetuned):
    
    prompt_subdir = "finetuned" if finetuned else "frozen"
    
    model_path    = "./models/" + dataset_name + "/" + task 
    prompt_path   = "./prompts/" + dataset_name + "/" + prompt_subdir + "/" + task 
    
    create_dir(model_path)
    create_dir(prompt_path)
    
    return model_path, prompt_path
  
    
def get_comet_experiment(config_path="comet_config.yml"):
    
    with open(config_path, "r") as file:
        
        comet_data = yaml.load(file, Loader=yaml.FullLoader)
    
    experiment = Experiment(api_key=comet_data["api_key"],
                            project_name=comet_data["project_name"], 
                            workspace=comet_data["workspace"],)
    
    return experiment


def get_optimizer(model, visual_cue, learning_rate, weight_decay, layer_decay, batch_size, finetune, device, **kwargs):

    if finetune:
        
        #param_groups = lrd.param_groups_lrd(model.module.module, weight_decay=weight_decay, layer_decay=layer_decay) # no_weight_decay_list=model_without_ddp.no_weight_decay()
        #optimizer    = torch.optim.AdamW([visual_cue] + list(param_groups), lr=learning_rate)
        
        #optimizer   = torch.optim.Adam([visual_cue] + list(model.parameters()), lr=learning_rate)
        optimizer   = torch.optim.Adam(list(visual_cue.parameters()) + list(model.parameters()), lr=learning_rate)
        
    else:    
        
        optimizer    = torch.optim.AdamW(list(visual_cue.parameters()), lr=learning_rate)

    return optimizer


def initialize_visual_cue(num_tasks, image_size):
    
    return torch.nn.init.xavier_uniform(torch.nn.Parameter(torch.zeros(num_tasks, image_size, int(image_size/2), 3)))


def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clamp((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def prepare_model(chkpt_dir=None, arch='mae_vit_large_patch16'):
    # build model
    model = getattr(models_mae, arch)()
    
    if torch.cuda.device_count() > 1:
        model.cuda()
    
    # load model
    
    if chkpt_dir is not None:
        
        checkpoint = torch.load(chkpt_dir, map_location=device) #'cpu')
        msg        = model.load_state_dict(checkpoint['model'], strict=False)
        print(msg)
    
    return model


def run_one_image(img, model, mask_ratio=0.75, display=True, rgb_output=True, save_name='1.png'):
    x = torch.tensor(img).to(device)

    # make it a batch-like
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    # run MAE
    #loss, y, mask = model(x.float(), mask_ratio=mask_ratio)
    loss, y, mask = model(**dict(imgs=x.float(), mask_ratio=mask_ratio))
 
    
    if torch.cuda.device_count() > 1:
        
        y         = model.module.module.unpatchify(y)
    
    else:
        
        y         = model.unpatchify(y)
  
        y         = torch.einsum('nchw->nhwc', y).detach().cpu()
   
 

    # visualize the mask
    mask = mask.detach()

    
    if torch.cuda.device_count() > 1:
        
        mask = mask.unsqueeze(-1).repeat(1, 1, model.module.module.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.module.module.unpatchify(mask)  # 1 is removing, 0 is keeping
    
    else:
        
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)
    
    
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
    x = torch.einsum('nchw->nhwc', x)

    # masked image

    im_masked = x * (1 - mask.to(device))
    
    # !!!!modify
    y = torch.einsum('nchw->nhwc', y)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask.to(device)) + y.to(device) * mask.to(device)

    # make the plt figure larger
    
    if display:
        
        plt.rcParams['figure.figsize'] = [24, 24]
        plt.rcParams['savefig.dpi'] = 200
        plt.rcParams['figure.dpi'] = 500

        plt.subplot(1, 3, 1)
        show_image(x[0].cpu(), "original")

        plt.subplot(1, 3, 2)
        show_image(im_masked[0].cpu(), "masked")

        #plt.subplot(1, 4, 3)
        #show_image(y[0].cpu(), "reconstruction")

        plt.subplot(1, 3, 3)
        show_image(im_paste[0].cpu(), "reconstruction + visible")

        plt.show()
        plt.savefig(save_name)
        
    return im_paste[0].cpu(), im_masked[0].cpu(), mask.cpu(), loss.cpu()



def load_model(ViT_mode="base", prompt=True):
    '''
    When prompt = True:
        replace random_masking with prompted_masking
    '''
    
    patch_size = 16
    
    if ViT_mode=="huge":
        
        if os.path.exists('mae_visualize_vit_huge.pth') == False:
            os.system("wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_huge.pth")  

        chkpt_dir  = 'mae_visualize_vit_huge.pth'
        model_mae  = prepare_model('mae_visualize_vit_huge.pth', 'mae_vit_huge_patch14') # load model_mae
    
        print('Model loaded.')
        
        patch_size = 14
        
    elif ViT_mode=="large":
        
        if os.path.exists('mae_visualize_vit_large.pth') == False:
            os.system("wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth")

        chkpt_dir = 'mae_visualize_vit_large.pth'
        model_mae = prepare_model('mae_visualize_vit_large.pth', 'mae_vit_large_patch16')
    
        print('Model loaded.')
        
    elif ViT_mode=="large_ganloss":   
        
        os.system("wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large_ganloss.pth")

        chkpt_dir = 'mae_visualize_vit_large_ganloss.pth'
        model_mae = prepare_model('mae_visualize_vit_large_ganloss.pth', 'mae_vit_large_patch16')
        
        print('Model loaded.')
        
    elif ViT_mode=="base":   
        
        os.system("wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_base.pth")

        chkpt_dir = 'mae_visualize_vit_base.pth'
        model_mae = prepare_model('mae_visualize_vit_base.pth', 'mae_vit_base_patch16')
    
        print('Model loaded.')
    
    elif ViT_mode is None:
        
        os.system("wget -nc https://dl.fbaipublicfiles.com/mae/visualize/mae_visualize_vit_large.pth")

        chkpt_dir = None
        model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    
        print('Model loaded.')
    
    
    if prompt:
        
        model_mae = promptify(model_mae)
        print('Model promptified.')
    
    config.NUM_PATCHES  = int(config.IMG_SIZE/patch_size)
    
    if torch.cuda.device_count() > 1:
        
        deviceIds = [int(d) for d in range(torch.cuda.device_count())] 
        model_mae = torch.nn.DataParallel(model_mae, device_ids=deviceIds)
    
    return model_mae, patch_size




def mae_forward_pass(model, 
                     valid_loader, 
                     visual_cue, 
                     IMG_SIZE, 
                     PATCH_SIZE, 
                     NUM_PATCHES, 
                     segment=False):
    
    segs_true    = []
    segs_rec     = []
    val_list     = []
        
    
    for data_batch in valid_loader:
        
        if visual_cue.shape[0]==1:
        
            #test_img        = torch.cat([torch.einsum("nchw->nhwc", data_batch[0]).squeeze(0), visual_cue[0, :, :, :].unsqueeze(0).detach().squeeze(0)], dim=1).cpu().detach()
            test_img        = stitch_visual_collage(data_batch, visual_cue, valid_loader.batch_size, out_shape="nhwc").cpu().detach()
        
        else:
            
            val_batch, val_label = data_batch
            test_img             = torch.cat([torch.einsum("nchw->nhwc", val_batch[0]).squeeze(0), visual_cue[val_label[0].item(), :, :, :].unsqueeze(0).detach().squeeze(0)], dim=1).cpu().detach() 
        
        test_img            = test_img - imagenet_mean
        test_img            = test_img / imagenet_std
    
        reconstructed_frame = run_one_image(test_img, model, display=False)
        rec_frame           = reconstructed_frame[0]
        loss                = reconstructed_frame[3] 
            
        val_list.append(np.float(loss.cpu().detach().numpy()))  
        
        if segment:

            seg_rec   = rec_frame[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
            seg_rec   = seg_rec * imagenet_std + imagenet_mean
        
            seg_true  = test_img[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
            seg_true  = ((seg_true * imagenet_std + imagenet_mean)==1) * 1
    
            segs_true.append(seg_true)
            segs_rec.append(seg_rec)
    
    val_loss      = np.mean(np.array(val_list))
    
    if segment:
        
        output = (val_loss, segs_true, segs_rec)
    
    else: 
        
        output = val_loss
        
    return output    


# No gradients


def forward_pass(model, 
                 data, 
                 visual_cue, 
                 IMG_SIZE, 
                 PATCH_SIZE, 
                 NUM_PATCHES):
    
    segs_true = []
    segs_rec  = []
    val_list  = []
        
    for indx in list(np.arange(0, data.shape[0])):
            
        test_img            = torch.cat([data[indx, :, :, :], visual_cue[0, :, :, :].unsqueeze(0).detach().squeeze(0)], dim=1).cpu().detach()
        test_img            = test_img - imagenet_mean
        test_img            = test_img / imagenet_std
    
        reconstructed_frame = run_one_image(test_img, model, display=False)
        rec_frame           = reconstructed_frame[0]
        loss                = reconstructed_frame[3] 
            
        val_list.append(np.float(loss.cpu().detach().numpy()))  

        seg_rec   = rec_frame[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
        seg_rec   = seg_rec * imagenet_std + imagenet_mean
        
        seg_true  = test_img[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
        seg_true  = ((seg_true * imagenet_std + imagenet_mean)==1) * 1
    
        segs_true.append(seg_true)
        segs_rec.append(seg_rec)
    
    val_loss      = np.mean(np.array(val_list))
    output        = (val_loss, segs_rec)
        
    return output 



def forward_collage(model, 
                    data, 
                    visual_cue, 
                    IMG_SIZE, 
                    PATCH_SIZE, 
                    NUM_PATCHES):
    
    rec_frames = []

        
    for indx in list(np.arange(0, data.shape[0])):
            
        test_img            = torch.cat([data[indx, :, :, :], visual_cue[0, :, :, :].unsqueeze(0).detach().squeeze(0)], dim=1).cpu().detach()
        test_img            = test_img - imagenet_mean
        test_img            = test_img / imagenet_std
    
        reconstructed_frame = run_one_image(test_img, model, display=False)
        rec_frame           = reconstructed_frame[0]

        seg_rec     = rec_frame[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
        #seg_rec     = seg_rec * imagenet_std + imagenet_mean
        
        seg_true    = test_img[int(NUM_PATCHES / 2) * PATCH_SIZE:, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
        #seg_true    = seg_true * imagenet_std + imagenet_mean
        
        input_image = test_img[:int(NUM_PATCHES / 2) * PATCH_SIZE, :int(NUM_PATCHES / 2) * PATCH_SIZE, :]
        #input_image = input_image * imagenet_std + imagenet_mean
        
        
        rec_frame_  = torch.cat([seg_true, input_image, seg_rec], axis=0)
        
        rec_frames.append(rec_frame_)
    
        
    return rec_frames


def validate(model, val_data, visual_cue, IMG_SIZE, patch_size, device="cuda"):
    
    val_loss_batch    = []
    
    for i, val_batch in enumerate(val_data):
           
        input_batch   = prepare_visual_collage_batch(val_batch[0], 
                                                     val_batch[1], 
                                                     IMG_SIZE).to(device) 
        
        _val_loss     = mae_forward_pass(model,
                                         input_batch.to(device), 
                                         visual_cue=visual_cue.to(device), 
                                         IMG_SIZE=IMG_SIZE, 
                                         PATCH_SIZE=patch_size, 
                                         NUM_PATCHES=config.NUM_PATCHES, 
                                         segment=False)
        
        val_loss_batch.append(_val_loss)
        
    return np.mean(np.array(val_loss_batch))   

