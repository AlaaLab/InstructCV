
NUM_PATCHES        = 16
IMG_SIZE           = 224
device             = "cuda"

redweb_images_dir  = "data/ReDWeb_V1/Imgs/"
redweb_label_dir   = "data/ReDWeb_V1/RDs/"

oxford_data_path   = "data/oxford-pets"
oxford_images_dir  = "data/oxford-pets/images/"
oxford_label_dir   = "data/oxford-pets/annotations/trimaps/"
voc_data_root      = "data/voc/VOCdevkit/VOC2012/"


redwebv1_params    = dict({"IMG_SIZE": 112,
                           "task": "depth_estimation",
                           "img_ext": ".jpg",
                           "label_ext": ".png",
                           "img_dir": redweb_images_dir,
                           "label_dir": redweb_label_dir,
                           "label_rgb": True})

oxford_params    = dict({"IMG_SIZE": 112,
                           "task": "semantic_segmentation", #"fine_grained_classification", #"classification", #
                           "img_ext": ".jpg",
                           "label_ext": ".png",
                           "img_dir": oxford_images_dir,
                           "label_dir": oxford_label_dir,
                           "data_path": oxford_data_path,
                           "label_rgb": False})

voc_params       = dict({"IMG_SIZE": 112,
                           "task": "semantic_segmentation", #"fine_grained_classification", #"classification", #
                           "img_ext": ".jpg",
                           "label_ext": ".png",
                           "data_path": voc_data_root})

dataset_params     = dict({"RedWeb_V1": redwebv1_params, 
                           "oxford-pets": oxford_params,
                           "voc":voc_params})
