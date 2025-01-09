import numpy as np
import torch

# coco resolution 640 x 480

# kitti resolutio 1240 x 375


# coco anchor boxes
#(10×13),(16×30),(33×23),(30×61),(62×45),(59×
#119),(116 × 90),(156 × 198),(373 × 326).

anchor_boxes = torch.tensor([[[40, 15], [31, 25], [62, 19]], [[62, 36], [124, 42], [122, 90]], [[245, 75], [310, 148], [640, 250]]])

scales = [32, 16, 8]

num_priors = 3

assert num_priors == anchor_boxes.shape[1], 'num_priors is not equal to nuo of priors in anchor_boxes in each scale'

kitti_obj_det_labels = ['Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc', 'DontCare']

dirlist = ['C:/Users/Kush/Downloads/data_object_label_2/training/label_2']

