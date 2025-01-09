import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
from math import floor, log
from torch.nn.functional import binary_cross_entropy
from config import num_priors


eps = 1e-7
#lambda_ = 0.85
#alpha = 10




class yolov3_loss(nn.Module):
	def __init__(self, anchor_boxes):
		super(yolov3_loss, self).__init__()
		self.lambda_ = lambda_
		self.alpha = alpha
		self.anchor_boxes = anchor_boxes

	def forward(self, gtboxes, gt_classes, out, scaled_anchor_images, scale, anchor_box_inds, iou_box_coords num_classes):
		loss_pred = 0
		loss_class = 0
		vals_per_prior = out.shape[1]/num_priors
		num_boxes = 0
		for j in out.shape[0]:
			#for i, gtbox in enumerate(gtboxes[j]):
			for i in range(gtboxes.shape[1]):
				box = gtboxes[j][i]/scale
				anchor_box = self.anchor_boxes[anchor_box_inds[j][i]]/scale
				#gt_offset_x = box[0] - floor(box[0])
				#gt_offset_y = box[1] - floor(box[1])
				gt_offset = box[0:2] - torch.floor(box[0:2])
				gt_pred = -torch.log((1 - gt_offset_x)/gt_offset)
				gt_hw_pred = torch.log(box[2:4]/anchor_box)
				loss_pred += torch.sum(torch.square(gt_pred - out[j,vals_per_prior*anchor_box_inds[j][i]:values_per_prior*anchor_box_inds[i]+2,iou_box_coords[0],iou_box_coords[1]])) 
				+ torch.sum(torch.square(gt_hw_pred - out[j,vals_per_prior*anchor_box_inds[j][i] + 2: vals_per_prior*anchor_box_inds[j][i] + 4,iou_box_coords[0],iou_box_coords[1]]))
																									
				loss_class += binary_cross_entropy(out[j,vals_per_prior*anchor_box_inds[j][i]+5:vals_per_prior*(anchor_box_inds[j][i]+1), iou_box_coords[0],iou_box_coords[1]], gt_classes[j][i])
				num_boxes += 1
			for n in range(num_priors)
				loss_obj += binary_cross_entropy(out[j,n*vals_per_prior + 4,scaled_anchor_images[j] != -1], scaled_anchor_images[j][scaled_anchor_images[j] != -1], reduction='sum')
			obj_sum += num_priors*torch.sum(scaled_anchor_images[j][scaled_anchor_images[j] != -1])
		
		loss_obj = loss_obj/obj_sum
		loss_class = loss_class/nun_boxes
		loss_pred = loss_pred/num_boxes
		loss = loss_pred + loss_class + loss_obj	
		return loss

