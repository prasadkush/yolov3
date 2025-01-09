import os
from math import ceil, floor
import numpy as np
import config
import torch


def compute_intersection_union(box1, box2):
# box [x_left, y_top, x_right, y_bottom]
  width = np.fmin(box1[2], box2[:,2]) - np.fmax(box1[0], box2[:,0]) 
  height = np.fmin(box1[3], box2[:,3]) - np.fmax(box1[1], box2[:,1])
  intersection = (width >= 0) * width * height * (height >= 0)
  union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + (box2[:,2] - box2[:,0]) * (box2[:,3] - box2[:,1]) - intersection
  #print('intersection: ', intersection)
  #print('union: ', union)
  #print('box1: ', box1)
  #print('box2: ', box2[0:10,:])
  iou = intersection/union
  #print('iou shape: ', iou.shape)
  return iou, intersection, union

# kitti obj detection labels: 'Car', 'Van', 'Truck', 'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc' or 'DontCare'

def parse_kitti_obj_det_file(filepath):
  #filepath = os.path.join(dirname, filename)
  box_vals = []
  with open(filepath,'r') as f:
    for line in f:
      vals = line.strip().split()
      print('vals: ', vals)
      label = vals[0]
      occluded = vals[2]
      box_coords = vals[4:8]  #left, top, right, bottom pixel coordinates
      confidence = float(vals[-1])
      x_c = (float(box_coords[0]) + float(box_coords[2]))/2
      y_c = (float(box_coords[1]) + float(box_coords[3]))/2
      width = float(box_coords[2])  - float(box_coords[0])
      height = float(box_coords[3]) - float(box_coords[1])
      vals = [label, x_c, y_c, width, height, confidence]
      print('label: ', label)
      print('xmin: ', float(box_coords[0]), ' xmax: ', float(box_coords[2]), ' ymin: ', float(box_coords[1]), ' ymax: ', float(box_coords[3]))
      print('x_c: ', x_c, 'y_c: ', y_c, 'width: ', width, 'height: ', height)
      box_vals.append(vals)


  
  return box_vals


def compute_anchor_box(gtboxes, scales, anchor_boxes, imgh, imgw, min_iou_threshold=0.4, min_ignore_threshold=0.4):
# box is the ground truth box
  imghts = []
  imgwdhts = []
  anchor_images = torch.zeros((len(scales), imgw, imgh))
  scaled_anchor_images = [torch.zeros((int(imgw/scales[i]), int(imgh/scales[i]))) for i in range(len(scales))]
  best_iou_boxes = [[None for i in range(len(scales))] for b in len(gtboxes)]
  best_iou_box_scaled_coords = [[None for i in range(len(scales))] for b in len(gtboxes)]
  best_anchor_box_inds = [[0 for i in range(len(scales))] for b in len(gtboxes)]
  for b, gtbox in enumerate(gtboxes):
    for i in range(len(scales)):
      #  imghts.append(int(imgh/scales[i]))
      #  imgwdhts.append(int(imgw/scales[i]))

      xmin_grid = max(0,floor(floor(gtbox[0])/scales[i]))
      ymin_grid = max(0,floor(floor(gtbox[1])/scales[i]))
      xmax_grid = min(int(imgw/scales[i]) - 1,ceil(ceil(gtbox[2])/scales[i]))
      ymax_grid = min(int(imgh/scales[i]) - 1,ceil(ceil(gtbox[3])/scales[i]))
      print('i: ', i)
      print('xmin_grid: ', xmin_grid, 'xmax_grid: ', xmax_grid)
      print('ymin_grid: ', ymin_grid, 'ymax_grid: ', ymax_grid)
      x = np.linspace(xmin_grid, xmax_grid, num=xmax_grid - xmin_grid + 1)
      y = np.linspace(ymin_grid, ymax_grid, num=ymax_grid - ymin_grid + 1)
      #print('x: ', x)
      #print('y: ', y)
      xx, yy = np.meshgrid(x, y)
      print('anchor_boxes[i]: ', anchor_boxes[i])
      print('gtbox: ', gtbox)
      #print('anchor_boxes[i].reshape((:,2)): ', anchor_boxes[i].reshape((-1,2)))
      grids = np.concatenate((xx.reshape((xx.shape[0]*xx.shape[1],1)), yy.reshape((yy.shape[0]*yy.shape[1],1)),
      xx.reshape((xx.shape[0]*xx.shape[1],1))+1, yy.reshape((yy.shape[0]*yy.shape[1],1))+1), axis=1)
      #print('grids[0:10,:]: ', grids[0:10,:])
      print('grids.shape[0]: ', grids.shape[0])
      grids_inds_val = np.zeros(grids.shape[0])
      #a3 = grids[:,0:2] + 0.5
      #print('a3: ', a3[0:10,:])
      #a2 = np.fmax(0,grids[:,0:2] + 0.5)
      #print('a2: ', a2[0:10,:])
      #a1 = np.fmax(0,grids[:,0:2] + 0.5)*scales[i] - anchor_boxes[i][0]/2
      #print('a1: ', a1[0:10,:])
      #a = np.concatenate((np.fmax(0,grids[:,0:2] + 0.5)*scales[i] - anchor_boxes[i][0]/2, (grids[:,0:2] - 0.5)*scales[i] + anchor_boxes[i][0]), axis=1)
      #print('a: ', a[0:10,:])
      boxes = np.concatenate((np.concatenate((np.fmax(0,grids[:,0:2] + 0.5)*scales[i] - anchor_boxes[i][0]/2, (grids[:,0:2] - 0.5)*scales[i] + anchor_boxes[i][0]), axis=1),
      np.concatenate((np.fmax(0,grids[:,0:2] + 0.5)*scales[i] - anchor_boxes[i][1]/2, (grids[:,0:2] + 0.5)*scales[i] + anchor_boxes[i][1]/2), axis=1), 
      np.concatenate((np.fmax(0,grids[:,0:2] + 0.5)*scales[i] - anchor_boxes[i][2]/2, (grids[:,0:2] + 0.5)*scales[i] + anchor_boxes[i][2]/2), axis=1)), axis=0)
      boxes[:,2:4] = np.fmin(boxes[:,2:4],np.array([imgw - 1, imgh - 1]))
      iou_vals, intersection, union = compute_intersection_union(gtbox, boxes)
      print('boxes shape: ', boxes.shape)
      print('iou_vals shape: ', iou_vals.shape)
      best_iou_inds = np.argsort(-iou_vals, axis=0)
      #print('iou_vals: ', iou_vals)
      best_iou_inds_minthresh = best_iou_inds[iou_vals[best_iou_inds] >= min_iou_threshold]
      print('best_iou_inds.shape: ', best_iou_inds_minthresh.shape)
      box_found = False
      best_iou_box_coord_ind = -1
      if best_iou_inds_minthresh.shape[0] > 0:      
        inds_count = 0
      #box_found = False
        while(box_found == False and inds_count < best_iou_inds_minthresh.shape[0]):
          best_anchor_box = anchor_boxes[i][floor(best_iou_inds_minthresh[inds_count] / grids.shape[0])]
          best_anchor_box_ind = floor(best_iou_inds_minthresh[inds_count] / grids.shape[0])
          print('floor(best_iou_inds[inds_count] / grids.shape[0]): ', floor(best_iou_inds_minthresh[inds_count] / grids.shape[0]))
          best_iou_box_coord = grids[best_iou_inds_minthresh[inds_count] % grids.shape[0], :]  # scaled coords
          best_iou_box_coord_ind = best_iou_inds_minthresh[inds_count] % grids.shape[0] 
          print('best_iou_box_coord: ', best_iou_box_coord)
          print('scales[i]: ', scales[i])
          scaled_coords = best_iou_box_coord[0:2]
          best_iou_box = boxes[best_iou_inds_minthresh[inds_count],:]
          if scaled_anchor_images[i][int(scaled_coords[0]), int(scaled_coords[1])] == 0:
            anchor_images[i][floor(best_iou_box[1]):ceil(best_iou_box[3]),floor(best_iou_box[0]):ceil(best_iou_box[2])] = 1
            box_found = True
            grids_inds_val[best_iou_inds_minthresh[inds_count] % grids.shape[0]] = 1
            scaled_anchor_images[i][int(scaled_coords[0]), int(scaled_coords[1])] = 1
          inds_count += 1
        
          if box_found:
            best_iou_boxes[b][i] = best_iou_box
            best_iou_box_scaled_coords[b][i] = best_iou_box_coord
            best_anchor_box_inds[b][i] = best_anchor_box_ind 
          #  print('iou_vals: ', iou_vals[best_iou_inds])
          #  print('best_anchor_box: ', best_anchor_box)
          #  print('best_iou_box: ', best_iou_box)
          #  print('gtbox: ', gtbox)
          #  print('iou: ', iou_vals[best_iou_inds[inds_count - 1]])
          #  print('intersection: ', intersection[best_iou_inds[inds_count - 1]])
          #  print('union: ', union[best_iou_inds[inds_count - 1]])
          #  print('inds_count-1: ', inds_count-1)
          #  print('scaled_anchor_image[i].shape: ', scaled_anchor_images[i].shape)
          #  print('anchor_image[i].shape: ', anchor_images[i].shape)
      best_iou_inds_ignore = best_iou_inds[iou_vals[best_iou_inds] >= min_ignore_threshold] % grids.shape[0]
      print('grids_inds_val: ', grids_inds_val)
      grids_inds_val[best_iou_inds_ignore[best_iou_inds_ignore != best_iou_box_coord_ind]] = -1  
      scaled_anchor_images[i][grids[grids_inds_val == -1,0:2]] = -1
      print('best_iou_box_coord_ind: ', best_iou_box_coord_ind)
      print('best_iou_inds_ignore: ', best_iou_inds_ignore)
      print('best_iou_inds_ignore[best_iou_inds_ignore != best_iou_box_coord_ind]: ', best_iou_inds_ignore[best_iou_inds_ignore != best_iou_box_coord_ind])
      print('grids_inds_val: ', grids_inds_val)
      print('iou_vals: ', iou_vals)
      #print('grids[0:10,:]: ', grids[0:10,:])
      #print('boxes[0:10,:]: ', boxes[0:10,:])
      #print('xx.shape: ', xx.shape)
      #print('yy.shape: ', yy.shape)
      #print('box_found: ', box_found)
      #print('grids.shape: ', grids.shape)
      #print('scaled_anchor_image[i]: ', scaled_anchor_images[i])
      #print('anchor_image[i]: ', anchor_images[i][20:55, 20:65])
      #print('xx.reshape((,1)): ', xx.reshape((xx.shape[0]*xx.shape[1],1)))
      #for x in range(xmin_grid, xmax_grid+1):
      #  for y in range(ymin_grid, ymax_grid+1):
      #    iou_vals = compute_intersection_union([x,y,x+1,y+1], box, scales[i]) 

  return anchor_images, scaled_anchor_images, best_iou_boxes, best_iou_box_scaled_coords, best_anchor_box_inds


print('config.anchor_boxes: ', config.anchor_boxes)
print('config.anchor_boxes * [1240, 375]: ', config.anchor_boxes*np.array([256/1240, 256/375]))
     
compute_anchor_box([np.array([230 * (256/1240), 106 * (256/375), 450 * (256/1240), 275 * (256/375)])], config.scales, config.anchor_boxes*np.array([256/1240, 256/375]), 256, 256)













#  for o = 1:numel(C{1})

  #% extract label, truncation, occlusion
  #lbl = C{1}(o);                   % for converting: cell -> string
  #objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
  #objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
  #objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
  #objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])

  #% extract 2D bounding box in 0-based coordinates
  #objects(o).x1 = C{5}(o); % left
  #objects(o).y1 = C{6}(o); % top
  #objects(o).x2 = C{7}(o); % right
  #objects(o).y2 = C{8}(o); % bottom


#dirname_ = "C:/Users/Kush/Downloads/data_object_label_2/training/label_2"

#filename_ = "001455.txt"

#box_vals = parse_kitti_obj_det_file(filename_, dirname_)

#for i in range(len(box_vals)):
#  print('i: ', i, ' box_vals[i]: ', box_vals[i])
