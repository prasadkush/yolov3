#from preprocess import get_std
import cv2
import numpy as np
import torch.nn as nn
from data import getDataset, get_inverse_transforms
from torch.utils.data import DataLoader, Dataset
import torch
from yolov3 import yolov3
import os
#from test_segmentation_camvid import label_colours
from data_config import rawdatalist, depthdatalist, valrawdatalist, valdepthdatalist
from time import time
from torchsummary import summary
#from losses import depth_loss
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pickle
import inspect

print('hi')


'''
resultsdiropen = 'results/trial3'

with open(resultsdiropen + '/training_loss_list.pkl', 'rb') as f:
	training_loss_list = pickle.load(f)
    print('training_loss_list: ', training_loss_list)
        #with open(resultsdir + '/mean_list.pkl', 'rb') as f:
        #    mean_list = pickle.load(f)
with open(resultsdiropen + '/absrellist.pkl', 'rb') as f:
	absrellist = pickle.load(f)
	print('absrellist: ', absrellist)
with open(resultsdiropen + '/rmselist.pkl', 'rb') as f:
	rmselist = pickle.load(f)
	print('rmselist: ', rmselist)
with open(resultsdiropen + '/absrel_val_list.pkl', 'rb') as f:
	absrel_val_list = pickle.load(f)
    print('absrel_val_list: ', absrel_val_list)
with open(resultsdiropen + '/rmse_val_list.pkl', 'rb') as f:
	rmse_val_list = pickle.load(f)
	print('rmse_val_list: ', rmse_val_list)
with open(resultsdiropen + '/loss_val_list.pkl', 'rb') as f:
	loss_val_list = pickle.load(f)
	print('loss_val_list: ', loss_val_list)

#std, mean = get_std(datapath=None, mean=None, dataset='kitti')
#print('std: ', std)
#print('mean: ', mean)
'''

batch_size = 4

batch_size_val = 4

imgh = 256
imgw = 256

#dataset = getDataset(rawdatapath=rawdatapath, depthdatapath=depthdatapath, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment=False, gt_present=True, mode='train')
dataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.20, gt_present=True, mode='train', resizex=imgw, resizey=imgh)

valdataset = getDataset(rawdatapath=valrawdatalist, depthdatapath=valdepthdatalist, max_depth=85, pct=0.1, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0, gt_present=True, mode='val', resizex=imgw, resizey=imgh)
data_loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)

val_data_loader = DataLoader(valdataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)

#dataset = getDataset(rawdatapath=rawdatalist, depthdatapath=depthdatalist, max_depth=85, pct=1.0, train_val_split=1.0, dataset='kitti', data_augment_flip=0, data_augment_brightness_color=0.20, gt_present=True, mode='train', resizex=224, resizey=224)

resultsdir = 'results/trial_layer_wise'

resultsdir_pretrained = 'results/trial_crf2'

modelpath_pretrained = resultsdir_pretrained + '/bestabsreldepthmodelnew.pt'

model = yolov3(inputfeatures=[32, 64, 128, 256, 512], outputfeatures=[64, 128, 256, 512, 1024], in_channels=3, numResUnits=[1, 2, 8, 8, 4], imgh=imgh, imgw=imgw, auxillary_loss=False, output_layer='last', initialize_weights=False)


#summary(model, input_size = (3, 256, 256), batch_size=4)


print('model: ', model)

#print('loading weigts')

#load_pretrained(modelpath_pretrained, model, last_layer='crf2')

#criterion = depth_loss(lambda_=0.55, alpha=10)



#data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)


#val_data_loader = DataLoader(valdataset, batch_size=batch_size_val, shuffle=True, pin_memory=True)

'''
members = inspect.getmembers(model)


for m in members:
	#print('m[1].__class__.__bases__[0].__name__: ', m[1].__class__.__bases__[0].__name__)
	if m[1].__class__.__bases__[0].__name__ == 'Module' and m[1].__class__:
		print('m.__class__: ', m[1].__class__)
		print('m.__class__.__name__: ', m[1].__class__.__name__)
		#print('len(m): ', len(m))
		print('m: ', m[0])
		print('m: ', type(m[1]))
		print('m[1].__bases__: ', m[1].__class__.__bases__)
'''

'''
for name, param in list(model.named_parameters()):
	print('name: ', name)
	#weights_init(param)
	if len(param.shape) == 4:
		print('param[0][0][0][0:5]: ', param[0][0][0][0:5])
	elif len(param.shape) == 3:
		print('param[0][0][0:5]: ', param[0][0][0:5])
	elif len(param.shape) == 2:
		print('param[0][0:5]: ', param[0][0:5])
	elif len(param.shape) == 1:
		print('param[0:5]: ', param[0:5])
'''



#train(data_loader, val_data_loader, model, criterion, epochs=16, batch_size=batch_size, batch_size_val=batch_size_val, dataset_name='kitti', shuffle=True, resume_training=False, resultsdir=resultsdir, auxillary_loss=True, resultsdiropen=None, modelpath=None, initialize_from_model=False, initialize_weights=False)
#loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

loaderiter = iter(data_loader)

data = next(loaderiter)

#print('data:', data)
d = data['image'].numpy()
print('data[image] shape: ', data['image'].shape)
out1, out2, out3  = model(data['image'])
print('out1 shape: ', out1.shape)
print('out2 shape: ', out2.shape)
print('out3 shape: ', out3.shape)

#predict_and_visualize(loader, batch_size=4, dataset_name='kitti', imgdir=None, model=model, modelpath=modelpath, modelname='NeWCRFDepth', gt_present=True, save_images=False)
#segmimgdir = imgdir + '/segm'
#origimgdir = imgdir + '/orig'
#overlayimgdir = imgdir + '/overlay'

#checkpoint = torch.load(modelpath)
#print('checkpoint[epoch]: ', checkpoint['epoch'])
#print('checkpoint[loss]: ', checkpoint['loss'])
#print('checkpoint[mean_iou]: ', checkpoint['mean_iou'])

#resultsdiropen = 'results/trial2'




'''
#loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)
pixelacc, meaniou, loss, intersect, union = compute_accuracy(datasetval, dataset_name='CamVid', imgdir=imgdir, model=model, modelpath=modelpath, modelname='SegmentationDil4', gt_present=True, save_images=True, criterion=criterion)
print('pixelacc: ', pixelacc)
print('loss: ', loss)
print('mean_iou: ', meaniou)
print('intersect: ', intersect)
print('union: ', union)
'''

'''
i = 0
alpha = 0.50
newimg = 255*np.ones((360,480,3))
#overimg = 255*np.ones((360,480,3))
blankcol = 255*np.ones((360,3,3))
for i in range(101):
	imgpath = segmimgdir + '/' + str(i) + '_outimg_.jpg'
	segm = cv2.imread(imgpath)
	origimgpath = origimgdir +'/' + str(i) + '_imgorig_.jpg'
	origimg = cv2.imread(origimgpath)
	overpath = overlayimgdir + '/' + str(i) + '_overlayimg_.jpg'
	overimg = cv2.addWeighted(segm, alpha, origimg, 1 - alpha,
		0)
	#cv2.imwrite(imgdir + '/' + str(i) + '_overlayimg_.jpg', overimg)
	#overlayimg = cv2.imread(overpath)
	newimg = np.concatenate((segm, blankcol, origimg, blankcol, overimg),axis=1)
	cv2.imwrite(overlayimgdir + '/' + str(i) + '_overlayimg_.jpg', newimg)
'''

#summary(model, input_size = (3, 360, 480), batch_size=4)

#overlay = cv2.imread(imgdir + '/1_outimg_.jpg')
#output = cv2.imread(imgdir + '/1_imgorig_.jpg')
#alpha = 0.5

#cv2.addWeighted(overlay, alpha, output, 1 - alpha,
#		0, output)

#cv2.imwrite(imgdir + '/1_overlayed.jpg', output)

#loader = DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True)
#pixelacc, meaniou, loss = compute_accuracy(dataset, dataset_name='kitti', imgdir=None, model=None, modelpath=modelpath, modelname='SegnetSkip', gt_present=True, save_images=False, criterion=criterion)
#print('pixelacc: ', pixelacc)
#print('loss: ', loss)
#print('mean_iou: ', meaniou)

#loader = DataLoader(dataset, batch_size=4, shuffle=True, pin_memory=True)

#loaderiter = iter(loader)

#data = next(loaderiter)

#color_arr = label_colours
#unarr = np.empty((0,))
#Normalize = get_inverse_transforms('kitti')

#print('data:', data)
'''
for i, data in enumerate(loader):
	d = data['image']
	ds = data['semantic']
	#print('d.shape: ', d.shape )
	#print('np.unique(np.flatten(ds)): ', np.unique(ds[2,:,:,].flatten()))
	#print('np.unique(np.flatten(ds)): ', np.unique(ds[3,:,:,].flatten()))
	indices = np.indices((d.shape[0], 360,480))
	outimg2 = np.ones((d.shape[0], 360,480,3))
	#outimg2[indices[0,:,:],indices[1,:,:],:] = color_arr[imgs]
	outimg2[indices[0,:,:,:], indices[1,:,:,:],indices[2,:,:,:],:] = color_arr[ds]
	outimg2 = outimg2.astype('uint8')	#
	imgorig = Normalize(d)
	imgorig = torch.permute(imgorig, (0,2,3,1))
	imgorig = 255*imgorig
	imgorig = imgorig.numpy().astype('ui#nt8')
	print('d.shape: ', d.shape#)
	print('ds.shape: ', ds.sha#pe)
	for j in range(d.shape[0])#:
		arr = np.unique(ds[j,:#,:,].flatten())
		unarr = np.union1d(unarr, arr)
		print('arr: ', arr)
		print('unarr: ', unarr)
		#cv2.imshow('seg: ', outimg2[i,:,:,:])
		#cv2.imshow('img: ', imgorig[i,:,:,:])
		#cv2.waitKey(0)
'''


#loss_criterion = depth_loss()
#rmse, absrel_error, sqrel_error, loss = compute_metrics(valdataset, dataset_name='kitti', imgdir='results', model=model, modelpath=None, modelname='NeWCRFDepth', gt_present=True, save_images=False, criterion=loss_criterion, epoch=1)

#print('rmse: ', rmse)
#print('absrel_error: ', absrel_error)
#print('sqrel_error: ', sqrel_error)
#print('loss: ', loss)

'''
blankcol = 255*np.ones((360,3,3)).astype(np.uint8)
plasma = plt.get_cmap('plasma')
greys = plt.get_cmap('Greys')
eps = 1e-7
normalizer = mpl.colors.Normalize(vmin=0, vmax=85)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
for i, data in enumerate(loader):
	img = data['image']
	imgorig1 = data['original']
	#x = model.forward(img)
	#print('x shape: ', x.shape)
	d = data['gt']
	#rmse = RMSE(x, d)
	#absrel_error = abs_rel_error(x,d)
	#sqrel_err = sq_rel_error(x,d)
	#loss = loss_criterion(x, d)
	#print('loss: ', loss)
	#print('RMSE: ', rmse)
	#print('absrel_error: ', absrel_error)
	#print('sq_rel_error: ', sqrel_err)
	imgorig = Normalize(img)
	imgorig = torch.permute(imgorig, (0,2,3,1))
	imgorig1 = torch.permute(imgorig1, (0,2,3,1))
	imgorign = imgorig.numpy()
	imgorign = (255*imgorign).astype('uint8')
	#imgorig = imgorig.numpy().astype('uint8')
	dn = d.numpy()[:,:,:]
	#dn = np.clip(dn,0,80)
	#depth = depth_png.astype(np.float) / 256.
    #depth[depth_png == 0] = -1.
	#print('dn shape: ', dn.shape)
	#print('d[0,:,:,0] == d[0,:,:,1]: ', np.sum(d[0,:,:,0] == d[0,:,:,1]))
	for j in range(img.shape[0]):
		#print('np.min(dn[j,:,:]): ', np.min(dn[j,:,:]))
		#print('np.max(dn[j,:,:]): ', np.max(dn[j,:,:]))
		#print('dn[j,:,:]*255/80: ', dn[j,:,:]*255/80)
		#print('dn[j,:,:]: ', dn[j,:,:])
		coloredDepth = (mapper.to_rgba(dn[j,:,:])[:, :, :3] * 255).astype(np.uint8)
		print('min mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', np.min(mapper.to_rgba(dn[j,:,:])[:, :, :3]))
		print('max mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', np.max(mapper.to_rgba(dn[j,:,:])[:, :, :3]))
		#print('mapper.to_rgba(dn[j,:,:])[:, :, :3]: ', mapper.to_rgba(dn[j,:,:])[:, :, :3])
		#coloredDepth = (greys(np.log10(dn[j,:,:]))[:, :, :3] * 255).astype('uint8')
		print('coloredDepth shape: ', coloredDepth.shape)
		print('coloredDepth type: ', type(coloredDepth))
		#print('imgorig[j,:,:,0]: ', imgorig[j,:,:,0])
		newimg = np.concatenate((imgorig[j,:,:,:], blankcol, coloredDepth, blankcol, imgorig1[j,:,:,:]), axis=1)
		dn[j,:,:] = ((dn[j,:,:]/85)*255).astype('uint8')
		#print('newimg: ', newimg[:,:,0])
		#cv2.imshow('img: ', imgorig[j,:,:,:])
		#cv2.imshow('coloredDepth: ', coloredDepth)
		cv2.imwrite('results/imgorig' + str(j) + '.png', imgorign[j,:,:,:])
		cv2.imwrite('results/depth' + str(j) + '.png', coloredDepth)
		cv2.imwrite('results/depth2' + str(j) + '.png', dn[j,:,:])
		cv2.imshow('newimg: ', newimg)
		cv2.waitKey(0)
		#print('min depth value: ', torch.min(d[j,:,:,:]))
		#print('max depth value: ', torch.max(d[j,:,:,:]))

'''

'''

loss_criterion = depth_loss()
d = data['image']
gt = data['gt']
#d = torch.unsqueeze(d,0)
#net = Segnet(7,3)
print('d shape: ', d.shape)
start_time = time()
x = model.forward(d)
end_time = time()
x = torch.squeeze(x,1)
print('x.shape: ', x.shape)
print('gt shape: ', gt.shape)
print('avg_time: ', (end_time - start_time)/x.shape[0])




loss = loss_criterion(x, gt)

print('loss: ', loss)

#print('std: ', std)
#print('mean: ', mean)

#cv2.imshow('img: ', img)
#cv2.waitKey(0)

'''