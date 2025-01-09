'''
MIT License

Copyright (c) 2024 Kush Prasad


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright (c) 2021 Bernhard Walser

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


References:

[2] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10012-10022).

[3] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 2881-2890).


'''




import torch  
import torch.nn as nn  
import torch.nn.functional as F 
import torchvision.models as models  
import numpy as np
#from Exceptions import OutofIndexError
from torchvision.models import vgg16_bn
from torchvision.models import VGG16_BN_Weights
from time import time
from einops import rearrange
from torch.nn.functional import pad
from torchvision.transforms import Resize
from torch.nn.functional import interpolate
import inspect
#from load_pretrained import weights_init



class MLPblock(nn.Module):
    def __init__(self, embedding_dims, hidden_dims):
        super(MLPblock, self).__init__()
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims
        self.layerNorm = nn.LayerNorm(embedding_dims)   
        self.mlplayer = nn.Sequential(nn.Linear(in_features = embedding_dims, out_features = hidden_dims), nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None),
            nn.Dropout(p=0.30), nn.Linear(in_features = hidden_dims, out_features = embedding_dims), nn.Dropout(p=0.30))
        # changed from GELU to PRELU
    def forward(self, x):
        out = self.layerNorm(x)
        out = self.mlplayer(out) + x
        return out

class ConvLayer(nn.Module):
    def __init__(self, inputfeatures, outputinter, kernel_size=7, stride=1, padding=3, dilation=1, output=64, layertype=1, droupout=False):
        super(ConvLayer, self).__init__()
        if droupout == False:
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
        else: 
            self.layer1 = nn.Sequential(
            nn.Conv2d(inputfeatures, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer2 = nn.Sequential(
            nn.Conv2d(outputinter, outputinter, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(outputinter), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
            self.layer3 = nn.Sequential(
            nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))

        self.layer4 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=True)
        self.layer5 = nn.MaxPool2d(kernel_size = 2, stride = 2, return_indices=False)
        self.layertype = layertype

    def forward(self, x):
        #print('ConvLayer: ')
        #print('x shape: ', x.shape)
        out1 = self.layer1(x)
        #print('out1 shape: ', out1.shape)
        if self.layertype == 1:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1, inds = self.layer4(out1)
            #print('out3 shape: ', out3.shape)
            return out1, inds
        elif self.layertype == 2:
            out1 = self.layer2(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer3(out1)
            #print('out3 shape: ', out3.shape)
            out1, inds = self.layer4(out1)
            #print('out4 shape: ', out4.shape)
            return out1, inds
        elif self.layertype == 3:
            out1 = self.layer3(out1)
            return out1
        elif self.layertype == 4:
            out1 = self.layer3(out1)
            #print('out2 shape: ', out2.shape)
            out1 = self.layer5(out1)
            #print('out3 shape: ', out3.shape)
            return out1


class dbl(nn.Module):
    def __init__(self, inputfeatures=256, output=512, kernel_size=3, stride=1, padding=1, dilation=1):
        super(dbl, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(inputfeatures, output, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(output), nn.Dropout(p=0.30),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))
    
    def forward(self, x):
        out = self.layer(x)
        return out


class Resunit(nn.Module):
    def __init__(self, inputfeatures=256, outputinter=256, outputfeatures=512):
        super(Resunit, self).__init__()
        self.layers = nn.ModuleList([dbl(inputfeatures=inputfeatures, output=outputinter, kernel_size=1, stride=1, padding=0), 
            dbl(inputfeatures=outputinter, output=outputfeatures, kernel_size=3, stride=1, padding=1)])

    def forward(self, x):
        print('inside Resunit, x shape: ', x.shape)
        out = x.clone()
        for l in self.layers:
            out = l(out)
        out = out + x
        return out


class ResBlock(nn.Module):
    def __init__(self, inputfeatures=256, outputfeatures=512, numUnits=1):
        super(ResBlock, self).__init__()
        self.dbl = dbl(inputfeatures=inputfeatures, output=outputfeatures, kernel_size=3, stride=2, padding=1)
        self.resunits = nn.ModuleList([Resunit(inputfeatures=outputfeatures, outputinter=inputfeatures, outputfeatures=outputfeatures) for i in range(numUnits)])

    def forward(self, x):
        print('inside ResBlock, x.shape: ', x.shape)
        out = self.dbl(x)
        print('inside ResBlock after dbl, out.shape: ', out.shape)
        for i in range(len(self.resunits)):
            out = self.resunits[i](out)
            print('i: ', i, ' inside ResBlock after resunits, out.shape: ', out.shape)
        return out


class yoloprediction(nn.Module):
    def __init__(self, inputfeatures=1024, outputinter=512, output=255, kernel_size=3):
        super(yoloprediction, self).__init__()

        self.pred = nn.ModuleList([dbl(inputfeatures=inputfeatures, output=outputinter), nn.Conv2d(outputinter, output, kernel_size=kernel_size, stride=1, padding=1)])

    def forward(self, x):
        out = x
        for l in self.pred:
            out = l(out)
        out[4:] = torch.sigmoid(out[4:])
        return out


class yolov3(nn.Module):
    def __init__(self, num_classes, inputfeatures=[32, 64, 128, 256, 512], outputfeatures=[64, 128, 256, 512, 1024], in_channels=3, numResUnits=[1, 2, 8, 8, 4], imgh=360, imgw=480, auxillary_loss=False, output_layer='last', initialize_weights=False):
        super(yolov3, self).__init__()

        self.imgh = imgh
        self.imgw = imgw
        self.last_layer = output_layer
        self.inputlayer = dbl(inputfeatures=in_channels, output=inputfeatures[0], kernel_size=3, stride=1, padding=1)
        self.ResBlocks = nn.ModuleList([ResBlock(inputfeatures=inputfeatures[i], outputfeatures=outputfeatures[i], numUnits=numResUnits[i]) for i in range(len(numResUnits))])
        self.yolo1conv = nn.ModuleList([dbl(inputfeatures=outputfeatures[-1], output=outputfeatures[-2], kernel_size=3, stride=1) if (i % 2 == 0) else 
        dbl(inputfeatures=outputfeatures[-2], output=outputfeatures[-1], kernel_size=3, stride=1) for i in range(5)])
        self.yolo2up = nn.ModuleList([dbl(inputfeatures=inputfeatures[-1], output=inputfeatures[-2], kernel_size=3), nn.Upsample(scale_factor=2, mode='bilinear')])        
        self.yolo2conv = nn.ModuleList([dbl(inputfeatures=inputfeatures[-2] + inputfeatures[-1], output=inputfeatures[-2], kernel_size=3, stride=1)] +
        [dbl(inputfeatures=inputfeatures[-2], output=inputfeatures[-1], kernel_size=3, stride=1) if (i % 2 == 1) 
        else dbl(inputfeatures=inputfeatures[-1], output=inputfeatures[-2], kernel_size=3, stride=1) for i in range(1,5)])        
        self.yolo3up = nn.ModuleList([dbl(inputfeatures=inputfeatures[-2], output=inputfeatures[-3], kernel_size=3), nn.Upsample(scale_factor=2, mode='bilinear')])
        self.yolo3conv = nn.ModuleList([dbl(inputfeatures=inputfeatures[-3] + inputfeatures[-2], output=inputfeatures[-3], kernel_size=3, stride=1)] +
            [dbl(inputfeatures=inputfeatures[-3], output=inputfeatures[-2], kernel_size=3, stride=1) if (i % 2 == 1) else 
            dbl(inputfeatures=inputfeatures[-2], output=inputfeatures[-3], kernel_size=3, stride=1) for i in range(1, 5)])
        self.yolopred1 = yoloprediction(inputfeatures=outputfeatures[-2], outputinter=outputfeatures[-1], output=3*(4 + 1 + num_classes), kernel_size=3)
        self.yolopred2 = yoloprediction(inputfeatures=outputfeatures[-3], outputinter=outputfeatures[-2], output=3*(4 + 1 + num_classes), kernel_size=3)
        self.yolopred3 = yoloprediction(inputfeatures=outputfeatures[-4], outputinter=outputfeatures[-3], output=3*(4 + 1 + num_classes), kernel_size=3)



        self.layers_to_be_initialized = []
        if initialize_weights:
            #weights_init(self.swintransformer)
            #weights_init(self.PPMhead)
            #weights_init(self.crf1)
            #weights_init(self.crf2)

            members = inspect.getmembers(self)
            for m in members:
                #print('m[1].__class__.__bases__[0].__name__: ', m[1].__class__.__bases__[0].__name__)
                if m[0] in self.layers_to_be_initialized:
                    #if m[1].__class__.__bases__[0].__name__ == 'Module' and m[1].__class__:
                    print('m.__class__: ', m[1].__class__)
                    print('m.__class__.__name__: ', m[1].__class__.__name__)
                    print('m: ', m[0])
                    print('m: ', type(m[1]))
                    print('m[1].__bases__: ', m[1].__class__.__bases__)
                    print('initializing weights')
                    weights_init(m[1])



    def forward(self, x):
        #start_time = time()
        print('x shape: ', x.shape)
        out = self.inputlayer(x) # output filters: 32
        print('out shape after inputlayer: ', out.shape)
        out = self.ResBlocks[0](out)  # output filters: 64
        print('out shape after ResBlocks[0]: ', out.shape)
        out = self.ResBlocks[1](out)  # output filters: 128
        print('out shape after ResBlocks[1]: ', out.shape)
        outRes2 = self.ResBlocks[2](out)  # output filters: 256
        print('outRees2 shape after ResBlocks[2]: ', outRes2.shape)
        outRes3 = self.ResBlocks[3](outRes2)
        print('outRes3 shape after ResBlocks[3]: ', outRes3.shape)
        out = self.ResBlocks[4](outRes3)  # output filters: 512
        print('out shape after ResBlocks[4]: ', out.shape)
        for l in self.yolo1conv:
            out = l(out)
            print('yolo1conv out shape: ', out.shape)
        print('out shape after yolo1conv: ', out.shape)
        outyoloconv1 = out.clone()
        out = self.yolopred1(out)
        print('out shape after yolopred1: ', out.shape)
        print('outRes3 shape after yolopred1: ', outRes3.shape)
        out2 = outyoloconv1
        for l in self.yolo2up:
            out2 = l(out2)
        print('out2 shape after yolo2up: ', out2.shape)
        out2 = torch.cat((out2, outRes3), dim=1)
        print('out2 shape after cat: ', out2.shape)
        for l in self.yolo2conv:
            out2 = l(out2)
        print('out2 shape after yoloconv2: ', out2.shape)
        outyoloconv2 = out2.clone()
        out2 = self.yolopred2(out2)
        print('out2 shape after yolopred2: ', out2.shape)
        out3 = outyoloconv2
        for l in self.yolo3up:
            out3 = l(out3)
        print('out3 shape after yolo3up: ', out3.shape)
        out3 = torch.cat((out3, outRes2), dim=1)
        print('out3 shape after cat: ', out3.shape)
        for l in self.yolo3conv:
            out3 = l(out3)
        print('out3 shape after yolo3conv: ', out3.shape)
        out3 = self.yolopred3(out3)
        print('out3 shape after yolopred3: ', out3.shape)
        return out, out2, out3







class PSPhead(nn.Module):
    def __init__(self, input_dim=1024, output_dims=256, final_output_dims=1024, pool_scales=[1,2,3,6]):
        super(PSPhead, self).__init__()
        self.ppm_modules = nn.ModuleList([nn.Sequential(nn.AdaptiveAvgPool2d(pool), nn.Conv2d(input_dim, output_dims, kernel_size=1),
            nn.BatchNorm2d(output_dims),
            #nn.ReLU())
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None)) for pool in pool_scales])

        self.bottleneck = nn.Sequential(nn.Conv2d(input_dim + output_dims*len(pool_scales), final_output_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(final_output_dims),
            nn.PReLU(num_parameters=1, init=0.25, device=None, dtype=None))


    def forward(self, x):
        x = x.permute((0,3,1,2))
        ppm_outs = []
        ppm_outs.append(x)
        for ppm in self.ppm_modules:
            #ppm_out = Resize((x.shape[2], x.shape[3]), interpolation=InterpolationMode.BILINEAR)
            ppm_out = interpolate(ppm(x), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=None)
            ppm_outs.append(ppm_out)
        ppm_outs = torch.cat(ppm_outs, dim=1)
        #print('ppm_outs shape: ', ppm_outs.shape)
        ppm_head_out = self.bottleneck(ppm_outs)
        #ppm_head_out = ppm_head_out.permute((0,2,3,1))
        #print('ppm_head_out shape: ', ppm_head_out.shape)
        return ppm_head_out



