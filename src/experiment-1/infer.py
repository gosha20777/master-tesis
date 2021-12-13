
#from network_unet_rrdb_all_nobn import network
from network import network
import numpy as np
import argparse
from keras.applications.vgg16 import VGG16
import os
from keras.models import Model
from load_data import load_testing_inp
import time
import cv2

parser = argparse.ArgumentParser()

parser.add_argument('-w' ,'--weights_file', type = str, default = 'weights' , help = 'best weight file name (only prefix while evaluating)')
parser.add_argument('-dataset' ,'--dataset_path', type = str, default = '/home/puneesh/isp_learn' , help = 'complete path for the dataset')
parser.add_argument('-path' ,'--main_path', type = str, default = '/home/puneesh/deep_isp_exps' , help = 'main path where the result/experiment folders are stored')
parser.add_argument('-res' ,'--results_folder', type = str, default = 'results' , help = 'folder to save inference results')

parser.add_argument('-e' ,'--epochs', type = int, default = 100, help = 'number of epochs for testing for eval mode')
parser.add_argument('-m' ,'--metrics_file', type = str, default = 'metrics', help = 'metrics file name to be saved')
parser.add_argument('-exp' ,'--experiment_title', type = str, default = 'isp_learn', help = 'experiment folder name to save respective files')


args = parser.parse_args()
n_epochs = args.epochs
metrics_file = args.metrics_file
exp_folder = args.experiment_title

weights_file = args.weights_file
dataset_dir = args.dataset_path
current_path = args.main_path
res_folder = args.results_folder

if not os.path.exists(os.path.join(current_path,res_folder)):
   os.mkdir(os.path.join(current_path,res_folder))

in_shape = (224,224,4)

base_vgg = VGG16(weights = 'imagenet', include_top = False, input_shape = (448,448,3))
vgg = Model(inputs = base_vgg.input, outputs = base_vgg.get_layer('block4_pool').output)
for layer in vgg.layers:
     layer.trainable = False

d_model = network(vgg, inp_shape = in_shape, trainable = False)

raw_imgs = load_testing_inp(dataset_dir, 224, 224)
filename = os.path.join(current_path, weights_file)
d_model.load_weights(filename)

t1=time.time()
out,_,_,_,_ = d_model.predict(raw_imgs)
t2=time.time()
t = (t2-t1)/raw_imgs.shape[0]
print(t)

for i in range(out.shape[0]):
   I = np.uint8(out[i,:,:,:]*255.0)
   cv2.imwrite(os.path.join(current_path, res_folder) + '/' +  str(i) + '.png', I[:,:,::-1])
