import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--num_images', type=int, default=-1, help='Number of images to test. -1 for all images')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--outdir_prefix', type=str, default="", required=False, help='Prefix to add to output directory')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,4])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

test_miou, test_miou_op = tf.metrics.mean_iou(labels=tf.argmax(net_output, -1),   
                                              predictions=tf.argmax(network, -1),
                                              num_classes=num_classes)

sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names, train_input_depth_names, train_output_names, val_input_names, val_input_depth_names, val_output_names, test_input_names, test_input_depth_names, test_output_names = utils.prepare_data_with_depth(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s"%("Test")):
        os.makedirs("%s"%("Test"))
        
num_test_images = len(test_input_names) if args.num_images == -1 else int(args.num_images)

# calculate the start crop coords for the exact middle of the image
in_h,in_w = cv2.imread(test_input_names[0]).shape[:2]
crp_xs = int(in_w/2)-int(args.crop_width/2)
crp_ys = int(in_h/2)-int(args.crop_height/2)
crp_xe = int(crp_xs + args.crop_width)
crp_ye = int(crp_ys + args.crop_height)

run_times_list = []

model_name = args.model
if len(args.outdir_prefix) > 0:
    model_name = args.outdir_prefix + "_" + model_name
    
dataset_dir = args.dataset if args.dataset[-1] != "/" else args.dataset[:-1]

file_dir = "%s/%s_%s"%("Test", model_name, os.path.basename(dataset_dir))
if not os.path.isdir(file_dir):
    os.makedirs(file_dir)

img_ind_shuffle = np.random.permutation(len(test_input_names))
    
# Run testing on ALL test images
for ind,img_ind in enumerate(img_ind_shuffle[:num_test_images]):
    sys.stdout.write("\rRunning test image %d / %d: "%(ind+1, num_test_images))
    sys.stdout.flush()

    input_image = utils.load_image(test_input_names[img_ind])

    # add the depth as the last channel to input
    input_depth_image = utils.load_image(test_input_depth_names[img_ind])
    if len(input_depth_image.shape) == 3 and input_depth_image.shape[2] == 3:
        input_depth_image = cv2.cvtColor(input_depth_image, cv2.COLOR_BGR2GRAY)
    input_image = np.dstack((input_image, input_depth_image))
    
    input_image = input_image[crp_ys:crp_ye,crp_xs:crp_xe]
    input_image = np.expand_dims(np.float32(input_image),axis=0)/255.0

    gt = utils.load_image(test_output_names[img_ind])[crp_ys:crp_ye,crp_xs:crp_xe]
    gt = np.expand_dims(helpers.one_hot_it(gt, label_values, tolerance=50), axis=0)

    st = time.time()
    
    output_image,curr_test_miou_op = sess.run([network,test_miou_op],feed_dict={net_input:input_image,net_output:gt})
    curr_test_miou = sess.run(test_miou,feed_dict={net_input:input_image,net_output:gt})
    
    print("Current mIOU %.3f"%(curr_test_miou))

    run_times_list.append(time.time()-st)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    gt = helpers.colour_code_segmentation(helpers.reverse_one_hot(gt[0,:,:,:]), label_values)

    file_name = os.path.splitext(os.path.basename(test_input_names[ind]))[0]
    cv2.imwrite("%s/%s_pred.png"%(file_dir, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%(file_dir, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

# normalize confusion matrix
curr_test_miou_op = curr_test_miou_op.astype('float') / curr_test_miou_op.sum(axis=1)[:, np.newaxis]

df = pd.DataFrame(curr_test_miou_op)
    
hm = sn.heatmap(df)
hm.get_figure().savefig(file_dir + "/confusion_matrix.png")

df.to_csv(file_dir+"/confusion_matrix.csv")

with open(file_dir+"/miou.txt", "w") as outf:
    outf.write("%.3f\n"%curr_test_miou)
    outf.close()

print("Overall mIOU %.3f"%(curr_test_miou))
print("Average inference time %.3f"%(np.mean(run_times_list)))
