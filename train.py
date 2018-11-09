from __future__ import print_function
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
import pandas as pd
import missinglink

missinglink_project = missinglink.TensorFlowProject(project='5730650862649344')

# use 'Agg' on matplotlib so that plots could be generated even without Xserver
# running
import matplotlib
matplotlib.use('Agg')

from utils import utils, helpers
from builders import model_builder

import matplotlib.pyplot as plt

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=5, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_batches', type=int, default=-1, help='Number of batches per epoch, use (-1) to derive from dataset size')
parser.add_argument('--num_val_images', type=int, default=20, help='The number of images to used for validations')
parser.add_argument('--num_inter_val_imgs', type=int, default=10, help='Number of images to used for intermediate validation within an epoch')
parser.add_argument('--num_inter_val_iters', type=int, default=20, help='Number of iterations after which to run intermediate validation')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change as a factor between 0.0 and 1.0. For example, 0.1 represents a max brightness change of 10%% (+-).')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle in degrees.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
args = parser.parse_args()


def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = 1.0 + random.uniform(-1.0*args.brightness, args.brightness)
        table = np.array([((i / 255.0) * factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=cv2.INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=cv2.INTER_NEAREST)

    return input_image, output_image


# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)


# Compute your softmax cross entropy loss
net_input = tf.placeholder(tf.float32,shape=[None,None,None,4])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes])

network, init_fn = model_builder.build_model(model_name=args.model, frontend=args.frontend, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output))

with tf.name_scope('train'):
    train_acc, train_acc_op = tf.metrics.accuracy(labels=tf.argmax(net_output, -1),   
                                      predictions=tf.argmax(network, -1))

    train_miou, train_miou_op = tf.metrics.mean_iou(labels=tf.argmax(net_output, -1),   
                                      predictions=tf.argmax(network, -1),
                                        num_classes=num_classes)

with tf.name_scope('val'):
    val_acc, val_acc_op = tf.metrics.accuracy(labels=tf.argmax(net_output, -1),   
                                      predictions=tf.argmax(network, -1))

    val_miou, val_miou_op = tf.metrics.mean_iou(labels=tf.argmax(net_output, -1),   
                                      predictions=tf.argmax(network, -1),
                                        num_classes=num_classes)

opt = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.995).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + os.path.basename(args.dataset) + ".ckpt"
if args.continue_training:
    print('Loading latest model checkpoint from: %s'%model_checkpoint_name)
    saver.restore(sess, model_checkpoint_name)

# Load the data
print("Loading the data ...")
train_input_names, train_input_depth_names, train_output_names, val_input_names, val_input_depth_names, val_output_names, test_input_names, test_input_depth_names, test_output_names = utils.prepare_data_with_depth(dataset_dir=args.dataset)



print("\n***** Begin training *****")
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Num Classes -->", num_classes)
print("Num train images -->", len(train_input_names))
print("Num val images -->", len(val_input_names))
print("Num test images -->", len(test_input_names))

print("Data Augmentation:")
print("\tVertical Flip -->", args.v_flip)
print("\tHorizontal Flip -->", args.h_flip)
print("\tBrightness Alteration -->", args.brightness)
print("\tRotation -->", args.rotation)
print("")

avg_loss_per_epoch = []
avg_scores_per_epoch = []
avg_iou_per_epoch = []

# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names)),num_vals)

# calculate the start crop coords for the exact middle of the image
in_h,in_w = cv2.imread(train_input_names[0]).shape[:2]
crp_x = int(in_w/2)-int(args.crop_width/2)
crp_y = int(in_h/2)-int(args.crop_height/2)


with missinglink_project.create_experiment(
    display_name='%s %s'%(args.model, args.dataset),
    description='...') as experiment:
    
    NUM_SAMPLE = len(train_input_names)
    NUM_BATCHES = int(NUM_SAMPLE / args.batch_size)
    
    if int(args.num_batches) > 0:
        NUM_BATCHES = int(args.num_batches)
        
    print("Num batches per epoch -->", NUM_BATCHES)

    miou_history = []

    # Do the training here
    for epoch in experiment.epoch_loop(args.num_epochs):

        current_losses = []

        cnt=0

        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))

        st = time.time()
        epoch_st=time.time()
        
        for i in experiment.batch_loop(NUM_BATCHES):

            input_image_batch = []
            output_image_batch = []

            # Collect a batch of images
            for j in range(args.batch_size):
                index = i*args.batch_size + j
                id = id_list[index]
                input_image = utils.load_image(train_input_names[id])

                # add the depth as the last channel to input
                input_depth_image = utils.load_image(train_input_depth_names[id])
                if len(input_depth_image.shape) == 3 and input_depth_image.shape[2] == 3:
                    input_depth_image = cv2.cvtColor(input_depth_image, cv2.COLOR_BGR2GRAY)
                input_image = np.dstack((input_image, input_depth_image))

                output_image = utils.load_image(train_output_names[id])

                with tf.device('/cpu:0'):
                    input_image, output_image = data_augmentation(input_image, output_image)

                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) / 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values, tolerance=50))

                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            with experiment.train(monitored_metrics={'loss': loss, 'acc': train_acc, 'miou': train_miou}):
                _,current,_,_=sess.run([opt,loss,train_acc_op,train_miou_op],feed_dict={net_input:input_image_batch,
                                                                          net_output:output_image_batch})
                        
            current_losses.append(current)
            cnt = cnt + args.batch_size
            
            if cnt % int(args.num_inter_val_iters) == 0:
                val_shuffle = np.random.permutation(len(val_indices))
                # Do the validation on a small set of validation images
                with experiment.validation(monitored_metrics={'loss': loss, 'acc': val_acc, 'miou': val_miou}):
                    for ind in val_shuffle[:int(args.num_inter_val_imgs)]:
                        rand_ind = val_indices[ind]
                        input_image = utils.load_image(val_input_names[rand_ind])

                        # add the depth as the last channel to input
                        input_depth_image = utils.load_image(val_input_depth_names[rand_ind])
                        if len(input_depth_image.shape) == 3 and input_depth_image.shape[2] == 3:
                            input_depth_image = cv2.cvtColor(input_depth_image, cv2.COLOR_BGR2GRAY)
                        input_image = np.dstack((input_image, input_depth_image))

                        input_cropped = input_image[crp_y:crp_y+512,crp_x:crp_x+512]
                        input_image = np.expand_dims(np.float32(input_cropped),axis=0)/255.0

                        gt = utils.load_image(val_output_names[rand_ind])[crp_y:crp_y+512,crp_x:crp_x+512]
                        gt = np.expand_dims(helpers.one_hot_it(gt, label_values, tolerance=50), axis=0)

                        _,_,_ = sess.run([loss,val_acc_op,val_miou_op],feed_dict={net_input:input_image,net_output:gt})
                
                    curr_val_miou = sess.run(val_miou,feed_dict={net_input:input_image,net_output:gt})
                    
                string_print = "Epoch = %d Count = %d/%d Current_Loss = %.4f Time = %.2f Val mIOU = %.3f"%(epoch,cnt,len(train_input_names),current,time.time()-st,curr_val_miou)
                utils.LOG(string_print)
                st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)

        # Create directories if needed
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
            os.makedirs("%s/%04d"%("checkpoints",epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint:", model_checkpoint_name)
        saver.save(sess,model_checkpoint_name)

        if epoch % args.validation_step == 0:
            print("Performing validation")
            with experiment.validation():
                # re-init val metrics counters
                stream_vars_valid = [v for v in tf.local_variables() if 'valid/' in v.name]
                sess.run(tf.variables_initializer(stream_vars_valid))

                for ind in val_indices:
                    input_image = utils.load_image(val_input_names[ind])

                    # add the depth as the last channel to input
                    input_depth_image = utils.load_image(val_input_depth_names[ind])
                    if len(input_depth_image.shape) == 3 and input_depth_image.shape[2] == 3:
                        input_depth_image = cv2.cvtColor(input_depth_image, cv2.COLOR_BGR2GRAY)
                    input_image = np.dstack((input_image, input_depth_image))

                    input_cropped = input_image[crp_y:crp_y+512,crp_x:crp_x+512]
                    input_image = np.expand_dims(np.float32(input_cropped),axis=0)/255.0

                    gt = utils.load_image(val_output_names[ind])[crp_y:crp_y+512,crp_x:crp_x+512]
                    gt = np.expand_dims(helpers.one_hot_it(gt, label_values, tolerance=50), axis=0)

                    output_image,_,_ = sess.run([network,val_acc_op,val_miou_op],feed_dict={net_input:input_image,net_output:gt})
                    curr_val_miou,curr_val_acc = sess.run([val_miou,val_acc],feed_dict={net_input:input_image,net_output:gt})

                    output_image = np.array(output_image[0,:,:,:])
                    output_image = helpers.reverse_one_hot(output_image)
                    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                    gt = helpers.colour_code_segmentation(helpers.reverse_one_hot(gt[0,:,:,:]), label_values)

                    file_name = os.path.splitext(os.path.basename(val_input_names[ind]))[0]
                    cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                    cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
            
            if (len(miou_history) > 0) and (curr_val_miou > np.max(miou_history)):
                epoch_ckpt_filename = "%s/best_model_%s_%s.ckpt"%("checkpoints",args.model,os.path.basename(args.dataset))
                print("Record miou - saving weights for this epoch:",epoch_ckpt_filename)
                saver.save(sess,epoch_ckpt_filename)
            
            miou_history += [curr_val_miou]

            print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, curr_val_acc))
            print("Validation IoU score = ", curr_val_miou)

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(args.num_epochs-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []

