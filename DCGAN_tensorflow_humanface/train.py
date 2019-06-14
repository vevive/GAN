#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Train a DCGAN to generating fake images.

Example Usage:
---------------
python3 train.py \
    --images_dir: Path to real images directory.
    --images_pattern: The pattern of input images.
    --generated_images_save_dir: Path to directory where to write gen images.
    --logdir: Path to log directory.
    --num_steps: Number of steps.
"""

import cv2
import glob
import numpy as np
import os
import tensorflow as tf

import model

flags = tf.flags

flags.DEFINE_string('images_dir', './Training_Data/face', 'Path to real images directory.')
flags.DEFINE_string('images_pattern', '*.jpg', 'The pattern of input images.')
flags.DEFINE_string('generated_images_save_dir', './generated_images', 'Path to directory '
                    'where to write generated images.')
flags.DEFINE_string('logdir', './training', 'Path to log directory.')
flags.DEFINE_integer('num_steps', 20000, 'Number of steps.')

FLAGS = flags.FLAGS


def get_next_batch(batch_size=64):
    """Get a batch set of real images and random generated inputs."""
    if not os.path.exists(FLAGS.images_dir):
        os.makedirs(FLAGS.images_dir)
        #raise ValueError('images_dir is not exist.')
       
    images_path = os.path.join(FLAGS.images_dir, FLAGS.images_pattern)
    image_files_list = glob.glob(images_path)
    image_files_arr = np.array(image_files_list)
    selected_indices = np.random.choice(len(image_files_list), batch_size)
    selected_image_files = image_files_arr[selected_indices]
    images = read_images(selected_image_files)
    
#    generated_inputs = np.random.normal(size=[batch_size, 64])
    generated_inputs = np.random.uniform(
        low=-1, high=1.0, size=[batch_size, 64])
    return images, generated_inputs
    
    
def read_images(image_files):
    """Read images by OpenCV."""
    images = []
    for image_path in image_files:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = (image - 127.5) / 127.5
        images.append(image)
    return np.array(images)


def write_images(generated_images, images_save_dir, num_step):
    """Write images to a given directory."""
    #Scale images from [-1, 1] to [0, 255].
    generated_images = ((generated_images + 1) * 127.5).astype(np.uint8)
    for j, image in enumerate(generated_images):
        image_name = 'generated_step{}_{}.jpg'.format(num_step+1, j+1)
        image_path = os.path.join(FLAGS.generated_images_save_dir,
                                  image_name)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image)


def main(_):
    # Define placeholder
    real_data = tf.placeholder(
        tf.float32, shape=[None, 64, 64, 3], name='real_data')
    generated_inputs = tf.placeholder(
        tf.float32, [None, 64], name='generated_inputs')
    
    # Create DCGAN model
    dcgan_model = model.DCGAN(is_training=True, final_size=64)
    outputs_dict = dcgan_model.dcgan_model(real_data, generated_inputs)
    generated_data = outputs_dict['generated_data']
    generated_data_ = tf.identity(generated_data, name='generated_data')
    discriminator_gen_outputs = outputs_dict['discriminator_gen_outputs']
    discriminator_real_outputs = outputs_dict['discriminator_real_outputs']
    generator_variables = outputs_dict['generator_variables']
    discriminator_variables = outputs_dict['discriminator_variables']
    loss_dict = dcgan_model.loss(discriminator_real_outputs,
                                 discriminator_gen_outputs)
    discriminator_loss = loss_dict['dis_loss']
    discriminator_loss_on_real = loss_dict['dis_loss_on_real']
    discriminator_loss_on_generated = loss_dict['dis_loss_on_generated']
    generator_loss = loss_dict['gen_loss']

    # Write loss values to logdir (tensorboard)
    tf.summary.scalar('discriminator_loss', discriminator_loss)
    tf.summary.scalar('discriminator_loss_on_real', discriminator_loss_on_real)
    tf.summary.scalar('discriminator_loss_on_generated',
                      discriminator_loss_on_generated)
    tf.summary.scalar('generator_loss', generator_loss)
    merged_summary = tf.summary.merge_all(key=tf.GraphKeys.SUMMARIES)
    
    # Create optimizer
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=0.0004,  # 0.0005
                                                     beta1=0.5)
    discriminator_train_step = discriminator_optimizer.minimize(
        discriminator_loss, var_list=discriminator_variables)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=0.0001,
                                                 beta1=0.5)
    generator_train_step = generator_optimizer.minimize(
        generator_loss, var_list=generator_variables)
    
    saver = tf.train.Saver(var_list=tf.global_variables())
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        # Write model graph to tensorboard
        if not FLAGS.logdir:
            raise ValueError('logdir is not specified.')
        if not os.path.exists(FLAGS.logdir):
            os.makedirs(FLAGS.logdir)
        writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)
        
        fixed_images, fixed_generated_inputs = get_next_batch()
        
        for i in range(FLAGS.num_steps):
            if (i+1) % 500 == 0:
                batch_images = fixed_images
                batch_generated_inputs = fixed_generated_inputs
            else:
                batch_images, batch_generated_inputs = get_next_batch()
            train_dict = {real_data: batch_images,
                          generated_inputs: batch_generated_inputs}
                
            # Update discriminator network
            sess.run(discriminator_train_step, feed_dict=train_dict)
            
            # Update generator network five times
            sess.run(generator_train_step, feed_dict=train_dict)
            sess.run(generator_train_step, feed_dict=train_dict)
            sess.run(generator_train_step, feed_dict=train_dict)
            sess.run(generator_train_step, feed_dict=train_dict)
            sess.run(generator_train_step, feed_dict=train_dict)
            
            summary, generated_images = sess.run(
                [merged_summary, generated_data], feed_dict=train_dict)
            
            # Write loss values to tensorboard
            writer.add_summary(summary, i+1)
            
            if (i+1) % 500 == 0:
                # Save model
                model_save_path = os.path.join(FLAGS.logdir, 'model.ckpt')
                saver.save(sess, save_path=model_save_path, global_step=i+1)
                
                # Save generated images
                if not FLAGS.generated_images_save_dir:
                    FLAGS.generated_images_save_dir = './generated_images'
                if not os.path.exists(FLAGS.generated_images_save_dir):
                    os.makedirs(FLAGS.generated_images_save_dir)
                write_images(
                    generated_images, FLAGS.generated_images_save_dir, i)
            
        writer.close()
        
        
if __name__ == '__main__':
    tf.app.run()
