
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Implementation of DCGAN.

This work was first described in:
    Unsupervised representation learning with deep convolutional generative 
    adversarial networks, Alec Radford et al., arXiv: 1511.06434v2
   
This module Based on:
    TensorFlow models/research/slim/nets/dcgan.py
    TensorFlow tensorflow/contrib/gan
"""

import math
import tensorflow as tf

slim = tf.contrib.slim


class DCGAN(object):
    """Implementation of DCGAN."""
    
    def __init__(self, 
                 is_training,
                 generator_depth=64,
                 discriminator_depth=64,
                 final_size=32,
                 num_outputs=3,
                 fused_batch_norm=False):
        """Constructor.
        
        Args:
            is_training: Whether the the network is for training or not.
            generator_depth: Number of channels in last deconvolution layer of
                the generator network.
            discriminator_depth: Number of channels in first convolution layer
                of the discirminator network.
            final_size: The shape of the final output.
            num_outputs: Nuber of output features. For images, this is the
                number of channels.
            fused_batch_norm: If 'True', use a faster, fused implementation
                of batch normalization.
        """
        self._is_training = is_training
        self._generator_depth = generator_depth
        self._discirminator_depth = discriminator_depth
        self._final_size = final_size
        self._num_outputs = num_outputs
        self._fused_batch_norm = fused_batch_norm
        
    def _validate_image_inputs(self, inputs):
        """Check the inputs whether is valid or not.
        
        Copy from:
            https://github.com/tensorflow/models/blob/master/research/
            slim/nets/dcgan.py
            
        Args:
            inputs: A float32 tensor with shape [batch_size, height, width, 
                channels].
            
        Raises:
            ValueError: If the input image shape is not 4-dimensional, if the 
                spatial dimensions aren't defined at graph construction time, 
                if the spatial dimensions aren't square, or if the spatial 
                dimensions aren't a power of two.
        """
        inputs.get_shape().assert_has_rank(4)
        inputs.get_shape()[1:3].assert_is_fully_defined()
        if inputs.get_shape()[1] != inputs.get_shape()[2]:
            raise ValueError('Input tensor does not have equal width and '
                             'height: ', inputs.get_shape()[1:3])
        width = inputs.get_shape().as_list()[2]
        if math.log(width, 2) != int(math.log(width, 2)):
            raise ValueError("Input tensor 'width' is not a power of 2: ",
                             width)
            
    def discriminator(self, 
                      inputs,
                      depth=64,
                      is_training=True,
                      reuse=None,
                      scope='Discriminator',
                      fused_batch_norm=False):
        """Discriminator network for DCGAN.
        
        Construct discriminator network from inputs to the final endpoint.
        
        Copy from:
            https://github.com/tensorflow/models/blob/master/research/
            slim/nets/dcgan.py
        
        Args:
            inputs: A float32 tensor with shape [batch_size, height, width, 
                channels].
            depth: Number of channels in first convolution layer.
            is_training: Whether the network is for training or not.
            reuse: Whether or not the network variables should be reused.
                'scope' must be given to be reused.
            scope: Optional variable_scope. Default value is 'Discriminator'.
            fused_batch_norm: If 'True', use a faster, fused implementation
                of batch normalization.
                
        Returns:
            logits: The pre-softmax activations, a float32 tensor with shape
                [batch_size, 1].
            end_points: A dictionary from components of the network to their
                activation.
                
        Raises:
            ValueError: If the input image shape is not 4-dimensional, if the 
                spatial dimensions aren't defined at graph construction time, 
                if the spatial dimensions aren't square, or if the spatial 
                dimensions aren't a power of two.
        """
        normalizer_fn = slim.batch_norm
        normalizer_fn_args = {
            'is_training': is_training,
            'zero_debias_moving_mean': True,
            'fused': fused_batch_norm}
        
        self._validate_image_inputs(inputs)
        height = inputs.get_shape().as_list()[1]
        
        end_points = {}
        with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
            with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
                with slim.arg_scope([slim.conv2d], stride=2, kernel_size=4,
                                    activation_fn=tf.nn.leaky_relu):
                    net = inputs
                    for i in range(int(math.log(height, 2))):
                        scope = 'conv%i' % (i+1)
                        current_depth = depth * 2**i
                        normalizer_fn_ = None if i == 0 else normalizer_fn
                        net = slim.conv2d(net, num_outputs=current_depth, 
                                          normalizer_fn=normalizer_fn_,
                                          scope=scope)
                        end_points[scope] = net
                    
                    logits = slim.conv2d(net, 1, kernel_size=1, stride=1,
                                         padding='VALID', normalizer_fn=None,
                                         activation_fn=None)
                    logits = tf.reshape(logits, [-1, 1])
                    end_points['logits'] = logits
                    
                    return logits, end_points
                
    def generator(self,
                  inputs,
                  depth=64,
                  final_size=32,
                  num_outputs=3,
                  is_training=True,
                  reuse=None,
                  scope='Generator',
                  fused_batch_norm=False):
        """Generator network for DCGAN.
        
        Construct generator network from inputs to the final endpoint.
        
        Copy from:
            https://github.com/tensorflow/models/blob/master/research/
            slim/nets/dcgan.py
        
        Args:
            inputs: A float32 tensor with shape [batch_size, N] for any size N.
            depth: Number of channels in last deconvolution layer.
            final_size: The shape of the final output.
            num_outputs: Nuber of output features. For images, this is the
                number of channels.
            is_training: Whether is training or not.
            reuse: Whether or not the network has its variables should be 
                reused. 'scope' must be given to be reused.
            scope: Optional variable_scope. Default value is 'Generator'.
            fused_batch_norm: If 'True', use a faster, fused implementation
                of batch normalization.
                
        Returns:
            logits: The pre-sortmax activations, a float32 tensor with shape
                [batch_size, final_size, final_size, num_outputs].
            end_points: A dictionary from components of the network to their
                activation.
            
        Raises:
            ValueError: If 'inputs' is not 2-dimensional, or if 'final_size'
                is not a power of 2 or is less than 8.
        """
        normalizer_fn = slim.batch_norm
        normalizer_fn_args = {
            'is_training': is_training,
            'zero_debias_moving_mean': True,
            'fused': fused_batch_norm}
        
        inputs.get_shape().assert_has_rank(2)
        if math.log(final_size, 2) != int(math.log(final_size, 2)):
            raise ValueError("'final_size' (%i) must be a power of 2."
                             % final_size)
        if final_size < 8:
            raise ValueError("'final_size' (%i) must be greater than 8."
                             % final_size)
            
        end_points = {}
        num_layers = int(math.log(final_size, 2)) - 1
        with tf.variable_scope(scope, values=[inputs], reuse=reuse) as scope:
            with slim.arg_scope([normalizer_fn], **normalizer_fn_args):
                with slim.arg_scope([slim.conv2d_transpose],
                                    normalizer_fn=normalizer_fn,
                                    stride=2, kernel_size=4):
                    net = tf.expand_dims(tf.expand_dims(inputs, 1), 1)
                    
                    # First upscaling is different because it takes the input
                    # vector.
                    current_depth = depth * 2 ** (num_layers - 1)
                    scope = 'deconv1'
                    net = slim.conv2d_transpose(net, current_depth, stride=1, 
                                                padding='VALID', scope=scope)
                    end_points[scope] = net
                    
                    for i in range(2, num_layers):
                        scope = 'deconv%i' % i
                        current_depth = depth * 2 * (num_layers - i)
                        net = slim.conv2d_transpose(net, current_depth, 
                                                    scope=scope)
                        end_points[scope] = net
                        
                    # Last layer has different normalizer and activation.
                    scope = 'deconv%i' % num_layers
                    net = slim.conv2d_transpose(net, depth, normalizer_fn=None,
                                                activation_fn=None, scope=scope)
                    end_points[scope] = net
                    
                    # Convert to proper channels
                    scope = 'logits'
                    logits = slim.conv2d(
                        net,
                        num_outputs,
                        normalizer_fn=None,
                        activation_fn=tf.nn.tanh,
                        kernel_size=1,
                        stride=1,
                        padding='VALID',
                        scope=scope)
                    end_points[scope] = logits
                    
                    logits.get_shape().assert_has_rank(4)
                    logits.get_shape().assert_is_compatible_with(
                        [None, final_size, final_size, num_outputs])
                    
                    return logits, end_points
                
    def dcgan_model(self, 
                      real_data, 
                      generator_inputs,
                      generator_scope='Generator',
                      discirminator_scope='Discriminator',
                      check_shapes=True):
        """Returns DCGAN model outputs and variables.
        
        Modified from:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
            contrib/gan/python/train.py
            
        Args:
            real_data: A float32 tensor with shape [batch_size, height, width, 
                channels].
            generator_inputs: A float32 tensor with shape [batch_size, N] for 
                any size N.
            generator_scope: Optional genertor variable scope. Useful if you
                want to reuse a subgraph that has already been created.
            discriminator_scope: Optional discriminator variable scope. Useful
                if you want to reuse a subgraph that has already been created.
            check_shapes: If 'True', check that generator produces Tensors
                that are the same shape as real data. Otherwise, skip this
                check.
                
        Returns:
            A dictionary containing output tensors.
            
        Raises:
            ValueError: If the generator outputs a tensor that isn't the same
                shape as 'real_data'.
        """
        # Create models
        with tf.variable_scope(generator_scope) as gen_scope:
            generated_data, _ = self.generator(
                generator_inputs, self._generator_depth, self._final_size,
                self._num_outputs, self._is_training)
        with tf.variable_scope(discirminator_scope) as dis_scope:
            discriminator_gen_outputs, _ = self.discriminator(
                generated_data, self._discirminator_depth, self._is_training)
        with tf.variable_scope(dis_scope, reuse=True):
            discriminator_real_outputs, _ = self.discriminator(
                real_data, self._discirminator_depth, self._is_training)
        
        if check_shapes:
            if not generated_data.shape.is_compatible_with(real_data.shape):
                raise ValueError('Generator output shape (%s) must be the '
                                 'shape as real data (%s).'
                                 % (generated_data.shape, real_data.shape))
                
        # Get model-specific variables
        generator_variables = slim.get_trainable_variables(gen_scope)
        discriminator_variables = slim.get_trainable_variables(dis_scope)
        
        return {'generated_data': generated_data,
                'discriminator_gen_outputs': discriminator_gen_outputs,
                'discriminator_real_outputs': discriminator_real_outputs,
                'generator_variables': generator_variables,
                'discriminator_variables': discriminator_variables}
        
    def predict(self, generator_inputs):
        """Return the generated results by generator network.
        
        Args:
            generator_inputs: A float32 tensor with shape [batch_size, N] for 
                any size N.
                
        Returns:
            logits: The pre-sortmax activations, a float32 tensor with shape
                [batch_size, final_size, final_size, num_outputs].
        """
        logits, _ = self.generator(generator_inputs, self._generator_depth,
                                   self._final_size, self._num_outputs,
                                   is_training=False)
        return logits
        
    def discriminator_loss(self, 
                           discriminator_real_outputs,
                           discriminator_gen_outputs,
                           label_smoothing=0.25):
        """Original minmax discriminator loss for GANs, with label smoothing.
        
        Modified from:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
            contrib/gan/python/losses/python/losses_impl.py
        
        Args:
            discriminator_real_outputs: Discriminator output on real data.
            discriminator_gen_outputs: Discriminator output on generated data.
                Expected to be in the range of (-inf, inf).
            label_smoothing: The amount of smoothing for positive labels. This
                technique is taken from `Improved Techniques for Training GANs`
                (https://arxiv.org/abs/1606.03498). `0.0` means no smoothing.
                
        Returns:
            loss_dict: A dictionary containing three scalar tensors.
        """
        # -log((1 - label_smoothing) - sigmoid(D(x)))
        losses_on_real = slim.losses.sigmoid_cross_entropy(
            logits=discriminator_real_outputs,
            multi_class_labels=tf.ones_like(discriminator_real_outputs),
            label_smoothing=label_smoothing)
        loss_on_real = tf.reduce_mean(losses_on_real)
        # -log(- sigmoid(D(G(x))))
        losses_on_generated = slim.losses.sigmoid_cross_entropy(
            logits=discriminator_gen_outputs,
            multi_class_labels=tf.zeros_like(discriminator_gen_outputs))
        loss_on_generated = tf.reduce_mean(losses_on_generated)
        
        loss = loss_on_real + loss_on_generated
        return {'dis_loss': loss,
                'dis_loss_on_real': loss_on_real,
                'dis_loss_on_generated': loss_on_generated}
        
    def generator_loss(self, discriminator_gen_outputs, label_smoothing=0.0):
        """Modified generator loss for DCGAN.
        
        Modified from:
            https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
            contrib/gan/python/losses/python/losses_impl.py
        
        Args:
            discriminator_gen_outputs: Discriminator output on generated data.
                Expected to be in the range of (-inf, inf).
                
        Returns:
            loss: A scalar tensor.
        """
        losses = slim.losses.sigmoid_cross_entropy(
            logits=discriminator_gen_outputs, 
            multi_class_labels=tf.ones_like(discriminator_gen_outputs),
            label_smoothing=label_smoothing)
        loss = tf.reduce_mean(losses)
        return loss
    
    def loss(self, discriminator_real_outputs, discriminator_gen_outputs):
        """Computes the loss of DCGAN.
        
        Args:
            discriminator_real_outputs: Discriminator output on real data.
            discriminator_gen_outputs: Discriminator output on generated data.
                Expected to be in the range of (-inf, inf).
                
        Returns:
            A dictionary contraining 4 scalar tensors.
        """
        dis_loss_dict = self.discriminator_loss(discriminator_real_outputs,
                                                discriminator_gen_outputs)
        gen_loss = self.generator_loss(discriminator_gen_outputs)
        dis_loss_dict.update({'gen_loss': gen_loss})
        return dis_loss_dict
