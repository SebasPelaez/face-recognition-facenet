import tensorflow as tf

import triplet_loss
import utils

class InceptionBlock(tf.keras.models.Model):
  def __init__(self,conv_filters,strides, **kwargs):
    super(InceptionBlock, self).__init__(**kwargs)
    
    conv2_1x1__filter = conv_filters[0]
    conv4_3x3_filter = conv_filters[1]
    conv3_1x1__filter = conv_filters[2]
    conv5_5x5_filter = conv_filters[3]
    conv6_1x1_filter = conv_filters[4]
                                                
    self.conv2_1x1 = tf.keras.layers.Conv2D(filters=conv2_1x1__filter,kernel_size=1,strides=1,padding='same')
    self.activation_conv2_1x1 = tf.keras.layers.Activation('relu')
    
    self.conv3_1x1 = tf.keras.layers.Conv2D(filters=conv3_1x1__filter,kernel_size=1,strides=1,padding='same')
    self.activation_conv3_1x1 = tf.keras.layers.Activation('relu')
    
    self.pool_3x3 = tf.keras.layers.MaxPool2D(pool_size=3,strides=strides,padding='same')
                                                
    self.conv4_3x3 = tf.keras.layers.Conv2D(filters=conv4_3x3_filter,kernel_size=3,strides=strides,padding='same')
    self.activation_conv4_3x3 = tf.keras.layers.Activation('relu')
    
    self.conv5_5x5 = tf.keras.layers.Conv2D(filters=conv5_5x5_filter,kernel_size=5,strides=strides,padding='same')
    self.activation_conv5_5x5 = tf.keras.layers.Activation('relu')
    
    self.conv6_1x1 = tf.keras.layers.Conv2D(filters=conv6_1x1_filter,kernel_size=1,strides=1,padding='same')
    self.activation_conv6_1x1 = tf.keras.layers.Activation('relu')

  def call(self, inputs, training=None):
        
    first_block = self.conv2_1x1(inputs)
    first_block = self.activation_conv2_1x1(first_block)
    
    second_block = self.conv3_1x1(inputs)
    second_block = self.activation_conv3_1x1(second_block)
    
    third_block = self.pool_3x3(inputs)
    
    first_block = self.conv4_3x3(first_block)
    first_block = self.activation_conv4_3x3(first_block)
    
    second_block = self.conv5_5x5(second_block)
    second_block = self.activation_conv5_5x5(second_block)
    
    third_block = self.conv6_1x1(third_block)
    third_block = self.activation_conv6_1x1(third_block)
    
    x = tf.concat(
            values=[first_block,second_block,third_block],
            axis=3
        )
    
    return x

class InceptionBlockWithoutDownSample(tf.keras.models.Model):
  def __init__(self,conv_filters, **kwargs):
    super(InceptionBlockWithoutDownSample, self).__init__(**kwargs)
    
    conv1_1x1_filter = conv_filters[0]
    
    self.conv1_1x1 = tf.keras.layers.Conv2D(filters=conv1_1x1_filter,kernel_size=1,strides=1,padding='same')
    self.activation = tf.keras.layers.Activation('relu')
    
    self.inception_block = InceptionBlock(conv_filters[1:],strides=1)

  def call(self, inputs, training=None):
        
    alone_conv = self.conv1_1x1(inputs)
    alone_conv = self.activation(alone_conv)
    
    inception_module = self.inception_block(inputs)
    
    x = tf.concat(
            values=[alone_conv,inception_module],
            axis=3
        )
    
    return x

class FaceNet_Architecture(tf.keras.models.Model):
  def __init__(self, **kwargs):
    super(FaceNet_Architecture, self).__init__(**kwargs)                                     
        
  def build(self, input_shape):
        
    input_channels = int(input_shape[-1])
    
    self.conv1 = tf.keras.layers.Conv2D(filters=64,kernel_size=7,strides=2,padding='same')
    self.bn1 = tf.keras.layers.BatchNormalization()
    self.activation1 = tf.keras.layers.Activation('relu')
    self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
    
    self.conv2 = tf.keras.layers.Conv2D(filters=192,kernel_size=3,strides=1,padding='same')
    self.bn2 = tf.keras.layers.BatchNormalization()
    self.activation2 = tf.keras.layers.Activation('relu')
    self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(3,3),strides=2,padding='same')
    
    self.inception_3a = InceptionBlockWithoutDownSample(conv_filters=[64,96,128,16,32,32])
    self.inception_3b = InceptionBlockWithoutDownSample(conv_filters=[64,96,128,32,64,64])
    self.inception_3c = InceptionBlock(conv_filters=[128,256,32,64,input_channels],strides=2)
    
    self.inception_4a = InceptionBlockWithoutDownSample(conv_filters=[256,96,192,32,64,128])
    self.inception_4b = InceptionBlockWithoutDownSample(conv_filters=[224,112,224,32,64,128])
    self.inception_4c = InceptionBlockWithoutDownSample(conv_filters=[192,128,256,32,64,128])
    self.inception_4d = InceptionBlockWithoutDownSample(conv_filters=[160,144,288,32,64,128])
    self.inception_4e = InceptionBlock(conv_filters=[160,256,64,128,input_channels],strides=2)
    
    self.inception_5a = InceptionBlockWithoutDownSample(conv_filters=[384,192,384,48,128,128])
    self.inception_5b = InceptionBlockWithoutDownSample(conv_filters=[384,192,384,48,128,128])
    
    self.average_pool = tf.keras.layers.GlobalAveragePooling2D()
    
    self.fc = tf.keras.layers.Dense(units = 128)
    self.activation_fc = tf.keras.layers.Activation('relu')

  def call(self, inputs, training=None):
        
    first_block = self.conv1(inputs)
    first_block = self.bn1(first_block, training=training)
    first_block = self.activation1(first_block)
    first_block = self.pool1(first_block)
    
    second_block = self.conv2(first_block)
    second_block = self.bn2(second_block, training=training)
    second_block = self.activation2(second_block)
    second_block = self.pool2(second_block)
    
    first_inception_block = self.inception_3a(second_block)
    first_inception_block = self.inception_3b(first_inception_block)
    first_inception_block = self.inception_3c(first_inception_block)
    
    second_inception_block = self.inception_4a(first_inception_block)
    second_inception_block = self.inception_4b(second_inception_block)
    second_inception_block = self.inception_4c(second_inception_block)
    second_inception_block = self.inception_4d(second_inception_block)
    second_inception_block = self.inception_4e(second_inception_block)
    
    third_inception_block = self.inception_5a(second_inception_block)
    third_inception_block = self.inception_5b(third_inception_block)
    
    global_average_pooling = self.average_pool(third_inception_block)
    
    fully_connected = self.fc(global_average_pooling)
    fully_connected = self.activation_fc(fully_connected)
    
    output = tf.nn.l2_normalize(fully_connected)
    
    return output

def preprocess_image(x):
  return tf.cast(x, tf.float32) / 255

def denormalize_image(x):
  return tf.cast(x * 255, tf.uint8)

def model_fn(features, labels, mode, params):

  training = mode == tf.estimator.ModeKeys.TRAIN

  preprocessed_images = preprocess_image(features['image'])
  batch_label = labels

  images = tf.reshape(features['image'], [-1, 250, 250, 1])
    
  model = FaceNet_Architecture()
  embeddings = model(preprocessed_images, training=training)

  predictions = {
    'embeddings': embeddings
  }
    
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  loss, fraction = triplet_loss.batch_all_triplet_loss(batch_label, embeddings, margin=0.5,squared=False)

  pairwise_dist = triplet_loss._pairwise_distances(embeddings, squared=False)

  validation_rate = VAL_metric(pairwise_dist, batch_label, params, 'VAL_metric')
  false_accept_rate = FAR_metric(pairwise_dist, batch_label, params, 'FAR_metric')

  eval_metric_ops = {"fraction_positive_triplets": tf.metrics.mean(fraction)}
  eval_metric_ops['validation_rate'] = validation_rate
  eval_metric_ops['false_accept_rate'] = false_accept_rate

  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)

  tf.summary.scalar('fraction_positive_triplets', fraction)
  tf.summary.scalar('validation_rate', validation_rate[0][0])
  tf.summary.scalar('false_accept_rate', false_accept_rate[0][0])
  tf.summary.image('train_image', images, max_outputs=1)  

  if mode == tf.estimator.ModeKeys.TRAIN:
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(
        learning_rate=params['learning_rate']
      )
      
      train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step()
      )

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

def VAL_metric(pairwise_dist,labels,params,name):

  acumulative_val = tf.get_variable(
    name=name,
    shape=(1),
    initializer=tf.zeros_initializer(),
    dtype=tf.float32)

  mask_anchor_positive = triplet_loss._get_anchor_positive_triplet_mask(labels)
  mask_anchor_positive = tf.to_float(mask_anchor_positive)

  validation_rate = compute_metrics_rate(pairwise_dist,mask_anchor_positive,params)
  
  new_validation_rate = acumulative_val + validation_rate

  update_op = acumulative_val.assign(new_validation_rate)

  return new_validation_rate, update_op
  
def FAR_metric(pairwise_dist, labels, params, name):

  acumulative_far = tf.get_variable(
    name=name,
    shape=(1),
    initializer=tf.zeros_initializer(),
    dtype=tf.float32)
  
  mask_anchor_negative = triplet_loss._get_anchor_negative_triplet_mask(labels)
  mask_anchor_negative = tf.to_float(mask_anchor_negative)

  false_accept_rate = compute_metrics_rate(pairwise_dist,mask_anchor_negative,params)

  new_false_accept_rate = acumulative_far + false_accept_rate

  update_op = acumulative_far.assign(new_false_accept_rate)

  return new_false_accept_rate, update_op

def compute_metrics_rate(pairwise_dist, mask_anchor, params):

  anchor_dist = tf.multiply(mask_anchor, pairwise_dist)

  zeros = tf.to_float(tf.math.equal(anchor_dist,0))
  zeros = tf.reduce_sum(zeros)

  valid_tuples = tf.to_float(tf.greater(anchor_dist, 1e-16))
  num_tuples = tf.reduce_sum(valid_tuples) / 2

  accepted_values = tf.to_float(tf.less(anchor_dist, params['distance_threshold']))
  accepted_values = (tf.reduce_sum(accepted_values) - zeros) / 2

  accepted_rate = accepted_values / num_tuples

  return accepted_rate

if __name__ == '__main__':
  inputs = tf.keras.layers.Input(shape=(250,250, 3))
  model = FaceNet_Architecture()
  x = model(inputs, training=False)
  model.summary()