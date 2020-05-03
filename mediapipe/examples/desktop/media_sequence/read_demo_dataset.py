import tensorflow as tf
from mediapipe.examples.desktop.media_sequence.demo_dataset import DemoDataset
demo_data_path = '/tmp/demo_data/'
with tf.Graph().as_default():
   d = DemoDataset(demo_data_path)
   dataset = d.as_dataset('test')
   # implement additional processing and batching here
   dataset_output = dataset.make_one_shot_iterator().get_next()
   images = dataset_output['images']
   labels = dataset_output['labels']

   with tf.Session() as sess:
     images_, labels_ = sess.run([images, labels])
   print('The shape of images_ is %s' % str(images_.shape))
   print('The shape of labels_ is %s' % str(labels_.shape))
