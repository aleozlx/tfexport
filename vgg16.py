from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.keras.backend.clear_session()

with tf.variable_scope('DataSource'):
    images = tf.placeholder(tf.uint8, (1,256,256,3))
    images = tf.image.per_image_standardization(images)

with tf.variable_scope('DCNN'):
    dcnn = tf.keras.applications.VGG16(
        input_tensor=images,
        include_top=False,
        weights=None,
    )

OUTPUT = Path('/tmp')
FNAME = 'vgg16'

with tf.Session() as sess:
    tf.keras.backend.set_session(sess)
    with tf.variable_scope('Initializer'):
        dcnn.load_weights('/tank/datasets/research/model_weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
    tf.io.write_graph(sess.graph_def, str(OUTPUT), f'{FNAME}.pbtxt', as_text=True)
    tf.io.write_graph(sess.graph_def, str(OUTPUT), f'{FNAME}.pb', as_text=False)
    saver = tf.train.Saver(tf.trainable_variables())
    saver.save(sess, str(OUTPUT / FNAME))

    outgraph_def = graph_util.convert_variables_to_constants(
        sess, sess.graph_def,
        'DCNN/block5_pool/MaxPool'.replace(" ", "").split(","),
        # variable_names_whitelist=variable_names_whitelist,
        # variable_names_blacklist=variable_names_blacklist
    )
    tf.io.write_graph(outgraph_def, str(OUTPUT), f'{FNAME}.frozen.pb', as_text=False)
