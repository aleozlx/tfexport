from pathlib import Path
import tensorflow as tf
from tensorflow.python.framework import graph_util
tf.keras.backend.clear_session()

BUILD = '/tank/builds/spfreq2'
OUTPUT = Path('/tank/datasets/research/model_weights')
FNAME = 'vgg16sp'

with tf.variable_scope('DataSource'):
    images = tf.placeholder(tf.uint8, (1,256,256,3), name="input_image")
    images = tf.image.per_image_standardization(images)
    superpixels = tf.placeholder(tf.int32, (1,256,256), name="input_superpixels")

with tf.variable_scope('DCNN'):
    dcnn = tf.keras.applications.VGG16(
        input_tensor=images,
        include_top=False,
        weights=None,
    )

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
spfreq_mod = tf.load_op_library(f"{BUILD}/spfreq2_op.so")
spfreqOp = spfreq_mod.superpixel_freq

def superpixel_pool(inputs):
    superpixels, convmaps = inputs
    superpixel_frequency = spfreqOp(superpixels, output_shape = (300, 8, 8))
    sp_shape = tf.shape(superpixel_frequency)
    superpixel_normalizer = tf.reduce_sum(superpixel_frequency, axis=[2,3], keepdims=True)
    superpixel_frequency = tf.div_no_nan(superpixel_frequency, tf.broadcast_to(superpixel_normalizer, sp_shape))
    cm_shape = tf.shape(convmaps)
    return tf.matmul(
        tf.reshape(superpixel_frequency, (sp_shape[0], sp_shape[1], -1)),
        tf.reshape(convmaps, (cm_shape[0], -1, cm_shape[-1])))

with tf.variable_scope('Superpixels'):
    convmaps = dcnn.layers[-1].output
    feat_sp = superpixel_pool((superpixels, convmaps))

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

        # 'DCNN/block5_pool/MaxPool'.replace(" ", "").split(","),
        'Superpixels/MatMul'.replace(" ", "").split(","),

        # variable_names_whitelist=variable_names_whitelist,
        # variable_names_blacklist=variable_names_blacklist
    )
    tf.io.write_graph(outgraph_def, str(OUTPUT), f'{FNAME}.frozen.pb', as_text=False)

    