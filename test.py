import tensorflow as tf

def _scatter_nd(enc_batch, batch_size, batch_nums, extended_vsize, copy_dist):
    shape = [batch_size, extended_vsize]
    copy_prob = tf.reshape(tf.zeros(shape, dtype=tf.float32), [-1])
    linear_indices = tf.reshape(batch_nums, [-1]) * extended_vsize  +  tf.reshape(enc_batch, [-1])
    flat_w_copy = tf.reshape(copy_dist, [-1])
    unchanged_indices = tf.range(tf.size(copy_prob))
    flat_vocab_prob = tf.dynamic_stitch([unchanged_indices, linear_indices], [copy_prob, flat_w_copy])
    vocab_copy = tf.reshape(flat_vocab_prob, [batch_size, extended_vsize])
    return vocab_copy




def gather_2d(params, indices1, indices2):
    # only for two dim now
    shape0, shape1 = tf.shape(params)[0], tf.shape(params)[1]
    flat = tf.reshape(params, [-1])
    flat_idx = indices1 * shape1 + indices2
    return tf.gather(flat, flat_idx)

