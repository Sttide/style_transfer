import reader
import tensorflow as tf
import image_transfer_net

tf.app.flags.DEFINE_string("model_path", "models", "Path to read/write trained models")
tf.app.flags.DEFINE_string("model_name", "", "For finding the model")
tf.app.flags.DEFINE_string("image", "", "Path to image to trainsform")

FLAGS = tf.app.flags.FLAGS

model_path = FLAGS.model_path + '/' + FLAGS.model_name

def main(argv=None):
    with open(FLAGS.image, 'rb') as f:
        jpg = f.read()

    image = tf.image.convert_image_dtype(tf.image.decode_jpeg(jpg, channels=3), tf.float32) * 255.
    images = tf.expand_dims(image, 0)

    generated_images = image_transfer_net.net(images - reader.mean_pixel, training=False)

    output_image = tf.cast(generated_images, tf.uint8)
    jpegs = tf.map_fn(lambda image: tf.image.encode_jpeg(image), output_image, dtype=tf.string)

    with tf.Session() as sess:
        file = tf.train.latest_checkpoint(model_path)
        if not file:
            print('Could not find trained model in %s' % model_path)
            return
        print('Using model from %s' % file)
        saver = tf.train.Saver()
        saver.restore(sess, file)

        res = sess.run(jpegs)
        with open('res.jpg', 'wb') as f:
            f.write(res[0])

if __name__ == '__main__':
    tf.app.run()
