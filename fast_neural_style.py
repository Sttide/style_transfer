import tensorflow as tf
import image_transfer_net
import reader
import time
import vgg
import os

#超参数
CONTENT_WEIGHT =  2e1
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e-5

#分离的图层
CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"]

#地址
style_path = "./style.jpg"
content_path = "./content.jpg"
model_path = "models"
TRAIN_IMAGES_PATH = "train2014"

#图片名称
STYLE_IMAGES = "style.jpg"
CONTENT_IMAGE = "content.jpg"

#生成图片的大小
IMAGE_SIZE = 512

#每次训练图片的数量
BATCH_SIZE  = 4


#全变差正则项
def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.stack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.stack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

#gram矩阵
def gram(layer):
    shape = tf.shape(layer)
    num_images = shape[0]
    width = shape[1]
    height = shape[2]
    num_filters = shape[3]
    filters = tf.reshape(layer, tf.stack([num_images, -1, num_filters]))
    grams = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(width * height * num_filters)

    return grams

def get_style_features(style_paths, style_layers):
    with tf.Graph().as_default() as g:
        size = int(round(IMAGE_SIZE))
        images = tf.stack([reader.get_image(path, size) for path in style_paths])

        net, _ = vgg.net(images - reader.mean_pixel)
        features = []
        for layer in style_layers:
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

def main(argv=None):

    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    style_paths = STYLE_IMAGES.split(',')

    style_features_t = get_style_features(style_paths, STYLE_LAYERS)

    images = tf.expand_dims(reader.get_image(content_path, 256), 0)
    generated = image_transfer_net.net(images - reader.mean_pixel, training=True)

    #将生成图片和一次训练的图片一起通过VGG以提高效率
    net, _ = vgg.net(tf.concat([generated, images],0) - reader.mean_pixel)


    content_loss = 0
    for layer in CONTENT_LAYERS:
        generated_images, content_images = tf.split(net[layer],2,axis=0)
        size = tf.size(generated_images)
        shape = tf.shape(generated_images)
        width = shape[1]
        height = shape[2]
        num_filters = shape[3]
        content_loss += tf.nn.l2_loss(generated_images - content_images) / tf.to_float(size)
    content_loss = content_loss

    style_loss = 0
    for style_grams, layer in zip(style_features_t, STYLE_LAYERS):
        generated_images, _ = tf.split(net[layer],2,axis = 0)
        size = tf.size(generated_images)
        for style_gram in style_grams:
            style_loss += tf.nn.l2_loss(gram(generated_images) - style_gram) / tf.to_float(size)
    style_loss = style_loss / len(style_paths)

    tv_loss = total_variation_loss(generated)

    loss = STYLE_WEIGHT * style_loss + CONTENT_WEIGHT * content_loss + TV_WEIGHT * tv_loss

    global_step = tf.Variable(0, name="global_step", trainable=False)
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)
    output_image = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(generated) + reader.mean_pixel, tf.uint8))

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        file = tf.train.latest_checkpoint(model_path)
        sess.run([tf.initialize_all_variables(), tf.initialize_local_variables()])

        if file:
            print('Restoring model from {}'.format(file))
            saver.restore(sess, file)
        #多线程
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        try:
            start_time = time.time()
            this_time = start_time
            while not coord.should_stop():
                _, loss_t, step = sess.run([train_op, loss, global_step])
                if step % 100 == 0:
                    elapsed = time.time() - this_time
                    total_time = time.time() - start_time
                    this_time = time.time()
                    print("step:",step," total_loss:",loss_t," this time:",elapsed," total time:",total_time)
                if step % 10000 == 0:
                    saver.save(sess, model_path + '/fast-style-tresfer', global_step=step)
            with open('out.jpg', 'wb') as f:
                    f.write(output_image)
        except tf.errors.OutOfRangeError:
            saver.save(sess, model_path + '/fast-style-tresfer-done')
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
