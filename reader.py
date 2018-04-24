import tensorflow as tf
from os import listdir
from os.path import isfile, join


mean_pixel = [123.68, 116.779, 103.939] # ImageNet average from VGG ..

def preprocess(image, size):
    #initial image
    shape = tf.shape(image)
    size_t = tf.constant(size, tf.float64)
    height = tf.cast(shape[0], tf.float64)
    width = tf.cast(shape[1], tf.float64)


    #取height或者width中小的一个
    cond_op = tf.less(height, width)

    #size_t 为size的tf张量形式
    #将content改变成正方形
    new_height, new_width = tf.cond(
        cond_op,
        lambda: (size_t, (width * size_t) / height),
        lambda: ((height * size_t) / width, size_t))

    #用tf.image.resize_images函数改变image的size
    resized_image = tf.image.resize_images(
            image,
            [tf.to_int32(new_height), tf.to_int32(new_width)],
                method=tf.image.ResizeMethod.BICUBIC)   #双三次插值生成方形图片
    #将resized_image调整为size*size图片
    cropped = tf.image.resize_image_with_crop_or_pad(resized_image, size, size)
    return cropped


def get_image(path, size):
    #读取图片
    #按jpg模式或png模式解码图片
    png = path.lower().endswith('png')
    img_bytes = tf.read_file(path)
    print(path,size)
    if png:
        getedimage = tf.image.decode_png(img_bytes, channels=3)
    else:
        getedimage = tf.image.decode_jpeg(img_bytes, channels=3)
    print('True')
    #调用初始化图片，将content图片调整为size大小
    return preprocess(getedimage, size)

#训练模型喂入图片数据
def image(n, size, path, epochs=2, shuffle=True, crop=True):
    print(1)
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    print(2)
    if not shuffle:
        filenames = sorted(filenames)
    print(3)
    png = filenames[0].lower().endswith('png')

    print(4)
    filename_queue = tf.train.string_input_producer(filenames, shuffle=shuffle, num_epochs=epochs)
    print(5)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(filename_queue)
    image = tf.image.decode_png(img_bytes, channels=3) if png else tf.image.decode_jpeg(img_bytes, channels=3)


    processed_image = preprocess(image, size)
    return tf.train.batch([processed_image], n, dynamic_pad=True)
