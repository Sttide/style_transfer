import tensorflow as tf
from PIL import Image
import reader
import time
import vgg



#超参数
CONTENT_WEIGHT = 2e1
STYLE_WEIGHT = 1e2
TV_WEIGHT = 1e-5
#分离的图层
CONTENT_LAYERS = ["relu4_2"]
STYLE_LAYERS = ["relu1_1","relu2_1","relu3_1","relu4_1","relu5_1"]
#学习率
LEARNING_RATE = 10
#生成图片的大小
IMAGE_SIZE = 512
#地址
style_path = "./your_name.jpg"
content_path = "./content.jpg"
#迭代次数
step_num = 5


def total_variation_loss(layer):
    #传进来的参数是initial 输出图片
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.stack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.stack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

# Gram matrix
def gram(layer):
    #layer 依次为 "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"的运算结果
    shape = tf.shape(layer)
    #当前层的shape
    #relu1_1 , shape=(1, 512, 512, 64)
    num_filters = shape[3]
    #shape[3] num_filters 当前层卷积核的数量
    size = tf.size(layer)
    #size为layer的所有结果的个数H*W*C，矩阵一共有多少个元素
    filters = tf.reshape(layer, tf.stack([-1, num_filters]))
    #将Ml转换成一维向量
    gram = tf.matmul(filters, filters, transpose_a=True) / tf.to_float(size)
    #得到当前层gram矩阵
    return gram


def get_style_features():
    with tf.Graph().as_default() as g:
        size = int(round(IMAGE_SIZE))
        images = tf.stack([reader.get_image(style_path, size)])
        net, _ = vgg.net(images)
        features = []
        for layer in STYLE_LAYERS:
            # net字典形式，net{"conv1_1":"1_1的结果“}
            # layer 依次 = "relu1_1,relu2_1,relu3_1,relu4_1,relu5_1"
            #返回每一层的gram矩阵
            features.append(gram(net[layer]))

        with tf.Session() as sess:
            return sess.run(features)

def get_content_features(content_layers):
    with tf.Graph().as_default() as g:
        #tf.expand_dims 在位置0扩展了一个维度[2,3]=>[1,2,3]变成图片
        image = tf.expand_dims(reader.get_image(content_path, IMAGE_SIZE), 0)
        net, _ = vgg.net(image)
        layers = []
        for layer in content_layers:
            # net字典形式，net{"conv1_1":"conv1_1的结果“}
            layers.append(net[layer])

        with tf.Session() as sess:
            #返回features + 图片
            return sess.run(layers + [image])

def activate():

    im = Image.open(style_path)
    print(im)

    style_features_t = get_style_features()
    #res为features + 图片
    res = get_content_features(CONTENT_LAYERS)
    content_features_t, image_t = res[:-1], res[-1]

    #随机生成一张正态分布的空白噪声图片
    random = tf.random_normal(image_t.shape)
    initial = tf.Variable(random)

    #对output图片作处理，并返回每一层的处理结果
    net, _ = vgg.net(initial)

    # net字典形式，net{"conv1_1":"1_1的结果“},输出图片每次的计算结果
    content_loss = 0
    #遍历每一层的内容 relu4_2 一一对应
    for content_features, layer in zip(content_features_t, CONTENT_LAYERS):
        layer_size = tf.size(content_features)
        content_loss += tf.nn.l2_loss(net[layer] - content_features) / tf.to_float(layer_size)
    #可以从多个层提取content，这里只使用一层
    content_loss = CONTENT_WEIGHT * content_loss  / len(CONTENT_LAYERS)

    style_loss = 0
    for style_gram, layer in zip(style_features_t, STYLE_LAYERS):
        layer_size = tf.size(style_gram)
        style_loss += tf.nn.l2_loss(gram(net[layer]) - style_gram) / tf.to_float(layer_size)
    style_loss = STYLE_WEIGHT * style_loss

    #total_variation_loss
    tv_loss = TV_WEIGHT * total_variation_loss(initial)
    total_loss = content_loss + style_loss + tv_loss

    #训练方法
    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(total_loss)
    #生成图片
    output_image = tf.image.encode_jpeg(tf.saturate_cast(tf.squeeze(initial) + reader.mean_pixel, tf.uint8))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        start_time = time.time()
        for step in range(step_num):
            _, loss_t, cl, sl = sess.run([train_op, total_loss, content_loss, style_loss])
            print("step:",step," total_loss:",loss_t,"content_loss:",cl,"style_loss:",sl)
        print("Done in time ",time.time()-start_time)
        out = sess.run(output_image)
        with open('out.jpg', 'wb') as f:
            f.write(out)

if __name__ == '__main__':
    activate()
