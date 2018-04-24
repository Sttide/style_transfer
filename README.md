# Fast neural style transfer

This is the code learned from the paper [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://arxiv.org/abs/1603.08155) by Johnson et al.

This technique uses loss functions based on a perceptual similarity and style similarity as described by [Gatys et al](http://arxiv.org/abs/1508.06576) to train a transformation network to synthesize the style of one image with the content of arbitrary images. After it's trained for a particular style it can be used to generate stylized images in one forward pass through the transformer network .


### Usage method

1.First get the dependecies (COCO training set images and VGG model weights):

`./essential.sh`

2.(1)To generate an image directly from style and content, typically to explore styles and parameters(slowly):

`python3 style_transfer.py`

(2)To train a model for fast stylizing first download dependences (training set images and VGG model weights):

`./essential.sh`

Then start training:
`python3 fast-neural-style.py  --model_name my_model`

`--model_name ` is file name of model weights. The paper uses the [COCO image dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip) (13GB).

To generate images fast with an already trained model:
`python3 generate.py --model_name my_model`

All settings and hyperparameters in "style_transfer.py" or "fast-neural-style.py" where you can modify them.There are some default settings."content.jpg":the name of picture you want to change. "style.jpg" the name of style provider.

### Requirements

- Tensorflow  python3
- [VGG-19 model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
- [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip)

### Acknowledgement

- [github] (https://github.com/Sttide/style_transfer)
