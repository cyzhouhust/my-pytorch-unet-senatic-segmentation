# my-pytorch-unet-senatic-segmentation
senatic segmentation based on unet<br>
这是一个基于韦兹曼恩数据包的语义分割实验<br>
运行说明：<br>
1.创建文件夹，把unet.py放入目录<br>
2.在目录下创建bag_dataset和bag_dataset_mask两个文件夹，并分别放入韦兹曼恩数据包<br>
3.在目录中创建checkpoints数据包，用来存放训练好的模型<br>
本次实验中选用了unet网络模型<br>
![alt](https://pic2.zhimg.com/v2-a2dff868c27f24fb778912ae150e994f_r.jpg "title")<br>
一.模型介绍<br>
Unet 发表于 2015 年，属于 FCN 的一种变体。Unet 的初衷是为了解决生物医学图像方面的问题，由于效果确实很好后来也被广泛的应用在语义分割的各个方向，比如卫星图像分割，工业瑕疵检测等。Unet 跟 FCN 都是 Encoder-Decoder 结构，结构简单但很有效。<br>
Encoder 负责特征提取，你可以将自己熟悉的各种特征提取网络放在这个位置。由于在医学方面，样本收集较为困难，作者为了解决这个问题，应用了图像增强的方法，在数据集有限的情况下获得了不错的精度。<br>
二.代码实现过程<br>
1.首先将数据和标签图片读入<br>
2.导入unet固定网络<br>
3.编写训练函数，利用梯度下降法得到训练模型并保存<br>
4.在测试集上检验时，不需要梯度回传<br>
5.利用biou和miou检验训练精度<br>

