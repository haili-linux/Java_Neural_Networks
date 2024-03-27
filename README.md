# Java_Neural_Networks
   完全基于java的神经网络，可用于java以下创建和训练较小的神经网络模型。出于学习机器学习底层原理目的编写。可以创建cnn、resNet、gan、autoencode等模型。

主要内容:
1. 类似TensorFlow的Sequential类创建模型。
2. 以层Layer为最基础模块(Sequential也可以视为层)，可对层forward、backword，求解层的权重参数的Gradient。
3. 支持的层如下: Dense、Conv2D、Pooling2D、Conv2DTranspose、ResBlock。可根据Layer基类自定义层。
4. 常见activation和loss，如sigmoid、tanh、rule、LRelu、softmax。mseloss、celoss。
5. 支持Adam梯度下降优化。
6. 可以多线程并行mini-batch训练。
7. 将模型保存到文件、从保存的模型文件加载模型。


使用样例见https://github.com/haili-linux/MNIST_Java
