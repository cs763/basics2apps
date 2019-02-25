<!DOCTYPE html>
<html>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="lab-1"><p align="center">LAB 1</p></h1>
    <h1 id="february-20-2019"><p align="center">February 25 2019</p></h1>
    <h3 id="problem"><p align="center">Object Classification</p></h3>
    <br>
<p>The purpose of this lab is to train a neural network using PyTorch
  on an image classification task. And get good results!

<p> We are going to use a subset of Norb dataset for learning object
  classification. The dataset provided contains approximately 30K
  grayscale images of size 108 x 108. A sample snapshot of the images,
  as provided in the official <a
  href="https://cs.nyu.edu/~yann/research/norb/">webpage</a> is shown
  below. 
<div align="center">
    <img src="https://cs.nyu.edu/~yann/research/norb/jitt-clutt-train-12x4-large.png"
    width=800px /> 
</div>

<p> Zooming into the image, you should find different
  <i>categories</i> of images against complex backgrounds, and taken
  under different illumination conditions. There are six categories:
  humans, aircraft, four-legged animals, trucks, cars and, lastly,
  no-object.

<p> Our data set has can be found in <code>norb.tar.gz</code>. The
  <code>train.npy</code> and <code>test.npy</code> contain the train
  and test images. The corresponding labels can be found in
  <code>train_cat.npy</code> and <code>test_cat.npy</code> files.
  

<p> We also hope
  to compare the performance of a fully-connected network architecture
  against a Convolutional Neural Network architecture.</p> 
<p>The implementation tasks for the assignment are divided into two parts:</p>
<ol>
<li>Designing a network architecture using PyTorch’s nn module.</li>
<li>Training the designed network using PyTorch’s optim module.</li>
</ol>

<p>Below you will find the details of tasks required for this assignment.</p>
<ol>
<li><strong>Fully-Connected Network</strong>: Write a function named <code>create_fcn()</code> in <em><strong>model.py</strong></em> file which returns a fully connected network variable. The network can be designed using nn.Sequential() container (refer to <a href="https://pytorch.org/docs/stable/nn.html#sequential">Sequential</a> and <a href="https://pytorch.org/docs/stable/nn.html#linear">Linear</a> layer’s documentation). The network should have a series of Linear and ReLU layers with the output layer having as many neurons as the number of classes.<br></li>
<li><strong>Criterion</strong>: Define the criterion in line number x. A criterion defines a loss function. In our case, use <code>nn.CrossEntropyLoss()</code> to define a cross entropy loss. We’ll use this variable later during optimization.<br></li>
<li><strong>Optimizer</strong>: In the file <em><strong>train.py</strong></em>, we have defined a Stochastic Gradient Descent Optimizer. Fill-in the values of learning rate, momentum, weight decay, etc. You may also wish to experiment with other optimization functions like RMSProp, ADAM, etc which are provided by nn.optim package. Their documentation can be found in the this <a href="https://pytorch.org/docs/stable/optim.html">link</a>.<br></li>
<li><strong>Data Processing</strong>: The data, stored in <em><strong>dat.npy</strong></em>, contains images of toys captured from various angles. Each image has only one toy and the corresponding label of the images are stored in <em><strong>cat.npy</strong></em>. We have already set-up the data processing code for you. You may, optionally, want to play with the minibatch size or introduce noise to the data. You may also wish to preprocess the data differently. All this should be done in the functions <code>preprocess_data()</code> and <code>add_noise()</code> functions if you wish.<br></li>
<li><strong>Experiments</strong>: Finally, test the networks on the given data. Train the networks for at least 10 epochs and observe the validation accuracy. You should be able to achieve at least 42% accuracy with a fully connected network.<br></li>
<li><strong>Convolutional Neural Network</strong>: So far, we used <code>nn.Sequential</code> to construct our network. However, a Sequential container can be used only for simple networks since it restricts the network type. It is not usable in the case of, say, a Residual Network (ResNet) where the layers are not always stacked serially. To have more control over the network architecture, we’ll define a model class that implements PyTorch’s <code>nn.Module</code> superclass. We have provided a skeleton code in <em><strong>model.py</strong></em> file consisting of a simple CNN. The idea is simple: you need to write the <code>forward()</code> function which takes a variable x as input. As shown in the skeleton code, you may use <code>__init__()</code> function to initialize your layers and simply connect them the way you want in <code>forward()</code> function.  You are free to design your custom network. Again, write a function named <code>create_cnn()</code> which returns a CNN model. This time, we need to use <code>nn.Conv2d()</code> and <code>nn.MaxPool2d()</code> functions. After stacking multiple Conv-ReLU-MaxPool layers, flatten out the activation using <code>torch.View()</code> and feed them to a fully connected layer that outputs the class probabilities.</li>
</ol>
    <p>For your CNN model, make appropriate changes in the <strong><em>train_cnn.py</em></strong> file. Once done, train this network. You should be able to see a significant improvement in performance (45% vs 75%).</p>
</div>
</body>

</html>
