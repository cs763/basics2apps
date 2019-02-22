<!DOCTYPE html>
<html>


<body class="stackedit">
  <div class="stackedit__html"><h1 id="lab-1"><p align="center">LAB 3</p></h1>
<h1 id="february-20-2019"><p align="center">February 27 2019</p></h1>
    <h3 id="problem"><p align="center">Problem</p></h3>
    <br>
In this lab, we’ll try to perform a sequential classification task using RNN. The task is to predict
the country of origin of an input lastname. We’ll write a character-level RNN - an RNN which
takes one character at one time step - and output the category once it has seen all the input
letters of a name. The dataset can be found in ./data/ directory.
<p>The implementation tasks for the assignment are divided into two parts:</p>
<ol>
<li>Writing a dataloader to load the temporal data in an orderly fashion.</li>
<li>Designing an RNN architecture using PyTorch’s nn module.</li>
<li>Training the RNN</li>
</ol>
<p>Below you will find the details of tasks required for this assignment.</p>
<ol >
<li><strong>DataLoader</strong>: Like last time, we provide a skeleton code for the
dataloader. the <code>__init__()</code> function reads the data and creates two
lists: <code>self.inputs</code> and <code>self.labels</code>. The self.inputs list
contains all the names in the dataset shuffled randomly.
<code>self.labels</code> contains the corresponding country names. Your task is
to implement the <code>__getitem__()</code> function. It’s input is a random
index in the range between 0 and number of training images. Note
that both the input and labels are currently lists of strings. You
need to do the following pre-processing:</li>
<ol >
<li>
<p><strong>inputs</strong>: Consider the input three lettered name ’abc’. Assuming    there are n letters in your alphabet, the preprocessed name should be
a numpy array of dimension 3 × <em>n characters</em>. For every character,
create a one-hot vector input of dimension 1<em>×n</em>. Now, since
different names can be of different length, you need to appropriately
pad the inputs to enable batching of your inputs. <strong>Yes, the
dataloader expects every element of the batch to be of the same
size.</strong> Find the max number of letters that a name has in the dataset
(18), and create a numpy array of the same size. The padding should
be prepended, i.e. if your name has 15 characters and the max length
is 18, the first three inputs should be zero and the remaining 15
should be the actual one-hot-encoded vectors.</p>
</li>
<li>
<p><strong>labels</strong>: For granting labels, associate every country name with a number between 0 to <em>n_countries</em>. This is similar to what we did in Lab-1.</p></li>
</ol>
<p>At the end of preprocessing, your dataloader should output a <code>n_batch × max_length × n</code> dimensional input and a n batch dimensional target tensor.</p>


<li><strong>Designing an RNN:</strong> Please find the skeleton code in <a href="http://model.py">model.py</a> file. We have designed a basic RNN network for you. It contains just one hidden layer. You may try playing with the the layer size or number of hidden layers. You may also wish to implement a custom LSTM of your own.</li>
<li><strong>Training:</strong> RNNs need a small tweak in the way the gradients are backpropagated. The inputs need to be fed to the model in a loop, one time step at a time, and, the backprop happens only after the loop is complete. Note that this is where the underlying graph structure plays its role and automatically unrolls the network during backprop. Another detail that needs to be kept in mind is that the hidden state should be initialized before every iteration.</li>
</ol>
<p>In this part, fill-in the missing lines in <code>train()</code> function which runs the loop for one epoch.<br>
If trained well, the network should reach a training accuracy of 75%.</p>
</div>
</body>

</html>