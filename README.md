## bitwise_batch_normalization
 Implementation of test code for bitwise batch normalization for binary neural networks
 
 Purpose: 
 
 - find out if my proposed version is indeed an acceptable approximation
 - investigate how much of the information would be lost
 - test the gradient computation for the backpropagation
 - vanilla reference implementation for hardware friendly implementation for FPGA


my reference implementation:

https://github.com/davisking/dlib/blob/master/dlib/dnn/cpu_dlib.cpp#L635 

DLIB is a pure C++ library for deep learning that also uses tensors. The library also provides an CUDA implementation but for our purposes, the CPU implementation seems to be a good fit. 

There is also the Tensorflow implementation but since that one is in python, it's less clear how we would use it

https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/ops/nn_impl.py#L730 


gradient decent theory for standard batch normalization:

https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html 

Theory for bitwise implementation on shared drive document "batch normalization explanation"
