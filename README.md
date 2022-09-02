# MNIST Neural Network Widgets

### **[Try the demo](https://www.cluoma.com/?page=blog&id=48)**

This repo is for a pair of web component widgets. Both are based around training a neural network on the MNIST dataset. They allow users to draw a digit and have the neural network predict what they drew.

Both widgets run completely client-side using either straightforward matrix calculations or TensorFlow.js.

![Alt text](widget_screenshot.png?raw=true "screenshot")

## Multilayer Perceptron Network

`widget/`

The `nn_c2.c` file contains code to train a fully connected NN model using only basic linear algebra.

It outputs 6 CSV files with the weights and biases of the trained model. Combine these into a single JSON using the `to_json.R` script.

The feedforward calculations are recreated in the widget using Math.js and the trained model parameters.

## Convolutional Network

`widget_conv/`

The convolutional network is trained using Keras and is contained in the `mnist_conv_train.py` script. The model output is stored in `tfmodels/` using TensorFlow.js so that it is ready for use in the widget.

The widget uses TensorFlow.js to load the trained model and do the predictions.

## Acknowledgements

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap2.html)
- [https://github.com/mco-gh/mnist-draw](https://github.com/mco-gh/mnist-draw)

