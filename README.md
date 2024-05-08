# ResNet-50 Image Classifier

This is an Image Classifier that follows the Residual Network architecture with 50 layers that can be used to classify objects from among 6 different categories with a high accuracy. In recent years, neural networks have become deeper, with state-of-the-art networks going from just a few layers (e.g., AlexNet) to over a hundred layers. The main benefit of a very deep network is that it can represent very complex functions. It can also learn features at many different levels of abstraction, from edges (at the lower layers) to very complex features (at the deeper layers). However, using a deeper network doesn't always help. A huge barrier to training them is vanishing gradients: very deep networks often have a gradient signal that goes to zero quickly, thus making gradient descent unbearably slow.

<div align="center">
   <img src="./images/resnet.png" width=450 height=350>
</div>

In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:

<div align="center">
   <img src="./images/skip_connection_kiank.png" width=650 height=250>
</div>

The "identity block" is the standard block used in ResNets, and corresponds to the case where the input activation (say a[l]) has the same dimension as the output activation (say a[l+2]):

<div align="center">
   <img src="./images/idblock2_kiank.png" width=650 height=250>
</div>

Next, the ResNet "convolutional block" is the other type of block. You can use this type of block when the input and output dimensions don't match up. The difference here is that there is a CONV2D layer in the shortcut path:

<div align="center">
   <img src="./images/convblock_kiank.png" width=650 height=250>
</div>

The detailed structure of this ResNet-50 model (I've also added additional Batch-Norm and Dropout Layers since they're absolutely awesome):

![ResNet-50](./images/resnet_kiank.png)

## Dataset

The machine learning tool was trained on a dataset of 730 wounds, collected by and labeled by specialists at the AZH Wound and Vascular Center in Wisconsin. These 730 images were then perturbed, by adding random noise, to make a training set of 3650 images total. The images were split into the following six categories: background, normal, surgical, venous, diabetic, pressure. The categorical split was approximately equal.

Our preprocessing script `prepro.py` will handle the data structure.

## Getting Started

In order to train the model and make predictions, you will need to install the required python packages using:

```bash
pip install -r requirements.txt
```

Now, we need to do some preprocessing (Data Augmentation and Train/Val/Test Split) of our dataset:

```bash
python prepro.py --dataset-path datasets/caltech_101
```

Once you're done with all that, you can open up a terminal and start training the model (FYI: it takes a **while**):

```bash
python train.py -lr 0.005 --num-epochs 50 --batch-size 32 --save-every 5 --tensorboard-vis
```

Passing the `--tensorboard-vis` flag allows you to view the training/validation loss and accuracy in you browser using:

```bash
tensorboard --logdir=./logs
```

Once you're done training run the prediction script which will load the pretrained model and make a prediction on your test image:

```bash
python predict.py images/test.jpg
```

## Results

Training:

```yaml
number of training examples: 72893
X_train shape: (72893, 64, 64, 3)
Y_train shape: (72893, 101)
```

```yaml
number of validation examples: 2020
X_train shape: (2020, 64, 64, 3)
Y_train shape: (2020, 101)
```

```
Epoch 50/50:
2187/2187 [==============================] - 393s 175ms/step - loss: 0.0341 - acc: 0.9888 - val_loss: 0.3118 - val_acc: 0.9311

Val Loss = 0.3118
Val Accuracy = 93.11% (0.9311)
```

Testing:

```yaml
number of test examples: 2020
X_test shape: (2020, 64, 64, 3)
Y_test shape: (2020, 101)
```

```
68/68 [==============================] - 30s 437ms/step
Loss = 0.3213
Test Accuracy = 92.97% (0.9297)
```

Model Parameters:

```yaml
Total params: 23,794,661
Trainable params: 23,741,541
Non-trainable params: 53,120
```

## Built With

* Python
* Keras
* TensorFlow
* NumPy
