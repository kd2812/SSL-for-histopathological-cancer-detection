# SSL-for-histopathological-cancer-detection

This project aims to apply contrastive Self-Supervised learning to detection of cancer in histopathological images of size 96x96. It highlights the differences of SSL with a baseline of similar architecture but no pretraining. 

## Motivation
Medical histopathological cancer images have a big disadvantage. The amount of labelled data is sparse and becomes a bottle-neck for ML models to train on. However, there are lots of images which are unlabelled. This is where we come across Contrastive Self-supervised Learning. Contrastive SSL trains on images by creating latent vectors of each image and increasing the similarity between similar images. Similar images are made by changing the original image in a certain aspect.

## Dataset
The Dataset details are given below:
- Dataset Used : PatchCamelyon
- Each Image : 96 x 96 image of a patch of cells
- test file : 58k images
- train file : 150k images

```Implementation details:```
- Training Set: 58k test_file images + 80k train_file images
- Linear Eval Set: 70k train_file images
- Eval Set: 70k train_file images

## Approach 
The Project use SimCLR using NTXent Loss to increase similarity between similar images. A linear layer is put on top of this to process the the representations and give a single logit output. 

### Phase 1: SimCLR Pretraining 
The SimCLR model consists of a Resnet18 model with a few changes.
- Replaced the default 7×7, stride=2 stem convolution with a 3×3, stride=1 convolution so spatial resolution is preserved early on
- Removed the initial maxpool downsampling layer to prevent shrinking small feature maps too quickly
- Removed the final 1,000-class linear layer so the model outputs a 512-dimensional feature embedding 

```Training details:```
- Train set : 138k images [96 x 96 patches]
- Optimizer : AdamW
- Learning Rate : 5e-5
- Weight Decay : 1e-5
- Loss Function : NTXent Loss
- Batch Size : 64
- Epochs : 2
- temperature(tau) : 0.5

- The pytorch dataset class returns the original image and the image after some data augmentations.

```Data Augmentations```
- Color Jitter
- Image Inversions
- Gaussian Blurr


```NTXent Loss function details```
- loss function for a pair:

$$\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- loss function for a mini-batch :

$$\mathcal{L} = \frac{1}{2N} \sum_{k=1}^{N} \left[ \ell(2k-1, 2k) + \ell(2k, 2k-1) \right]$$

where :

* $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ is the cosine similarity between two L2-normalized vectors.
* $z_i, z_j$ are the projected representations of a positive pair (two augmented views of the same sample).
* $N$ is the batch size (resulting in $2N$ augmented views).
* $\tau$ is the temperature scaling parameter.
* $\mathbb{1}_{[k \neq i]} \in \{0, 1\}$ is an indicator function evaluating to $1$ iff $k \neq i$.


### Phase 2: Label-Efficiency evaluation 
- We train the MLP on top of the SSL Encoder by freezing the weights of the Encoder and training the layer by using different percentage of number of labelled data. 
- We use a Dataset class in pytorch that has a parameter p which sets the percentage of labels used.
- We plot the efficiency curves of the Model trained at different label percentages on different metrics.

```Metrics Documented:```
- Accuracy
- F1-Score [threshold = 0.5]
- ECE
- Specificity
- Recall
- AUC-ROC


### Phase 3: Calibration and Interpretability


## Project structure

## Setup and Installation 
## How to run  

## Results
### Efficiency Curve
### Calibration [Reliability Diagram]
### Embedding Visualisation [t-SNE]
### Grad-CAM

## Discussion 
## Limitations
## Scope changes
## Reproducibility
## Future Work
## References