## ML_repo 

<p align="center"><img width="100" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/TensorFlowLogo.svg/225px-TensorFlowLogo.svg.png" />  <img width="100" src="https://media-thumbs.golden.com/OLqzmrmwAzY1P7Sl29k2T9WjJdM=/200x200/smart/golden-storage-production.s3.amazonaws.com/topic_images/e08914afa10a4179893eeb07cb5e4713.png" /></p>

ML 관련 network별 정리된 코드 모아놓은 repo. 


## Index
#### 0. reference 
- [모두를 위한 머신러닝/딥러닝 강의 - 김성훈교수님](https://hunkim.github.io/ml/) 
- [MNIST dataset소개 - 박형준](https://github.com/DevHyung/ML_repo/blob/master/ref/%5B190401%5D%20MNIST.pdf) - Introduce the dataset
- [Google Colab소개 - 박형준](https://github.com/DevHyung/ML_repo/blob/master/ref/%5B190408%5D%20Google%20colab.pdf) - Introduce the how to use
- [PyTorch framework 개요 - 박형준](https://github.com/DevHyung/ML_repo/blob/master/ref/%5B190408%5D%20PyTorch.pdf) - Introduce the how to use
  - exist the torch example code in the folder 
  
#### 1. CNN(Convolutional Neural Network)

- 1-1. [TextCNN](https://github.com/DevHyung/ML_repo/tree/master/1-1.TextCNN) - **Binary Sentiment Classification**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - Colab - [TextCNN_Tensor.ipynb](https://colab.research.google.com/drive/1JCxbYSIdtDA-MpaOIDX2qnEiU8gh84ir), [TextCNN_Torch.ipynb](https://colab.research.google.com/drive/1d64elZC54k6UJsQz_VOmsX8D0cFfuKk3)

#### 2. RNN(Recurrent Neural Network)

- 2-1. [TextRNN](https://github.com/DevHyung/ML_repo/tree/master/2-1.TextRNN) - **Predict Next Step**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN_Tensor.ipynb](https://colab.research.google.com/drive/14Y_KZJGani5CQ2k2kaY8Mz4vKnNRpubg), [TextRNN_Torch.ipynb](https://colab.research.google.com/drive/1SbOU-fkpHmI5DMIgNiAhL1xulcI3XpEl)
- 2-2. [Bi-LSTM](https://github.com/DevHyung/ML_repo/tree/master/2-2.Bi-LSTM) - **Predict Next Word in Long Sentence**
  - Paper - [Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification](https://www.aclweb.org/anthology/P16-2034)
  - Colab - [Bi_LSTM_Tensor.ipynb](https://colab.research.google.com/drive/18HrxF3JDlerlyaDR9NMbYI52ZnNRIENr), [Bi_LSTM_Torch.ipynb](https://colab.research.google.com/drive/1IhCC-mFitDRIQ3ghS8wxVxzpQfG4hfnq)

#### 3. Example MNIST
- 3-1. [Basic MNIST](http://solarisailab.com/archives/303)
  - Colab - [MNIST-Tensor.ipynb](https://colab.research.google.com/drive/1wju9m13NEjRgh0pgx65vhCa9mPUN56lg)
- 3-2. [CNN MNIST](https://coderkoo.tistory.com/13)
  - Colab - [MNIST-CNN-Tensor.ipynb](https://coderkoo.tistory.com/13)
## Env

- Python 3.x
- Pytorch 0.4.1+

## source from by 
- @ioatr
- @graykode

