# Deep Adaptive Input Normalization for Price Forecasting using Limit Order Book Data


Deep Learning (DL) models can be used to tackle time series analysis tasks with great success. However, the performance of DL models can degenerate rapidly if the data are not appropriately normalized. This issue is even more apparent when
DL is used for financial time series forecasting tasks, where the non-stationary and multimodal nature of the data pose significant challenges and severely affect the performance of DL models. Deep Adaptive Input Normalization (DAIN) is a simple, yet effective, neural layer, that is capable of adaptively normalizing the input time series, while taking into
account the distribution of the data. DAIN is trained in an end-to-end fashion using back-propagation and can lead to significant performance improvements.


In this repository we provide an implementation of the [Deep Adaptive Input Normalization (DAIN)](https://arxiv.org/pdf/1902.07892.pdf) using PyTorch. Sample data loaders to evaluate the proposed method with a effectiveness of the proposed method is demonstrated using a large-scale limit order book dataset (FI-2010 dataset) are also provided. 

We provide an example of using the proposed method in run_exp.py and we compare DAIN to other normalization approaches. The proposed method can both increase the price forecasting as shown below (evaluation on all splits using a two layer MLP, prediction horizon = 10, window = 15):


| Method         | F1 score  | Cohen's kappa | 
| -------------  | --------- | ------------- |  
| z-score        |   0.550   |     0.327     | 
| Sample Avg.    |   0.434   |     0.205     | 
| DAIN (full)    |   0.682   |     0.514     | 

Please download the preprocessed data from [here](https://www.dropbox.com/s/vvvqwfejyertr4q/lob.tar.xz?dl=0). The dataset was based on the [FI-2010 dataset](https://etsin.avointiede.fi/dataset/urn-nbn-fi-csc-kata20170601153214969115).

If you use this code in your work please cite the following paper:

<pre>
@article{dain,
  title={Deep Adaptive Input Normalization for Price Forecasting using Limit Order Book Data},
  author={Passalis, Nikolaos and Tefas, Anastasios and Kanniainen, Juho and Gabbouj, Moncef and Iosifidis, Alexandros},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2019}
}
</pre>

Check my [website](https://passalis.github.io) for more projects and stuff!

