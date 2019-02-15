# Deriving Machine Attention from Human Rationales

This repo contains the code and data of the following paper:

**[Deriving Machine Attention from Human Rationales](https://arxiv.org/pdf/1808.09367.pdf).** *Yujia Bao, Shiyu Chang, Mo Yu and Regina Barzilay. EMNLP 2018.* 

If you find this work useful and use it on your own research, please cite our paper.
```
@article{bao2018deriving,
  title={Deriving Machine Attention from Human Rationales},
  author={Bao, Yujia and Chang, Shiyu and Yu, Mo and Barzilay, Regina},
  journal={arXiv preprint arXiv:1808.09367},
  year={2018}
}
```

## Overview

The R2A model first learns to map binary rationales into continuous attention scores on the source tasks. Then the trained R2A model is used to predict how attention should look like based on human-annotated rationales for the low-resource target task. Finally, we train a target classifier under the supervision of both the annotated labels and the R2A-generated attention. The following figure illustrates our learning pipeline.

<p align="center">
<img src="assets/pipeline.png" alt="drawing" width="500"/>
</p>


## Models
**Instructions to run the code are provided within each directory.**
+ Directory [`r2a`](r2a/) contains the source code and pre-trained models for our R2A model.
+ Directory [`rationalization`](rationalization/) contains the code we used for automatic rationale generation.


## Data
### Download
The original raw dataset can be found at: [beer review](https://snap.stanford.edu/data/web-BeerAdvocate.html), [hotel review](http://www.cs.virginia.edu/~hw5x/dataset.html).

We provide the processed data (together with the machine-generated rationales) that we used for all our experiments at [data.zip](https://people.csail.mit.edu/yujia/files/r2a/data.zip). **Important Note:** this data is for research-purpose only.


### Usage
1. Unzip [data.zip](https://people.csail.mit.edu/yujia/files/r2a/data.zip) to the root directory of this repo.
2. There are three directories under the directory `data`, named as `source`, `target` and `oracle`.
   + `source` includes all source data files. Each data file is a *tsv* file that contains the following fields: task name, label, text (tokenized and separated by space), rationale label (a sequence of binary integer separated by space).
   
     | Task | #train (file) | #dev (file) |
     | -----|-------|-----|
     | Beer look | 43,351 (beer0.train) | 10,170 (beer0.dev)
     | Beer aroma | 39,825 (beer1.train) | 8,772 (beer1.dev)
     | Beer palate | 30,041 (beer2.train) | 7,152 (beer2.dev)

   + `oracle` contains the data used to derive the oracle attention. The data format is the same as the one in `source`.
     
     | Task | #train (file) | #dev (file) |
     | -----|-------|-----|
     | Beer look | 32,276 (beer0.train) | 6392 (beer0.dev)
     | Beer aroma | 28,984 (beer1.train) | 5,720 (beer1.dev)
     | Beer palate | 25,748 (beer2.train) | 4,994 (beer2.dev)
     | Hotel location | 14,472 (hotel_Location.train) | 1,813 (hotel_Location.dev) |
     | Hotel cleanliness | 150,098 (hotel_Cleanliness.train) | 18,764 (hotel_Cleanliness.dev) |
     | Hotel service | 101,484 (hotel_Service.train) | 12,689 (hotel_Service.dev) |
     
   + `target` contains the data for the target tasks. 
      + `hotel_unlabeled.train`, `hotel_unlabeled.dev`: unlabeled data file. Each row is a hotel review. Used for training the domain-invariant encoder of our R2A model.
      + `*.dev`, `*.test`: target development and test set. The data format is the same as the one in `source`.
          
           | Task | #dev (file) | #test (file) |
           | -----|-------|-----|
           | Beer look | 200 (beer0.dev) | 4014 (beer0.test)
           | Beer aroma | 200 (beer1.dev) | 4212 (beer1.test)
           | Beer palate | 200 (beer2.dev) |  3804 (beer2.test)
           | Hotel location | 200 (hotel_Location.dev) | 1808 (hotel_Location.test) |
           | Hotel cleanliness | 200 (hotel_Cleanliness.dev) | 12684 (hotel_Cleanliness.test) |
           | Hotel service | 200 (hotel_Service.dev) | 18762 (hotel_Service.test) |
      
      + `*.train`: target training set. Each data file (except `hotel_unlabeled.dev` and `hotel_unlabeled.train`) is a *tsv* file that contains the following fields: 1) task name, 2) label, 3) text (tokenized and separated by space), 4) rationale label (a sequence of binary integer), 5) R2A-generated attention (a sequence of float), 6) oracle attention (a sequence of float), frequency of a word being highlighted as rationale (a sequence of float).
        + `beer0.train` (beer look), `beer1.train` (beer aroma), `beer2.train` (beer palate), `hotel_Location.train`, `hotel_Cleanliness.train`, `hotel_Service.train`: Each data file consists of 200 labeled examples with human annotated rationales. The entries for R2A-generated attention and oracle attention are all zero.
        + `*.pred_att.gold_att.train`: the file contains R2A-generated attention and the oracle attention from pretrained models.
   

## Dependency
+ PyTorch 0.4.1
+ numpy 1.15.1
+ torchtext 0.2.1
+ termcolor 1.1.0
+ tqdm 4.24.0
+ scikit-learn 0.19.2
+ spacy 2.0.12
+ colored 1.3.5
