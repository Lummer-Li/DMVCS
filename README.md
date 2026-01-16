<h2 align="center">✨Disentangled Contrastive Multi-view Clustering via Semantic Relevance Invariance</h2>


<p align="center">
  <b>Pengyuan Li<sup>1</sup>, Dongxia Chang<sup>1</sup>, Yiming Wang<sup>2</sup>, Zisen Kong<sup>1</sup>, Linhua Kong<sup>1</sup>, Yao Zhao<sup>1</sup></b>
</p>

<p align="center">
  <sup>1</sup>Institute of Information Science, Beijing Jiaotong University, Beijing, China<br>
  <sup>2</sup>School of Computer Science, Nanjing University of Posts and Telecommunications, Nanjing, China<br>
</p>

<p align="center">
  <!-- IEEE TKDE Badge -->
  <a href="#" target="_blank">
    <img src="https://img.shields.io/badge/Neurocomputing-2025-blueviolet.svg?style=flat-square" alt="IEEE TKDE">
  </a>
  <!-- arXiv Badge -->
  <!-- <a href="https://arxiv.org/abs/2412.08345" target="_blank">
    <img src="https://img.shields.io/badge/arXiv-2412.08345-b31b1b.svg?style=flat-square" alt="arXiv Paper">
  </a> -->
  <!-- Contact Badge -->
  <a href="pengyuanli@bjtu.edu.cn" target="_blank">
    <img src="https://img.shields.io/badge/Email-pengyuanli%40bjtu.edu.cn-blue.svg" alt="Contact Author">
  </a>
</p>

<p align="center">
  🔥 Our work has been accepted by TKDE 2025!<br>
</p>

## Overview🔍
<div>
    <img src="https://github.com/Lummer-Li/DMVCS/blob/main/frame.png" width="90%" height="90%">
</div>

**Figure 1. The framework of the proposed DMVCS.**


**_Abstract -_** The real-world data is relatively complex, generally formed by the interaction of different latent factors. Disentanglement of these latent factors can effectively improve the robustness and interpretability of sample representation. However, most existing disentangled multi-view clustering methods focus on the irrelevance of disentangled representations, ignoring the semantic relevance invariance between different latent factors. To address this issue, we propose a disentangled contrastive multi-view clustering via semantic relevance invariance (DMVCS) to learn the disentangled representations and maintain their semantic relevance. Specifically, we first decompose each view into consistent and specific representations by maximizing semantic consistency and minimizing the correlation between multiple views. Meanwhile, to ensure that different disentangled representations have similar semantic relevance, a cross-component semantic relevance alignment module is proposed. Combined with the hierarchical sampling strategy, the learned semantic relevances are aligned progressively in a locally structure-aware manner. Besides, to learn a clustering-friendly unified representation, we propose a multi-hop neighbor contrastive learning to extend the range of positive samples. Comprehensive experiments on ten public multi-view datasets demonstrate that DMVCS outperforms the state-of-the-art clustering methods.

## Datasets📚
To comprehensively evaluate the performance of the proposed DMVCS method, we conducted experiments using ten publicly available multi-view datasets, as shown below. 

| Dataset      | Samples  | Views  | View Dimensions         | Clusters  |
|--------------|----------|--------|-------------------------|-----------|
| ALOI         | 1079     | 4      | 64/64/77/13             | 10        |
| Caltech-5V   | 1400     | 5      | 40/254/1984/512/928     | 7         |
| UCI-Digit    | 2000     | 3      | 216/76/64               | 10        |
| Handwritten  | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| Mfeat        | 2000     | 6      | 216/76/64/6/240/47      | 10        |
| 100Leaves    | 1600     | 3      | 64/64/64                | 100       |
| ALOI100      | 10800    | 4      | 77/13/64/125            | 100       |
| STL10        | 13000    | 3      | 1024/512/2048           | 10        |
| Caltech256   | 30607    | 3      | 1024/512/2048           | 257       |
| Cifar10      | 50000    | 3      | 512/2048/1024           | 10        |



## Experimental Results🏆


**Table 1. Clustering average results and standard deviations for ten multi-view datasets. The optimal and suboptimal results are highlighted in red and blue. The notation O/M signifies an out-of-memory error encountered during the training process.**
<div>
    <img src="https://github.com/Lummer-Li/DMVCS/blob/main/assets/tab1.png" width="80%" height="96%">
</div>
<!-- <br> </br> -->

**Table 2. Ablation study on UCI-Digit, Handwritten, and 100Leaves, respectively.**
<div>
    <img src="https://github.com/Lummer-Li/DMVCS/blob/main/assets/tab2.png" width="30%" height="96%">
</div>



## Getting Started🚀
### Data Preparation
The dataset should be organised as follows, taking Handwritten as an example:
```text
Handwritten
├── X
│   ├── X1
│   ├── X2
│   ├── X3
│   ├── ...
├── Y
```

### Training and Evaluation
- To train the DMVCS, run: `main.py`. The prediction results obtained using the K-Means algorithm.



## Cite our work📝
```bibtex

```

## License📜
The source code is free for research and educational use only. Any commercial use should get formal permission first.



