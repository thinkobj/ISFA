## Unsupervised Domain Adaptation Based Image Synthesis and Feature Alignment for Joint Optic Disc and Cup Segmentation
by Haijun Lei, Weixin Liu, Hai Xie, Benjian Zhao, Guanghui Yue and Baiying Lei

# 1.Introduction
This repository is the Pytorch implementation of ''Unsupervised Domain Adaptation Based Image Synthesis and Feature Alignment for Joint Optic Disc and Cup Segmentation''

# 2. Abstract
Due to the discrepancy of different devices for fundus image collection, a well-trained neural network is usually unsuitable for another new dataset. To solve this problem, the unsupervised domain adaptation strategy attracts a lot of attentions. In this paper, we propose an unsupervised domain adaptation method based image synthesis and feature alignment (ISFA) method to segment optic disc and cup on the fundus image. The GAN-based image synthesis (IS) mechanism along with the boundary information of optic disc and cup is utilized to generate target-like query images, which serves as the intermediate latent space between source domain and target domain images to alleviate the domain shift problem. Specifically, we use content and style feature alignment (CSFA) to ensure the feature consistency among source domain images, target-like query images and target domain images. The adversarial learning is used to extract domain invariant features for output-level feature alignment (OLFA). To enhance the representation ability of domain-invariant boundary structure information, we introduce the edge attention module (EAM) for low-level feature maps. Eventually, we train our proposed method on the training set of the REFUGE challenge dataset and test it on Drishti-GS and RIM-ONE_r3 datasets. On the Drishti-GS dataset, our method achieves about 3% improvement of Dice on optic cup segmentation over the next best method. We comprehensively discuss the robustness of our method for small dataset domain adaptation. The experimental results also demonstrate the effectiveness of our method. 

# 3. Domain shift
![Image text](https://github.com/thinkobj/ISFA/blob/main/figure/domain%20shift.png)

# 4. Network Structure
![Image text](https://github.com/thinkobj/ISFA/blob/main/figure/network%20structure.png)

# 5. Training and testing
## 5.1 Preparation for dataset
apply image synthesis to oatain target-like query images. (the code of image synthesis will release soon)

## 5.2 Training the model
python train.py --data_dir=/path/to/ISFA/data/ 
## 5.3 Predict the masks:
python test.py --data_dir=/path/to/ISFA/data/ --model-file=./logs/your_checkpoint_dir  


# 6. Unsupervised Domain Adaptation Segmentation Results
## 6.1 The segmentation performance of different methods
![Image text](https://github.com/thinkobj/ISFA/blob/main/figure/result%20of%20segmentation%20performance.png)

## 6.2 Visualization of segmentation results
![Image text](https://github.com/thinkobj/ISFA/blob/main/figure/visualization%20of%20segmentaion%20results.png)

# 7. Citation
@ARTICLE{9444869,  
author={Lei, Haijun and Liu, Weixin and Xie, Hai and Zhao, Benjian and Yue, Guanghui and Lei, Baiying},  
journal={IEEE Journal of Biomedical and Health Informatics},   
title={Unsupervised Domain Adaptation Based Image Synthesis and Feature Alignment for Joint Optic Disc and Cup Segmentation},  
year={2021},  
volume={},  
number={},  
pages={1-1},  
doi={10.1109/JBHI.2021.3085770}}

# 8. Acknowledgement
Part of the code is revised from the Pytorch implementation of [BEAL](https://github.com/emma-sjwang/BEAL).
