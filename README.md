# Deep learning model for early Alzheimer’s disease detection from structural MRIs
This repository contains code for a reproduction of a  medical [paper](https://www.nature.com/articles/s41598-022-20674-x) and a machine learning [paper](http://proceedings.mlr.press/v116/liu20a) on deep learning for dementia. 


<p float="left" align="center">
<img src="overview.png" width="800" /> 
<figcaption align="center">
Figure: Overview of the deep learning framework and performance for Alzheimer’s automatic diagnosis. (a) Deep learning framework used for automatic diagnosis. 
  

## Prerequisites
- Python 3.6
- PyTorch 0.4
- torchvision
- progress
- matplotlib
- numpy
- visdom


## Download OASIS data
1. Download OASIS-1 data at [OASIS website](https://sites.wustl.edu/oasisbrains/home/oasis-1/)
2. Download both the raw data scans and the clinical data. From the linked page scroll down to `Download Instructions`. To download the imaging data, click on `OASIS-1: Raw Data Download` and download all the MRI images.
3. To download the clinical data, click on `OASIS-1: Demographic and Clinical Data` and download `CSV File with Demographic, and Clinical Data`.

## Data Preprocessing
Data Preprocessing with Clinica:
1. **Convert data into BIDS format**: please read the docs on [Clinica website](https://aramislab.paris.inria.fr/clinica/docs/public/dev/Converters/OASIS2BIDS/), and install required softwares and use the previously downloaded clinical files. Note that we first preprocess the training set to generate the template and use the template to preprocess validation and test set. You can find the script we use to run the converter at /datasets/files:
```
run_convert.sh
```

2. **preprocess converted and splitted data**: you can refer our scripts at /datasets/files. For training data, refer:
```
run_adni_preprocess.sh
```
For val and test refer:
```
run_adni_preprocess_val.sh
```
and 
```
run_adni_preprocess_test.sh
```

## Examples in the preprocessed dataset
Here are some examples of scans for each categories in our test dataset:

<p align="center">
<img src="data_examples/CN_example.png" width="600" /> 
<img src="data_examples/MCI_example.png" width="600" /> 
<img src="data_examples/AD_example.png" width="600" /> 
</p>

## Neural Network Training
Train the network ADNI dataset:

```
python main.py
```

You can create your own config files and add a **--config** flag to indicate the name of your config files.

## Model Evaluation
We provide the evaluation code in **Model_eval.ipynb**, where you can load and evaluate our trained model. The trained best model (with widening factor 8 and adding age) can be found [here](https://drive.google.com/file/d/1zU21Kin9kXg_qmj7w_u5dGOjXf1D5fa7/view?usp=sharing). 


## Results
<center>

| Dataset           | ADNI held-out        | ADNI held-out          | NACC external validation | NACC external validation |
| ----------------- | -------------------- | ---------------------- | -----------------------  | ------------------------ | 
|   Model           | Deep Learning model  | Volume/thickness model | Deep Learning model      | Volume/thickness model   |
| Cognitively Normal              | 87.59     | 84.45          | 85.12       | 80.77       |
| Mild Cognitive Impairment       | 62.59     | 56.95          | 62.45       | 57.88       |
| Alzheimer’s Disease Dementia    | 89.21     | 85.57          | 89.21       | 81.03       |
</center>
  
Table 1: Classifcation performance in ADNI held-out set and an external validation set. Area under ROC
curve for classifcation performance based on the  learning model vs the ROI-volume/thickness model,
for ADNI held-out set and NACC external validation set. Deep learning model outperforms ROI-volume/
thickness-based model in all classes. Please refer [paper](https://www.nature.com/articles/s41598-022-20674-x) for more details.

<p float="left" align="center">
<img src="AD_progression_new.png" width="800" /> 
<figcaption align="center">  
Figure: Progression analysis for MCI subjects. The subjects in the ADNI test set are divided
into two groups based on the classifcation results of the deep learning model from their frst scan diagnosed
as MCI: group A if the prediction is AD, and group B if it is not. The graph shows the fraction of subjects that
progressed to AD at diferent months following the frst scan diagnosed as MCI for both groups. Subjects in
group A progress to AD at a signifcantly faster rate, suggesting that the features extracted by the deep-learning
model may be predictive of the transition. 

<center>

| Method             | Acc.        | Balanced Acc. | Micro-AUC  | Macro-AUC |
| ----------------- | ----------- | ----------- | -----------  | ----------- | 
| ResNet-18 3D    | 52.4%      | 53.1%           | -           | -           |
| AlexNet 3D      | 57.2%      | 56.2%           | 75.1%       | 74.2%       |
| X 1             | 56.4%      | 54.8%           | 74.2%       | 75.6%       |
| X 2             | 58.4%      | 57.8%           | 77.2%       | 76.6%       |
| X 4             | 63.2%      | 63.3%           | 80.5%       | 77.0%       |
| X 8             | 66.9%      | 67.9%           | 82.0%       | 78.5%       |
| **X 8 + age**       | 68.2%      | 70.0%           | 82.0%       | 80.0%       |

</center>
  
Table 2: Classifcation performance in ADNI held-out with different neural network architectures. Please refer [paper](http://proceedings.mlr.press/v116/liu20a) for more details.
  
  
## References
  
```
@article{liu2022generalizable,
  title={Generalizable deep learning model for early Alzheimer’s disease detection from structural MRIs},
  author={Liu, Sheng and Masurkar, Arjun V and Rusinek, Henry and Chen, Jingyun and Zhang, Ben and Zhu, Weicheng and Fernandez-Granda, Carlos and Razavian, Narges},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={1--12},
  year={2022},
  publisher={Nature Publishing Group}
}
```
  
```
@inproceedings{liu2020design,
  title={On the design of convolutional neural networks for automatic detection of Alzheimer’s disease},
  author={Liu, Sheng and Yadav, Chhavi and Fernandez-Granda, Carlos and Razavian, Narges},
  booktitle={Machine Learning for Health Workshop},
  pages={184--201},
  year={2020},
  organization={PMLR}
}
```
