#  BloodNet: An attention-based deep network for accurate, efficient, and costless bloodstain time since deposition inference
****
## The time since deposition (TSD) of a bloodstain, i.e., the time of a bloodstain formation is an essential piece of biological evidence in crime scene investigation. The practical usage of these existing microscopic methods is limited, as their performance strongly relies on high-end instrumentation and/or rigorous laboratory conditions. This paper presents a practically applicable deep learning-based method (i.e., BloodNet) for efficient, accurate, and costless TSD inference from a macroscopic view, i.e., by using easily accessible bloodstain photos. To this end, we established a benchmark database containing around 50,000 photos of bloodstains with varying TSDs.

# BloodNet
****
![image](https://github.com/shenxiaochenn/BloodNet/blob/main/fig1.jpg)
****
# Class Activation Mapping and attention weights map
![image](https://github.com/shenxiaochenn/BloodNet/blob/main/fig5.jpg)
## Installation (only Linux)
### 1. [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
### 2. Clone the repository in your computer
```
git clone https://github.com/shenxiaochenn/BloodNet.git && cd BloodNet
```
### 3. Build dependencies
3.1 [Install Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

3.2 Install dependency in command line
```Bash
conda create -n bloodnet python=3.8

conda activate bloodnet

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt 

```
### 4. Prepare the blood dataset

4.1 Download the [data.zip](https://figshare.com/articles/dataset/BloodNet_An_attention-based_deep_network_for_accurate_efficient_and_costless_bloodstain_time_since_deposition_inference/21291825) Dataset under "./data/".

### 5. Prepare the model weight

5.1 Download the [model_weight](https://figshare.com/articles/dataset/BloodNet_An_attention-based_deep_network_for_accurate_efficient_and_costless_bloodstain_time_since_deposition_inference/21291825) model weights under "./weight/".

----bloodnet50_new.pth The weights corresponding to the classification model.

----bloodnet50_reg.pth  The weights corresponding to the regression model.

----bloodnet(small).pth The weights corresponding to the small classification model.

----seresnet50-60a8950a85b2b.pkl The weights corresponding to the Imagenet pretrain model.


*******
## Usage: To train and/or test the BloodNet 
First, you need to put the data into the data folder and the weights of the model into the weight folder.
## a quick start

```python
(the accuaracy of bloodnet in test dataset)
CUDA_VISIBLE_DEVICES=0,1,2 python testset_test.py

(the R^2 of bloodnet in test dataset)
CUDA_VISIBLE_DEVICES=0,1,2 python regression_test.py

```
### Train

### 1. train the classification model
```
cd ./BloodNet/train_test/

CUDA_VISIBLE_DEVICES=0,1,2 python main_train.py --weights='../weight/bloodnet50_new.pth' --batch_size=64 --learning_rate=3e-4 --num_workers=8

```
### 2. train the regression model
```
cd ./BloodNet/train_test/


CUDA_VISIBLE_DEVICES=0,1,2 python main_regression.py --weights '../weight/seresnet50-60a8950a85b2b.pkl' --batch_size=128 --learning_rate=3e-4 --num_workers=8

```



## Datasets and network weights

|  | Data | weights |
| :-----: | :----: | :----: |
| link(bai du) | [bloodstain](https://pan.baidu.com/s/1cCS1ky7O9Mcv-gCId1VRGQ) | [model weights](https://pan.baidu.com/s/1b8MPJcDt59vx8Cfm1zE94w) |
| access code | shen | chen |
| link(figshare) | [data](https://figshare.com/articles/dataset/BloodNet_An_attention-based_deep_network_for_accurate_efficient_and_costless_bloodstain_time_since_deposition_inference/21291825) | [model_weight](https://figshare.com/articles/dataset/BloodNet_An_attention-based_deep_network_for_accurate_efficient_and_costless_bloodstain_time_since_deposition_inference/21291825) | 

The name of each folder implies the label of the category (regression).

### Note: The data here are already pre-processed. You only need to follow the code we have written above to successfully reproduce our results. For more details or if you have any concerns, please do not hesitate to contact us via chunfeng.lian@xjtu.edu.cn.
 
