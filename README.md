#  BloodNet: An attention-based deep network for accurate, efficient, and costless bloodstain time since deposition inference
****
## The time since deposition (TSD) of a bloodstain, i.e., the time of a bloodstain formation is an essential piece of biological evidence in crime scene investigation. The practical usage of these existing microscopic methods is limited, as their performance strongly relies on high-end instrumentation and/or rigorous laboratory conditions. This paper presents a practically applicable deep learning-based method (i.e., BloodNet) for efficient, accurate, and costless TSD inference from a macroscopic view, i.e., by using easily accessible bloodstain photos. To this end, we established a benchmark database containing around 50,000 photos of bloodstains with varying TSDs.

# BloodNet
****
![image](https://github.com/shenxiaochenn/BloodNet/blob/main/fig1.jpg)
****
# Class Activation Mapping and attention weights map
![image](https://github.com/shenxiaochenn/BloodNet/blob/main/fig5.jpg)
## Installation
1. [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. Clone the repository in your computer
```
git clone https://github.com/shenxiaochenn/BloodNet.git && cd BloodNet
```



*******
## Usage: To train and/or test the BloodNet 
First, you need to put the data into the data folder and the weights of the model into the weight folder.
### Train
```
cd /BloodNet-main/train_test/

python main_train.py --weights xxx --batch_size xxx --learning_rate xxx

python main_regression.py --weights xxx --batch_size xxx --learning_rate xxx
```

### Test
```
cd /BloodNet-main/train_test/

python testset_test.py  #(for classification)

python regression_test.py #(for regression)
```

## Datasets and network weights

|  | Data | weights |
| :-----: | :----: | :----: |
| link | [bloodstain](https://pan.baidu.com/s/1cCS1ky7O9Mcv-gCId1VRGQ) | [model weights](https://pan.baidu.com/s/1b8MPJcDt59vx8Cfm1zE94w) |
| access code | shen | chen |

The name of each folder implies the label of the category (regression).

### Note: The data here are already pre-processed. You only need to follow the code we have written above to successfully reproduce our results. For more details, you can get them by registering the use by sending an email to the corresponding author and filling out the questionnaire. chunfeng.lian@mail.xjtu.edu.cn.
 
