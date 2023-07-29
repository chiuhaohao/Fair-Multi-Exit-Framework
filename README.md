# Toward Fairness Through Fair Multi-Exit Framework for Dermatological Disease Diagnosis [MICCAI 2023]

This is the official repository of the following paper:

**Toward Fairness Through Fair Multi-Exit Framework for Dermatological Disease Diagnosis**<br>
Ching-Hao Chiu, Hao-Wei Chung, Yu-Jen Chen, Yiyu Shi, Tsung-Yi Ho

[[arxiv]([https://arxiv.org/pdf/2306.14518v1.pdf](https://arxiv.org/pdf/2306.14518.pdf))] 


## Setup & Preparation
### Environment setup
```bash
pip install -r requirements.txt
```

## Training
For training, using the command in train.sh to train the ME-model.
The command is as follow
```bash
python3 [script name] --training_title [folder title] --epochs [num of epoch] --lr [learning rate] --batch_size [batch size] --dataset [isic2019 or fitzpatrick17k] --model [model type] --class_num [8 or 114]  
```

## Evaluation
Using eval_me.py to evaluate the accuracy and fairness scores for model.
```bash
python3 eval_me.py
```
