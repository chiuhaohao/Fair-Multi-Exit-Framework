from cProfile import label
from cmath import nan
from math import inf
import torch
import time
import data
import pickle
import argparse
import numpy as np
import torch.nn as nn
import aux_funcs as af
import torch.optim as optim

from data import *
from tqdm import tqdm
from model import *
from util.custom_loss import *
from fairness_metric import *
from torch.optim import SGD
from torchvision import models
from collections import Counter
from random import choice, shuffle
from eval_me import *
from torch.utils.tensorboard import SummaryWriter


def forward_features(model, x):
    """
    Dump each IC's feature map.
    Args:
        model (nn.module) : model
        x (torch.tensor) : model input
    """
    early_exits_outputs = []
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            ic_out = early_exits_layer[0](x)
            ic_out = early_exits_layer[1](ic_out)
            early_exits_outputs.append(ic_out)
    early_exits_outputs.append(torch.flatten(x, start_dim=1))
    return early_exits_outputs


def cnn_training_step(model, optimizer, data, labels, sensitive_attr, Lambda, device, args):
    if(not args.train_teacher):
        teacher_model = torch.load(args.teacher_path) 
        teacher_model=teacher_model.to(device)

    mmd_loss = MMDLoss(w_m=args.Lambda, sigma=args.sigma, num_classes=args.class_num, num_groups=args.group_num, kernel='rbf')
    model.train()
    coeff_list = [0.3, 0.45, 0.6, 0.75, 0.9] # 0.1, 0.2, 0.3, 0.4, 1.0
    b_sensitive = sensitive_attr.to(device)
    b_x = data.to(device)          # batch x
    b_y = labels.to(device)        # batch y
    output = model(b_x)            # cnn final output
    if(not args.train_teacher):
        teacher_output = teacher_model(b_x)
        teacher_feature = forward_features(teacher_model, b_x)
    feature = forward_features(model, b_x)
    criterion = af.get_loss_criterion()
    loss = 0
    total_ce_loss = 0
    total_regularization_loss = 0
    for ic_idx in range(5):
        if(not args.train_teacher):
            ce_loss = criterion(output[ic_idx], b_y)
            regularization_loss = mmd_loss.forward(feature[ic_idx], teacher_feature[ic_idx], groups=b_sensitive, labels=b_y)
            if(torch.isnan(ce_loss).any() or torch.isnan(regularization_loss).any()):
                    continue
            if(torch.isinf(ce_loss).any() or torch.isinf(regularization_loss).any()):
                    continue
            loss += (coeff_list[ic_idx] * (ce_loss + args.alpha * regularization_loss))
            total_regularization_loss += args.alpha * regularization_loss
            total_ce_loss += ce_loss
        else:
            ce_loss = criterion(output[ic_idx], b_y)
            if(torch.isnan(ce_loss).any()):
                continue
            if(torch.isinf(ce_loss).any()):
                continue
            loss += (coeff_list[ic_idx] * (ce_loss))
            total_ce_loss += ce_loss
    optimizer.zero_grad()           # clear gradients for this training step
    loss.mean().backward()                 # backpropagation, compute gradients
    torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()                # apply gradients
    if(not args.train_teacher):
        return loss.mean(), total_regularization_loss.mean(), total_ce_loss.mean()
    else:
        return loss.mean(), total_ce_loss.mean()


def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', models_path=None, Lambda=0.5, args=None):
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    for epoch in range(1, epochs):
        loss = []
        ce_loss = []
        regularization_loss = []
        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        # eval_me_original(model, one_batch_dataset, '', '', device)
        for x, y, gender, idx in tqdm(train_loader):
            if(not args.train_teacher):
                _loss, _regularization_loss, _ce_loss = cnn_training_step(model, optimizer, x, y, gender, Lambda, device, args=args)
                loss.append(_loss)
                regularization_loss.append(_regularization_loss)
                ce_loss.append(_ce_loss)
            else:
                _loss, _ce_loss = cnn_training_step(model, optimizer, x, y, gender, Lambda, device, args=args)
                loss.append(_loss)
                ce_loss.append(_ce_loss)
        scheduler.step()
     
        if(not args.train_teacher):
            print("Loss: {}".format(sum(loss) / len(loss)))
            print("Regularization Loss: {}".format(sum(regularization_loss) / len(regularization_loss)))
            print("CE Loss: {}".format(sum(ce_loss) / len(ce_loss)))
        else:
            print("Loss: {}".format(sum(loss) / len(loss)))
            print("CE Loss: {}".format(sum(ce_loss) / len(ce_loss)))
        
            
        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            eval_me_original(model, one_batch_dataset, '', '', device)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_me')
    parser.add_argument('--training_title', type=str, default='',
                    help='')
    parser.add_argument('--epochs', type=int, default=200,
                    help='')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='')
    parser.add_argument('--batch_size', type=int, default=128,
                    help='')
    parser.add_argument('--dataset', type=str, default='isic2019',
                    help='')
    parser.add_argument('--model', type=str, default='me_mfd_resnet18',
                    help='')
    parser.add_argument('--class_num', type=int, default=8,
                    help='')
    parser.add_argument('--group_num', type=int, default=2,
                    help='')
    parser.add_argument('--Lambda', type=float, default=5,
                    help='')
    parser.add_argument('--sigma', type=float, default=0.5,
                    help='')
    parser.add_argument('--alpha', type=float, default=0.02,
                    help='')
    parser.add_argument('--train_teacher', type=bool, default=False,
                    help='')
    parser.add_argument('--teacher_path', type=str, default='',
                    help='')
    args = parser.parse_args()
    training_title = args.training_title
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    print('Random Seed: {}'.format(random_seed))
    device = af.get_pytorch_device()
    models_path = 'networks/{}/{}'.format(af.get_random_seed(), training_title)
    tensor_board_path = 'runs/{}/train_models{}'.format(training_title, af.get_random_seed())

    af.create_path(models_path)
    af.create_path(tensor_board_path)
    af.create_path('outputs/{}'.format(training_title))
    af.set_logger('outputs/{}/train_models{}'.format(training_title, af.get_random_seed()))

    print("Arguments: ")
    argument_list = ""
    for arg in vars(args):
        argument_list += " --{} {}".format(arg, getattr(args, arg))
    print(argument_list)
    
    ds_handler = dataset_handler(args)
    dataset = ds_handler.get_dataset()
    one_batch_dataset = ds_handler.get_dataset(is_one_batch=True)
    
    model = model_handler(args.model, args.class_num)
    model.to(device)
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, models_path, args.Lambda, args=args)
    # eval_me(model, one_batch_dataset, 'resnet', dataset_name='')

