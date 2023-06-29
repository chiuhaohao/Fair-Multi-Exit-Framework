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
from util.hsic import *
from fairness_metric import *
from torch.optim import SGD
from torchvision import models
from collections import Counter
from random import choice, shuffle
from eval_me import *
from torch.utils.tensorboard import SummaryWriter


def cnn_test(model, loader, device='cpu'):

    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    groupAcc=[]
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            gender  = batch[2].to(device)
            output = model(b_x)[-1]
            _, pred = torch.max(output, 1)
            
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.cpu().numpy())
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)

    for k, v in fairness_metrics.items():
        print('{}:{:.4f}'.format(k, v))


def cnn_training_step(args, model, optimizer, data, labels, sensitive_attr, device, labmda, hsic):
    model.train()
    b_sensitive = sensitive_attr.to(device)
    b_x = data.to(device)          # batch x
    b_y = labels.to(device)        # batch y
    output = model(b_x)
    stu_logits = output[-1]           # cnn final output
    feature = torch.flatten(output[-2], start_dim=1)
    criterion = af.get_loss_criterion()
    ce_loss = criterion(stu_logits, b_y)
    group_onehot = F.one_hot(b_sensitive).float()
    hsic_loss = 0
    for l in range(args.class_num):
        mask = b_y == l
        if feature[mask].shape[0] == 0:
            continue
        if(torch.isnan(hsic.unbiased_estimator(feature[mask], group_onehot[mask])).any() or torch.isinf(hsic.unbiased_estimator(feature[mask], group_onehot[mask])).any()):
            continue
        hsic_loss += hsic.unbiased_estimator(feature[mask], group_onehot[mask])

    loss = ce_loss + labmda * hsic_loss
    
    optimizer.zero_grad()           # clear gradients for this training step
    # loss.backward()                 # backpropagation, compute gradients
    loss.mean().backward()
    optimizer.step()                # apply gradients
    # return loss.mean(), labmda * hsic_loss.mean(), ce_loss.mean()
    return loss, labmda * hsic_loss, ce_loss


def cnn_train(args, model, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path=None, models_path=None, labmda=0):
    writer_me = SummaryWriter(tensor_board_path)
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}
    hsic = RbfHSIC(1, 1)
    for epoch in range(1, epochs):
        loss = []
        ce_loss = []
        hsic_loss = []
        cur_lr = af.get_lr(optimizer)

        if not hasattr(model, 'augment_training') or model.augment_training:
            train_loader = data.aug_train_loader
        else:
            train_loader = data.train_loader

        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))
        for x, y, gender, idx in tqdm(train_loader):
            _loss, _hsic_loss, _ce_loss = cnn_training_step(args, model, optimizer, x, y, gender, device, labmda, hsic)
            loss.append(_loss.mean())
            ce_loss.append(_ce_loss.mean())
            hsic_loss.append(_hsic_loss)

        scheduler.step()

        print("Loss: {}".format(sum(loss) / len(loss)))
        writer_me.add_scalar("Total Loss/train", sum(loss) / len(loss), epoch)       
        print("CE Loss: {}".format(sum(ce_loss) / len(ce_loss)))
        writer_me.add_scalar("CE Loss/train", sum(ce_loss) / len(ce_loss), epoch)        
        print("HSIC Loss: {}".format(sum(hsic_loss) / len(hsic_loss)))
        writer_me.add_scalar("HSIC Loss/train", sum(hsic_loss) / len(hsic_loss), epoch)      
            
        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            print('Start Testing :')
            cnn_test(model, data.test_loader, device)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_hsic')
    parser.add_argument('--training_title', type=str, default='',
                    help='')
    parser.add_argument('--epochs', type=int, default=200,
                    help='')
    parser.add_argument('--lr', type=float, default=0.01,
                    help='')
    parser.add_argument('--batch_size', type=int, default=256,
                    help='')
    parser.add_argument('--dataset', type=str, default='isic2019',
                    help='')
    parser.add_argument('--model', type=str, default='resnet18',
                    help='')
    parser.add_argument('--Lambda', type=float, default=0.5,
                    help='')
    parser.add_argument('--class_num', type=int, default=8,
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

    print(model)
    
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    cnn_train(args, model, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path, args.Lambda)

