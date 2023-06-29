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
from util.metric import *
from fairness_metric import *
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

def cnn_training_step(args, model, optimizer, data, labels, sensitive_attr, device):
    model.train()
    b_x = data.to(device)          # batch x
    b_y = labels.to(device)        # batch y
    output = model(b_x)
    loss = 0
    criterion = af.get_loss_criterion()
    for ic in range(model.num_output):
        if(args.model == "me_resnet18"):
            stu_logits = output[ic]           # cnn final output
            ce_loss = criterion(stu_logits, b_y)
            loss += ce_loss
        if(args.model == "me_vgg11"):
            coeff_list = [0.3, 0.45, 0.6, 0.75, 0.9]
            loss += coeff_list[ic]*criterion(output[ic], b_y)
    optimizer.zero_grad()           # clear gradients for this training step
    loss.mean().backward()                 # backpropagation, compute gradients
    optimizer.step()                # apply gradients
    return loss.mean()


def cnn_train(args, model, one_batch_dataset, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path=None, models_path=None):
    writer_me = SummaryWriter(tensor_board_path)
    metrics = {'epoch_times':[], 'test_top1_acc':[], 'test_top5_acc':[], 'train_top1_acc':[], 'train_top5_acc':[], 'lrs':[]}

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
            _loss = cnn_training_step(args, model, optimizer, x, y, gender, device)
        loss.append(_loss)
        scheduler.step()

        print("Loss: {}".format(sum(loss) / len(loss)))
        writer_me.add_scalar("Total Loss/train", sum(loss) / len(loss), epoch) 

        # eval_me_original(model, one_batch_dataset, '', '', device)
        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            eval_me_original(model, one_batch_dataset, '', '', device)
            

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_me_cnn')
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
    parser.add_argument('--model', type=str, default='me_resnet18',
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
    cnn_train(args, model, one_batch_dataset, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path)

