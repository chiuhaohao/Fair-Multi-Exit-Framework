import torch
import time
import data
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import aux_funcs as af

from util.custom_loss import *
from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from sklearn import metrics
import torch.nn.functional as F
from fairness_metric import *
from torchvision import models
from datetime import datetime
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

def cnn_test(model, loader, device='cpu'):

    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            gender  = batch[2].to(device)
            output, feature = model(b_x)
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

def cnn_train(model, data, epochs, optimizer, scheduler, device='cuda', tensor_board_path='', models_path='', args=None):
    if(not args.train_teacher):
        teacher_model = torch.load(args.teacher_path) 
        teacher_model = teacher_model.to(device)
    
    writer = SummaryWriter(tensor_board_path)
    mmd_loss = MMDLoss(w_m=args.Lambda, sigma=args.sigma, num_classes=args.class_num, num_groups=args.group_num, kernel='rbf')
    for epoch in range(1, epochs): 
        CE_loss = [] 
        MMD_loss = []
        total_loss = []
        label_list = []
        y_pred_list = []
        sensitive_group_list = []
        cur_lr = af.get_lr(optimizer)
        train_loader = data.train_loader

        start_time = time.time()
        model.train()
        print('Epoch: {}/{}'.format(epoch, epochs))
        print('Cur lr: {}'.format(cur_lr))

        for x, y, sensitive_group, idx in tqdm(train_loader):
            b_x = x.to(device)   # batch x
            b_y = y.to(device)   # batch y
            b_sensitive_group = sensitive_group.to(device)
            output, feature = model(b_x)  # cnn final output
            if(not args.train_teacher):
                teacher_output, teacher_feature = teacher_model(b_x)
            # teacher_feature = torch.squeeze(teacher_feature[-2], 2)
            _, preds = torch.max(output, 1) 
            
            criterion = af.get_loss_criterion('')
            if(not args.train_teacher):
                mmd = mmd_loss.forward(feature, teacher_feature, b_sensitive_group, b_y)
                ce_loss = criterion(output, b_y)
                if(torch.isnan(mmd).any() or torch.isnan(ce_loss).any()):
                    continue
                stage_loss = ce_loss + args.alpha * mmd
                optimizer.zero_grad()           # clear gradients for this training step
                stage_loss.mean().backward()           # backpropagation, compute gradients
                optimizer.step()   

                CE_loss.append(ce_loss.mean())
                MMD_loss.append(mmd.mean())
                total_loss.append(stage_loss.mean())
                label_list.append(b_y.detach().cpu().numpy())
                y_pred_list.append(preds.detach().cpu().numpy())
                sensitive_group_list.append(sensitive_group.numpy())
            else:
                ce_loss = criterion(output, b_y)
                if(torch.isnan(ce_loss).any()):
                    continue
                stage_loss = ce_loss 
                optimizer.zero_grad()           # clear gradients for this training step
                stage_loss.mean().backward()    # backpropagation, compute gradients
                optimizer.step()   

                CE_loss.append(ce_loss.mean())
                total_loss.append(stage_loss.mean())
                label_list.append(b_y.detach().cpu().numpy())
                y_pred_list.append(preds.detach().cpu().numpy())
                sensitive_group_list.append(sensitive_group.numpy())

        scheduler.step() 
        end_time = time.time()

        epoch_time = int(end_time-start_time)

        print('Total Loss: {}'.format(sum(total_loss) / len(total_loss)))
        if(not args.train_teacher):
            print('MMD Loss: {}'.format(sum(MMD_loss) / len(MMD_loss)))
        print('CE Loss: {}'.format(sum(CE_loss) / len(CE_loss)))        
        print('Epoch took {} seconds.'.format(epoch_time))
        writer.add_scalar('Total Loss:', sum(total_loss) / len(total_loss), epoch)
        if(not args.train_teacher):
            writer.add_scalar('MMD Loss:', sum(MMD_loss) / len(MMD_loss), epoch)
        writer.add_scalar('CE Loss: ', sum(CE_loss) / len(CE_loss), epoch)
        writer.add_scalar("Lr/train", cur_lr, epoch)
        print('Start testing...')
        cnn_test(model, data.test_loader, device)

        if epoch % 5 == 0:
            torch.save(model, '{}/{}.pth'.format(models_path, epoch))
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train_mfd')
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
    parser.add_argument('--model', type=str, default='mfd_resnet18',
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
    parser.add_argument('--teacher_path', type=str, default='',
                    help='')
    parser.add_argument('--train_teacher', type=bool, default=False,
                    help='')
    args = parser.parse_args()
    training_title = args.training_title
    print(training_title)
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

    model = model_handler(args.model, args.class_num)
    model.to(device)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-5, momentum=0.9, nesterov=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # print(model)
    ds_handler = dataset_handler(args)
    dataset = ds_handler.get_dataset()
    one_batch_dataset = ds_handler.get_dataset(is_one_batch=True)

    cnn_train(model, dataset, args.epochs, optimizer, scheduler, device, tensor_board_path, models_path, args=args)