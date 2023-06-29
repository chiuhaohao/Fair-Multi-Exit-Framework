import time
import torch
import numpy as np
import torch.nn as nn
import aux_funcs as af
import torch.optim as optim
import matplotlib.pyplot as plt

from data import *
from tqdm import tqdm
from model import *
from util.bce_acc import *
from fairness_metric import *
from random import choice, shuffle


def val_accuracy(pred, ta, sa, ta_cls, sa_cls, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = ta.size(0)
        pred = pred.t()
        correct = pred.eq(ta.view(1, -1).expand_as(pred))
        group=[]
        group_num=[]
        for i in range(ta_cls):
            sa_group=[]
            sa_group_num=[]
            for j in range(sa_cls):
                eps=1e-8
                sa_group.append( ((sa==j)*(ta==i)*(correct==1)).float().sum() * (100 /(((sa==j)*(ta==i)).float().sum()+eps)) )
                sa_group_num.append(((sa==j)*(ta==i)).float().sum()+eps)
            group.append(sa_group)
            group_num.append(sa_group_num)
       
        res=(correct==1).float().sum()*(100.0 / batch_size)
        
        return res,group,group_num


def get_confidence(dataset_name='', logits=None):
    softmax = nn.functional.softmax(logits, dim=0)
    return torch.max(softmax).cpu().numpy()

def early_exit_inference(x, model):
    outputs = []
    confidences = []
    output_id = 0
    last_output_id = model.num_output-1
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            early_exit_out = early_exits_layer(x)
            outputs.append(early_exit_out)
            confidence = get_confidence(logits=early_exit_out[0])
            confidences.append(confidence)
            if confidence >= model.confidence_threshold:
                is_early = True
                return early_exit_out, output_id, is_early
            if output_id == last_output_id:
                is_early = False
                max_confidence_output = np.argmax(confidences)
                return outputs[max_confidence_output], output_id, is_early
            output_id += 1
                
def occam_early_exit_inference(x, model):
    output, gate_output = model(x)
    for idx, confidence in enumerate(gate_output):

        if confidence > 0.99:
            return output[idx]
    return output[3]

def occam_test_early_exits_fairness(model, loader, device='cpu', model_name='', dataset_name=''):
    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output
    total_time = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].cuda()
            b_y = batch[1].cuda()
            gender = batch[2].cuda()
            start_time = time.time()
            output = occam_early_exit_inference(b_x, model)
            _, pred = torch.max(output, 1)
            # print(pred)
            label_list.append(b_y.long().detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.long().cpu().numpy())
            end_time = time.time()
            total_time+= (end_time - start_time)

    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)
    return fairness_metrics, early_output_counts, non_conf_output_counts, total_time

def early_exit_confidence_resnet(model, x, confidence_threshold=None):
    confidences = []

    early_exits_outputs = []
    cnt = 0
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            unsqueeze_tensor = early_exits_layer(x)
            # print(unsqueeze_tensor.shape)
            squeeze_tensor = torch.squeeze(unsqueeze_tensor, -1)
            squeeze_tensor = torch.squeeze(squeeze_tensor, -1)
            early_exits_outputs.append(squeeze_tensor)
            # early_exits_outputs.append(early_exits_layer(x))
            softmax = nn.functional.softmax(squeeze_tensor[0], dim=0)
            # softmax = nn.functional.softmax(early_exits_layer(x)[0], dim=0)
            confidence = torch.max(softmax)
            confidences.append(confidence)
        
            if confidence >= confidence_threshold:
                return squeeze_tensor, cnt
                # return early_exits_layer(x), cnt
            cnt += 1

    final_out = model.g(torch.flatten(x, start_dim=1))
    early_exits_outputs.append(final_out) # append final out
    return final_out, cnt

def early_exit_each_ic_resnet(model, x, ic_idx=None):
    confidences = []

    early_exits_outputs = []
    cnt = 0
    for layer, early_exits_layer in model.f:
        x = layer(x)
        if early_exits_layer != None:
            early_exits_outputs.append(early_exits_layer(x))
            softmax = nn.functional.softmax(early_exits_layer(x)[0], dim=0)
            confidence = torch.max(softmax)
            confidences.append(confidence)
            if (cnt == ic_idx):
                return early_exits_layer(x), cnt
            cnt += 1

    final_out = model.g(torch.flatten(x, start_dim=1))
    early_exits_outputs.append(final_out) # append final out
    return final_out, cnt

def me_test_early_exits_fairness_original(model, loader, device='cpu', confidence_threshold=None):
    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    early_output_counts = [0] * model.num_output
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            gender = batch[2].to(device)
            output, ic_idx = early_exit_confidence_resnet(model, b_x, confidence_threshold)
            _, pred = torch.max(output, 1)
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.cpu().numpy())
            early_output_counts[ic_idx] += 1
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)

    return fairness_metrics, early_output_counts

def me_test_early_exits_fairness_each_ic(model, loader, device='cpu', ic_idx=None):
    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    early_output_counts = [0] * model.num_output
    with torch.no_grad():
        for batch in loader:
            b_x = batch[0].to(device)
            b_y = batch[1].to(device)
            gender = batch[2].to(device)
            output, ic_idx = early_exit_each_ic_resnet(model, b_x, ic_idx)
            _, pred = torch.max(output, 1)
            label_list.append(b_y.detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.cpu().numpy())
            early_output_counts[ic_idx] += 1
    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)

    return fairness_metrics, early_output_counts

def me_test_early_exits_fairness(model, loader, device='cpu', model_name='', dataset_name=''):
    # print(dataset_name)
    model.eval()
    gender_list = []
    label_list = []
    y_pred_list = []
    early_output_counts = [0] * model.num_output
    non_conf_output_counts = [0] * model.num_output
    total_time = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            b_x = batch[0].cuda()
            b_y = batch[1].cuda()
            gender = batch[2].cuda()
            start_time = time.time()
            output, output_id, is_early = early_exit_inference(b_x, model)
            pred = None
            _, pred = torch.max(output, 1)

            label_list.append(b_y.long().detach().cpu().numpy())
            y_pred_list.append(pred.detach().cpu().numpy())
            gender_list.append(gender.long().cpu().numpy())
            end_time = time.time()
            total_time+= (end_time - start_time)
            if is_early:
                early_output_counts[output_id] += 1
            else:
                non_conf_output_counts[output_id] += 1

    label_list = np.concatenate(label_list)
    y_pred_list = np.concatenate(y_pred_list)
    gender_list = np.concatenate(gender_list)
    fairness_metrics = compute_fairness_metrics(label_list, y_pred_list, gender_list)
    return fairness_metrics, early_output_counts, non_conf_output_counts, total_time

def eval_me_original(model, one_batch_dataset, model_name, dataset_name='', device=None):
    for threshold in [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]:
        fairness_metric, early_exit_counts = me_test_early_exits_fairness_original(model, one_batch_dataset.vali_loader, device, threshold)
        print('early_exit_counts: {}'.format(early_exit_counts))
        print("Threshold: {}".format(threshold))
        for k, v in fairness_metric.items():
            print('{}:{:.4f}'.format(k, v))

def eval_me_each_IC_original(model, one_batch_dataset, model_name, dataset_name='', device=None):
    for ic_idx in [0, 1, 2, 3, 4]:
        fairness_metric, early_exit_counts = me_test_early_exits_fairness_each_ic(model, one_batch_dataset.test_loader, device, ic_idx)
        print('early_exit_counts: {}'.format(early_exit_counts))
        print("IC idx: {}".format(ic_idx))
        for k, v in fairness_metric.items():
            print('{}:{:.4f}'.format(k, v))

def eval_me(model, one_batch_dataset, model_name, dataset_name=''):
    for threshold in [0.8, 0.85, 0.9, 0.95, 0.99, 0.999]:
        model.confidence_threshold = threshold
        fairness_metric, early_exit_counts, non_conf_exit_counts, total_time = me_test_early_exits_fairness(model, one_batch_dataset.test_loader, model_name=model_name, dataset_name=dataset_name)
        print('early_exit_counts: {}, non_conf_exit_counts: {}'.format(early_exit_counts, non_conf_exit_counts))
        print("Threshold: {}".format(threshold))
        for k, v in fairness_metric.items():
            print('{}:{:.4f}'.format(k, v))


if __name__ == '__main__':
    model_name = 'resnet'
    dataset_name = 'isic2019'
    random_seed = af.get_random_seed()
    af.set_random_seeds()
    device = af.get_pytorch_device()
    print('Random Seed: {}'.format(random_seed))

    af.create_path('outputs/20230310_eval_cnn_vgg_isic')
    af.set_logger('outputs/20230202_eval_cnn_vgg_isic/train_models{}'.format(af.get_random_seed()))

    load_path = '/home/jinghao/Fairness/Shallow-Deep-Networks/networks/121/220220601_Resnet18_me_cof/245.pth'
    model = torch.load(load_path)
    model.cuda()

    one_batch_dataset = ISIC2019(batch_size=1)
    eval_me_original(model, one_batch_dataset, model_name, dataset_name='')