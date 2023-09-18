import os
import torch
import torch as nn
from typing import List
from copy import deepcopy
from utils.misc import clean_state_dict
import numpy as np
##################################################################################


def convert_sync_batchnorm(module, process_group=None):
    # convert both BatchNorm and BatchNormAct layers to Synchronized variants
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):

        # convert standard BatchNorm layers
        module_output = torch.nn.SyncBatchNorm(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
            process_group,
            )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, convert_sync_batchnorm(child, process_group))
    del module
    return module_output

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, os.path.split(filename)[0] + '/model_best.pth.tar')
        # shutil.copyfile(filename, os.path.split(filename)[0] + '/model_best.pth.tar')


def kill_process(filename:str, holdpid:int) -> List[str]:
    import subprocess, signal
    res = subprocess.check_output("ps aux | grep {} | grep -v grep | awk '{{print $2}}'".format(filename), shell=True, cwd="./")
    res = res.decode('utf-8')
    idlist = [i.strip() for i in res.split('\n') if i != '']
    print("kill: {}".format(idlist))
    for idname in idlist:
        if idname != str(holdpid):
            os.kill(int(idname), signal.SIGKILL)
    return idlist


def cal_confounder(train_loader, model, args):
    model = model.eval()
    if 'res' in args.backbone:
        feature_map_size = args.img_size // 32
        cfer = torch.zeros((args.num_class, args.feat_dim, feature_map_size, feature_map_size))
    else:
        cfer = torch.zeros((args.num_class, args.feat_dim))
    
    num_class = torch.zeros((args.num_class))
    to_device = True
    for i, (inputData, target) in enumerate(train_loader):
        inputData = inputData.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if to_device == True:
            cfer = cfer.to(inputData.device)
            num_class = num_class.to(inputData.device)
            to_device = False

        feat, logits = model(inputData)
       
        target_nz = target.nonzero()
        for label_info in target_nz:
            batch_id, label = label_info
            batch_id, label = int(batch_id), int(label)
            cfer[label] += feat.data[batch_id]
            num_class[label] += 1
    if cfer.dim() > 2:
        cfer = cfer.flatten(2).mean(dim=-1)
    cfer = cfer / num_class[:, None]
    model = model.train()
    del train_loader
    return cfer

def model_transfer(model, tde_model, confounder, args, logger):

    tde_model_dict = tde_model.state_dict()
    model_dict = clean_state_dict(model.state_dict())

    for k, v in model_dict.items():
        if k in tde_model_dict and v.shape == tde_model_dict[k].shape:
            if "selector" in k or "logit_clf" in k:
                v.requires_grad = True
            tde_model_dict[k] = v
                # print(k, tde_model_dict[k].requires_grad)
        else:
            logger.info('\tMismatched layers: {}'.format(k))
    
    # tde_model_dict.update(model_dict)
    tde_model.load_state_dict(tde_model_dict, strict=False)

    del tde_model_dict
    del model_dict
    
    tde_model.clf.stagetwo = True
    tde_model.clf.memory.data = confounder

    # print(confounder.shape)
    # prototype_embed_path = '/media/data2/maleilei/MLIC/CCD_MLIC/vis_prototype_embed_file_select_0.8.npy'
    # prototype_embed_path = '/media/data2/maleilei/MLIC/CCD_MLIC/query_embed_in_forward.npy'
    # prototype_embed = np.load(prototype_embed_path)
    # prototype_embed = torch.from_numpy(prototype_embed)
    # tde_model.clf.memory.data = prototype_embed


    # for k, v in tde_model.named_parameters():
    #     if  "memory" in k or "selector" in k or "logit_clf" in k:
    #         print(k, v.requires_grad)
            # v.requires_grad = True

    # resume 'requires_grad'
    for k, v in tde_model.named_parameters():
        if  "selector" in k or "logit_clf" in k:
            v.requires_grad = True
        if  "memory" in k:
            v.requires_grad = False

    model = model.cpu()
    del model

    torch.cuda.empty_cache()
    return tde_model