from ast import arg
from tkinter.tix import Tree
import torch
import torch.nn as nn
from torch.nn import Module as Module
from collections import OrderedDict
import torchvision
import math
import numpy as np
import torch.nn.functional as F

try:
    from .classifier import Interventional_Classifier, CosNorm_Classifier
    from .swin_transformer import SwinTransformer
    from .vision_transformer import VisionTransformer
except ImportError:
    from classifier import Interventional_Classifier, CosNorm_Classifier
    from swin_transformer import SwinTransformer
    from vision_transformer import VisionTransformer

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class CCD(Module):
    def __init__(self,backbone="resnet101",num_classes=80,pretrain=None,use_intervention=False,use_tde=False,feat_fuse='none', stagetwo=False, stageone=True):
        super(CCD,self).__init__()
        if backbone=="resnet101":
            self.backbone = resnet101_backbone(pretrain)
        elif backbone=="swim_transformer":
            self.backbone = swimtrans_backbone(num_classes,pretrain)
        elif backbone=="swim_transformer_large":
            self.backbone = swimtrans_backbone(num_classes,pretrain,large = True)
        elif backbone=="vit":
            self.backbone = Vit_backbone(num_classes,pretrain)

        self.feat_dim = self.backbone.feat_dim
        self.stagetwo = stagetwo
        self.stageone = stageone

        if use_tde:
            self.clf = tde_classifier(num_classes,self.feat_dim, use_intervention, stagetwo=self.stagetwo, feat_fuse=feat_fuse, stageone=self.stageone)
        else:
            if not use_intervention:
                self.clf = nn.Linear(self.feat_dim, num_classes)
            else:
                self.clf = Interventional_Classifier(num_classes=num_classes, feat_dim=self.feat_dim, num_head=4, tau=32.0, beta=0.03125)
    def forward(self,x):
        feats = self.backbone(x)# torch.Size([4, 2048, 14, 14])

        # if len(list(feats.size())) == 4:
        #     feats = feats.flatten(2).mean(dim=-1)

        logits = self.clf(feats)

        return feats, logits

 
class resnet101_backbone(Module):
    def __init__(self, pretrain=True):
        super(resnet101_backbone,self).__init__()

        res101 = torchvision.models.resnet101(pretrained=pretrain)

        train_backbone = True
        for name, parameter in res101.named_parameters():
            if not train_backbone or 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        numFit = res101.fc.in_features # 2048
        self.resnet_layer = nn.Sequential(*list(res101.children())[:-2])
   
        self.feat_dim = numFit

    def forward(self,x):
        feats = self.resnet_layer(x)
       
        return feats

class swimtrans_backbone(Module):
    def __init__(self,num_classes,pretrain,large=False):
        super(swimtrans_backbone,self).__init__()
        if large:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=192,depths=(2, 2, 18, 2),num_heads=(6, 12, 24, 48),window_size=12)
        else:
            self.model = SwinTransformer(img_size=384,patch_size=4,num_classes=num_classes,embed_dim=128,depths=(2, 2, 18, 2),num_heads=(4, 8, 16, 32),window_size=12)
        if pretrain:
            path = pretrain
            state = torch.load(path, map_location='cpu')['model']
            filtered_dict = {k: v for k, v in state.items() if(k in self.model.state_dict() and 'head' not in k)}
            self.model.load_state_dict(filtered_dict,strict=False)
        numFit = self.model.num_features
        self.feat_dim = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

class Vit_backbone(Module):
    def __init__(self,num_classes,pretrain):
        super(Vit_backbone,self).__init__()
        
        self.model = VisionTransformer(img_size=224, patch_size=16,num_classes=num_classes,embed_dim=1024,depth=24,num_heads=16,representation_size=1024)

        if pretrain:
            path = pretrain
            self.model.load_pretrained(path)
        numFit = self.model.num_features
        self.feat_dim = numFit
        del self.model.head

    def forward(self,x):
        feats = self.model.forward_features(x)
        return feats

class tde_classifier(Module):
    def __init__(self,num_classes, feat_dim, use_intervention, stagetwo=False, feat_fuse='selector', stageone=True):
        super(tde_classifier,self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.memory = nn.Parameter(torch.zeros((num_classes, feat_dim)), requires_grad=False)
        # self.memory = None
        self.feat_fuse = feat_fuse
        self.stagetwo = stagetwo
        self.stageone = stageone

        if feat_fuse=='selector':
            self.selector = nn.Linear(self.feat_dim,self.feat_dim)
        elif feat_fuse=='mlp':
            self.selector = MLP(input_dim=self.feat_dim, hidden_dim=self.feat_dim, output_dim=self.feat_dim, num_layers=2)
        elif feat_fuse=='concat':
            self.selector = MLP(input_dim=2*self.feat_dim, hidden_dim=self.feat_dim, output_dim=self.feat_dim, num_layers=2)
        
        self.to_device = True

        if use_intervention:
            self.context_clf = Interventional_Classifier(num_classes=num_classes, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
            self.logit_clf = Interventional_Classifier(num_classes=num_classes, feat_dim=feat_dim, num_head=4, tau=32.0, beta=0.03125)
        else:
            self.context_clf = nn.Linear(feat_dim,num_classes)
            self.logit_clf = nn.Linear(feat_dim,num_classes)
        self.softmax = nn.Softmax(dim=1) 
        
    def forward(self, feats):
        if len(list(feats.size())) == 2:
            global_feat = feats
            memory = self.memory
        else:
            global_feat = feats.flatten(2).mean(dim=-1) # (BS, feat_dim)
            feats = feats.flatten(2).max(dim=-1)[0] # (BS, feat_dim)
            memory = self.memory #(num_classes, feat_dim)

        if self.stagetwo:
            if self.to_device and memory is not None:
                memory = memory.to(feats.device)
                self.to_device = False

            pre_logits = self.softmax(self.context_clf(global_feat)) # [BS, 80]
            memory_feature = torch.mm(pre_logits, memory) # [BS, 80] x [80, 2048] => [BS, 2048]

            if self.feat_fuse == 'none':
                combine_feature = feats - memory_feature
            elif self.feat_fuse == 'selector':
                selector = self.selector(feats.clone()) # [BS, 2048] => [BS, 2048]
                selector = selector.tanh() # [BS, 2048]
                combine_feature = feats - selector * memory_feature # [BS, 2048]
            elif self.feat_fuse == 'mlp':
                combine_feature = feats - self.selector(memory_feature.clone())
            elif self.feat_fuse == 'concat':
                selector = torch.cat((feats, memory_feature.clone()), 1)
                combine_feature = feats - self.selector(selector)

            logits = self.logit_clf(combine_feature)
        else:
            logits = self.context_clf(global_feat)
        return logits


def build_ccd(args):

    model = CCD(
        backbone=args.backbone,
        num_classes=args.num_class,
        pretrain=args.pretrained,
        use_intervention=args.use_intervention,
        use_tde=args.use_tde,
        feat_fuse=args.feat_fuse,
        stagetwo=args.stagetwo,
        stageone = args.stageone
    )
    # clf.memory             # two
    # clf.selector.weight    # two
    # clf.selector.bias      # two
    # clf.context_clf.weight # one
    # clf.logit_clf.weight   # two

    # if args.stageone is True:
    #     print(" Frozen 'memory', 'selector' and 'logit_clf' ")
    #     for k, v in model.named_parameters():
    #         if "memory" in k or "selector" in k or "logit_clf" in k:
    #             v.requires_grad = False
    
    # if args.stageone is False:
    #     print(" Frozen 'memory', 'selector' and 'logit_clf' ")


    return model

if __name__ == '__main__':
    from classifier import Interventional_Classifier, CosNorm_Classifier
    from swin_transformer import SwinTransformer
    from vision_transformer import VisionTransformer
    stagetwo = True
    stageone = True
    # model = CCD(backbone='resnet101', num_classes=80, pretrain='', use_intervention=True, use_tde=True, feat_fuse='selector',stagetwo=stagetwo, stageone=True)
    model = CCD(backbone='resnet101', num_classes=80, pretrain='', use_intervention=True, use_tde=True, feat_fuse='concat',stagetwo=stagetwo, stageone=True)
    # print(model)
    # for k, v in model.state_dict().items():
    #     print(k)
    model.cuda().eval()
    image = torch.randn(4, 3, 448, 448)
    image = image.cuda()
    output1, output2 = model(image)


    # resume = '/media/data2/maleilei/MLIC/CCD_MLIC/output/CCD_ResNet101_448_COCO/bs128_base_83.78/model_best.pth.tar'
    # checkpoint = torch.load(resume, map_location='cpu')
    # # state_dict = clean_state_dict(checkpoint['state_dict'])
    # state_dict = checkpoint['state_dict']
    # for k, v in state_dict.items():
    #     if 'memory' in k: 
    #         # print(k)
    #         np.save('ccd_memory_embed_file.npy', v.cpu().detach().numpy())

    # backbone.resnet_layer.7.2.bn3.running_var
    # backbone.resnet_layer.7.2.bn3.num_batches_tracked
    # clf.memory
    # clf.selector.weight
    # clf.selector.bias
    # clf.context_clf.weight
    # clf.logit_clf.weight

    # backbone.resnet_layer.7.2.bn3.running_var
    # backbone.resnet_layer.7.2.bn3.num_batches_tracked
    # clf.context_clf.weight


        # name = 'resnet101'
        # dilation = False
        # res101 = getattr(torchvision.models, name)(
        #         replace_stride_with_dilation=[False, False, dilation],
        #         pretrained=True,
        #         norm_layer=FrozenBatchNorm2d)
        # train_backbone = True
        # for name, parameter in res101.named_parameters():
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        # train_backbone = True
        # for name, parameter in res101.named_parameters():
        #     if not train_backbone or 'layer1' not in name and 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)

        # if pretrain:
        #     path = pretrain
        #     state = torch.load(path, map_location='cpu')
        #     if type(state)==dict and "state_dict" in state:
        #         res101 = nn.DataParallel(res101)
        #         res101.load_state_dict(state["state_dict"])
        #         res101 = res101.module
        #     else:
        #         res101.load_state_dict(state)




























































