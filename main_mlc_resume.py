import math
import os, sys
import random
import time
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.optim import lr_scheduler
from config import parser_args
from train import train
from validate import validate
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from dataset.get_dataset import get_datasets

from utils.logger import setup_logger
import models
from models.aslloss import create_loss
from models.build import build_ccd
from utils.misc import clean_state_dict, setup_for_distributed
from utils.slconfig import get_raw_dict
from utils.util import add_weight_decay, ModelEma, save_checkpoint, kill_process,  \
    cal_confounder, model_transfer, convert_sync_batchnorm
from utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter

def get_args():
    args = parser_args()
    return args

best_mAP = 0

def main():
    global best_mAP
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    
    if 'WORLD_SIZE' in os.environ:
        assert args.world_size > 0, 'please set --world-size and --rank in the command line'
        # launch by torch.distributed.launch
        local_world_size = int(os.environ['WORLD_SIZE'])
        args.world_size = args.world_size * local_world_size
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.rank = args.rank * local_world_size + args.local_rank
        print('world size: {}, world rank: {}, local rank: {}'.format(args.world_size, args.rank, args.local_rank))
        print('os.environ:', os.environ)
    else:
        # single process, useful for debugging
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0

    if args.seed is None:
        args.seed = random.randint(1, 10000)
        print("Random Seed: {}".format(args.seed))
        
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    torch.cuda.set_device(args.local_rank)
    print('| distributed init (local_rank {}): {}'.format(
        args.local_rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend='nccl', init_method=args.dist_url, 
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True
    setup_for_distributed(args.rank == 0)

    os.makedirs(args.output, exist_ok=True)
    logger = setup_logger(output=args.output, distributed_rank=dist.get_rank(), color=False, name="Q2L")
    logger.info("Command: "+' '.join(sys.argv))
    if dist.get_rank() == 0:
        path = os.path.join(args.output, "config.json")
        with open(path, 'w') as f:
            json.dump(get_raw_dict(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

    logger.info('world size: {}'.format(dist.get_world_size()))
    logger.info('dist.get_rank(): {}'.format(dist.get_rank()))
    logger.info('local_rank: {}'.format(args.local_rank))

    # prepare data
    model, train_loader, val_loader, train_sampler = main_prepare(args, logger)

    model = main_worker(args, logger, model, train_loader, val_loader, train_sampler, best_mAP)
    # if args.use_tde:
    #     logger.info('building confounder......')
    #     confounder = cal_confounder(train_loader_cfer, model, args)
    #     args.stageone=False
    #     args.stagetwo=True
    #     # args.loss = 'focal'
    #     tde_model = build_ccd(args)
    #     tde_model = model_transfer(model, tde_model, confounder, args, logger)
    #     args.stage = 2
    #     tde_model = tde_model.cuda()
    #     torch.cuda.empty_cache()
    #     if args.SyncBatchNorm:
    #         tde_model = convert_sync_batchnorm(tde_model)
    #     tde_model = torch.nn.parallel.DistributedDataParallel(tde_model, device_ids=[args.local_rank], broadcast_buffers=False)
    #     return main_worker(args, logger, tde_model, train_loader, val_loader, train_sampler, best_mAP)

def main_prepare(args, logger):

    # build model
    args.stage = 2
    args.stageone=False
    args.stagetwo=True
    model = build_ccd(args)
 
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device(dist.get_rank()))

            if 'state_dict' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
            elif 'model' in checkpoint:
                state_dict = clean_state_dict(checkpoint['model'])
            else:
                raise ValueError("No model or state_dicr Found!!!")
            logger.info("Omitting {}".format(args.resume_omit))
            # import ipdb; ipdb.set_trace()

            # for k,v in state_dict.items():
            #     print(k)

            for omit_name in args.resume_omit:
                del state_dict[omit_name]
            model.load_state_dict(state_dict, strict=False)
            del checkpoint
            del state_dict
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    prototype_embed_path = '/media/data2/maleilei/MLIC/CCD_MLIC/ccd_memory_embed_file.npy'
    # prototype_embed_path = '/media/data2/maleilei/MLIC/CCD_MLIC/query_embed_in_forward.npy'
    prototype_embed = np.load(prototype_embed_path)
    prototype_embed = torch.from_numpy(prototype_embed)
    model.clf.memory.data = prototype_embed


    model = model.cuda()
    if args.SyncBatchNorm:
        model = convert_sync_batchnorm(model)
    torch.cuda.empty_cache()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], broadcast_buffers=False)

    # for k,v in model.named_parameters():
    #     print(k)
    # print('===================================================')
    

    # Data loading code
    train_dataset, val_dataset = get_datasets(args)
    assert args.batch_size // dist.get_world_size() == args.batch_size / dist.get_world_size(), 'Batch size is not divisible by num of gpus.'
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
   
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    return model, train_loader, val_loader, train_sampler


def main_worker(args, logger, model, train_loader, val_loader, train_sampler, best_mAP):

    torch.cuda.empty_cache()
    ema_m = ModelEma(model, args.ema_decay) # 0.9997^641 = 0.82503551031 

    # criterion
    criterion = create_loss(args)

    # optimizer
    args.lr_mult = args.batch_size / 128
    print("lr: {}".format(args.lr_mult * args.lr))
    if args.optim == 'AdamW':
        param_dicts = [
            {"params": [p for n, p in model.module.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, args.optim)(
            param_dicts,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )
    elif args.optim == 'Adam_twd':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(
            parameters,
            args.lr_mult * args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=0
        )          
    else:
        raise NotImplementedError


    # tensorboard
    if dist.get_rank() == 0:
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    if args.evaluate:
        _, mAP = validate(val_loader, model, criterion, args, logger)

        logger.info(' * mAP {mAP:.5f}'.format(mAP=mAP))
        return

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    losses = AverageMeter('Loss', ':5.3f', val_only=True)
    losses_ema = AverageMeter('Loss_ema', ':5.3f', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, losses, mAPs, losses_ema, mAPs_ema],
        prefix='=> Test Epoch: ')

    # one cycle learning rate
    args.lr_mult = args.batch_size / 128
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr * args.lr_mult, steps_per_epoch=len(train_loader), epochs=args.epochs, pct_start=0.2)

    end = time.time()
    best_mAP = 0
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        if args.ema_epoch == epoch:
            ema_m = ModelEma(model.module, args.ema_decay)
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        # train for one epoch
        loss = train(train_loader, model, ema_m, criterion, optimizer, scheduler, epoch, args, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('train_loss', loss, epoch)
            summary_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        if epoch % args.val_interval == 0:

            # evaluate on validation set
            loss, mAP = validate(val_loader, model, criterion, args, logger)
            loss_ema, mAP_ema = validate(val_loader, ema_m.module, criterion, args, logger)
            losses.update(loss)
            mAPs.update(mAP)
            losses_ema.update(loss_ema)
            mAPs_ema.update(mAP_ema)
            epoch_time.update(time.time() - end)
            end = time.time()
            eta.update(epoch_time.avg * (args.epochs - epoch - 1))

            regular_mAP_list.append(mAP)
            ema_mAP_list.append(mAP_ema)

            progress.display(epoch, logger)

            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('val_loss', loss, epoch)
                summary_writer.add_scalar('val_mAP', mAP, epoch)
                summary_writer.add_scalar('val_loss_ema', loss_ema, epoch)
                summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)

            # remember best (regular) mAP and corresponding epochs
            if mAP > best_regular_mAP:
                best_regular_mAP = max(best_regular_mAP, mAP)
                best_regular_epoch = epoch
            if mAP_ema > best_ema_mAP:
                best_ema_mAP = max(mAP_ema, best_ema_mAP)
            
            if mAP_ema > mAP:
                mAP = mAP_ema
                state_dict = ema_m.module.state_dict()
            else:
                state_dict = model.state_dict()
            is_best = mAP > best_mAP
            if is_best:
                best_epoch = epoch
            best_mAP = max(mAP, best_mAP)

            logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
            logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

            if dist.get_rank() == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': state_dict,
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint.pth.tar'))
            # filename=os.path.join(args.output, 'checkpoint_{:04d}.pth.tar'.format(epoch))

            if math.isnan(loss) or math.isnan(loss_ema):
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'best_mAP': best_mAP,
                    'optimizer' : optimizer.state_dict(),
                }, is_best=is_best, filename=os.path.join(args.output, 'checkpoint_nan.pth.tar'))
                logger.info('Loss is NaN, break')
                sys.exit(1)

            # early stop
            if args.early_stop:
                if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 8:
                    if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                        logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                        if dist.get_rank() == 0 and args.kill_stop:
                            filename = sys.argv[0].split(' ')[0].strip()
                            killedlist = kill_process(filename, os.getpid())
                            logger.info("Kill all process of {}: ".format(filename) + " ".join(killedlist)) 
                        break
            
            if args.stage==1 and epoch==args.stop_epoch and args.use_tde:
                return model

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()
    
    return 0

if __name__ == '__main__':
    main()
