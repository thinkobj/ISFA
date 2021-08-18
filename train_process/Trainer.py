from datetime import datetime
import os

import timeit
from torchvision.utils import make_grid
import time
import numpy as np
import pytz
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import tqdm
import socket
from utils.Utils import *
from utils.metrics import DiceLoss, pixel_acc
from utils.metric_1 import Dice_Smooth_loss

bceloss = torch.nn.BCELoss()
mseloss = torch.nn.MSELoss()
my_loss = torch.nn.BCEWithLogitsLoss()
L1_loss = torch.nn.L1Loss()


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):

    def __init__(self, cuda, model_gen, edge_dis, mask_dis, optimizer_gen, optimizer_edge, optimizer_mask,
                 val_loader, domain_loaderS, domain_loaderS2T, domain_loaderT, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8, warmup_epoch=10):
        self.cuda = cuda
        self.warmup_epoch = warmup_epoch
        self.model_gen = model_gen
        self.mask_dis = mask_dis
        self.edge_dis = edge_dis
        self.optim_gen = optimizer_gen
        self.optim_edge = optimizer_edge
        self.optim_mask = optimizer_mask
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.val_loader = val_loader
        self.domain_loaderS = domain_loaderS
        self.domain_loaderS2T = domain_loaderS2T
        self.domain_loaderT = domain_loaderT
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = datetime.now(pytz.timezone(self.time_zone))

        if interval_validate is None:
            self.interval_validate = int(10)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss_seg',
            'train/cup_dice',
            'train/disc_dice',
            'train/loss_adv',
            'train/loss_D_same',
            'train/loss_D_diff',
            'valid/loss_CE',
            'valid/cup_dice',
            'valid/disc_dice',
            'elapsed_time',
        ]

        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1

    def validate(self, epoch):
        training = self.model_gen.training
        self.model_gen.eval()

        val_loss = 0
        val_cup_dice = 0
        val_disc_dice = 0
        val_cup_pa = 0
        val_disc_pa = 0
        val_cup_iou = 0
        val_disc_iou = 0
        metrics = []
        with torch.no_grad():

            for batch_idx, sample in tqdm.tqdm(
                    enumerate(self.val_loader), total=len(self.val_loader),
                    desc='Valid iteration=%d' % self.iteration, ncols=80,
                    leave=False):
                data = sample['image']
                target_map = sample['map']
                target_boundary = sample['boundary']
                img_name = sample['img_name']
                if self.cuda:
                    data, target_map, target_boundary = data.cuda(), target_map.cuda(), target_boundary.cuda()
                with torch.no_grad():
                    content, style, predictions, boundary = self.model_gen(data)

                loss = F.binary_cross_entropy_with_logits(predictions, target_map)
                loss_data = loss.data.item()
                if np.isnan(loss_data):
                    raise ValueError('loss is nan while validating')
                val_loss += loss_data

                dice_cup, dice_disc = dice_coeff_2label(predictions, target_map)

                PA_cup, PA_disc, IOU_cup, IOU_disc = pixel_acc(predictions, target_map)

                val_cup_dice += dice_cup
                val_disc_dice += dice_disc

                val_cup_pa += PA_cup
                val_disc_pa += PA_disc

                val_cup_iou += IOU_cup
                val_disc_iou += IOU_disc

            val_loss /= len(self.val_loader)
            val_cup_dice /= len(self.val_loader)
            val_disc_dice /= len(self.val_loader)
            val_disc_pa /= len(self.val_loader)
            val_cup_pa /= len(self.val_loader)
            val_cup_iou /= len(self.val_loader)
            val_disc_iou /= len(self.val_loader)

            self.writer.add_scalar('val_data/val_CUP_PA', val_cup_pa, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_PA', val_disc_pa, self.epoch * (len(self.domain_loaderS)))

            self.writer.add_scalar('val_data/val_CUP_IOU', val_cup_iou, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_IOU', val_disc_iou, self.epoch * (len(self.domain_loaderS)))

            metrics.append((val_loss, val_cup_dice, val_disc_dice))
            self.writer.add_scalar('val_data/loss_CE', val_loss, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_CUP_dice', val_cup_dice, self.epoch * (len(self.domain_loaderS)))
            self.writer.add_scalar('val_data/val_DISC_dice', val_disc_dice, self.epoch * (len(self.domain_loaderS)))

            mean_dice = val_cup_dice + val_disc_dice
            is_best = mean_dice > self.best_mean_dice
            if is_best:
                self.best_epoch = self.epoch + 1
                self.best_mean_dice = mean_dice

                torch.save({
                    'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_edge_state_dict': self.optim_edge.state_dict(),
                    'optim_mask_state_dict': self.optim_mask.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'edge_dis_state_dict': self.edge_dis.state_dict(),
                    'mask_dis_state_dict': self.mask_dis.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_edge': get_lr(self.optim_edge),
                    'learning_rate_mask': get_lr(self.optim_mask),
                    'best_mean_dice': self.best_mean_dice,
                }, osp.join(self.out, 'checkpoint_%d,Disc_cup:%.4f,Disc_disc:%.4f,PA_cup:%.4f,PA_disc:%.4f, IOU_cup:%.4f,IOU_disc:%.4f.pth.tar'
                            % (self.best_epoch, val_cup_dice, val_disc_dice, val_cup_pa, val_disc_pa, val_cup_iou, val_disc_iou)))
            else:
                if (self.epoch + 1) % 100 == 0:
                    torch.save({
                        'epoch': self.epoch,
                    'iteration': self.iteration,
                    'arch': self.model_gen.__class__.__name__,
                    'optim_state_dict': self.optim_gen.state_dict(),
                    'optim_edge_state_dict': self.optim_edge.state_dict(),
                    'optim_mask_state_dict': self.optim_mask.state_dict(),
                    'model_state_dict': self.model_gen.state_dict(),
                    'edge_dis_state_dict': self.edge_dis.state_dict(),
                    'mask_dis2_state_dict': self.mask_dis.state_dict(),
                    'learning_rate_gen': get_lr(self.optim_gen),
                    'learning_rate_edge': get_lr(self.optim_edge),
                    'learning_rate_mask': get_lr(self.optim_mask),
                    'best_mean_dice': self.best_mean_dice,
                    }, osp.join(self.out, 'checkpoint_%d.pth.tar' % (self.epoch + 1)))


            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.now(pytz.timezone(self.time_zone)) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [''] * 5 + \
                       list(metrics) + [elapsed_time] + ['best model epoch: %d' % self.best_epoch]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            self.writer.add_scalar('best_model_epoch', self.best_epoch, self.epoch * (len(self.domain_loaderS)))
            if training:
                self.model_gen.train()
                self.edge_dis.train()
                self.mask_dis.train()


    def train_epoch(self):
        source_domain_label = 1
        target_domain_label = 0
        self.model_gen.train()
        self.edge_dis.train()
        self.mask_dis.train()
        self.running_seg_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_dis_diff_loss = 0.0
        self.running_dis_same_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0

        domain_t_loader = enumerate(self.domain_loaderT)
        domain_s2t_loader = enumerate(self.domain_loaderS2T)

        start_time = timeit.default_timer()
        for batch_idx, sampleS in tqdm.tqdm(
                enumerate(self.domain_loaderS), total=len(self.domain_loaderS),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):

            iteration = batch_idx + self.epoch * len(self.domain_loaderS)
            self.iteration = iteration

            assert self.model_gen.training
            assert self.edge_dis.training
            assert self.mask_dis.training

            self.optim_gen.zero_grad()
            self.optim_edge.zero_grad()
            self.optim_mask.zero_grad()

            # 1. train generator with source images
            for param in self.edge_dis.parameters():
                param.requires_grad = False
            for param in self.mask_dis.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True

            imageS = sampleS['image'].cuda()
            target_map = sampleS['map'].cuda()
            target_boundary = sampleS['boundary'].cuda()

            S_content, S_style, oS, boundaryS = self.model_gen(imageS)
            oS = F.interpolate(oS, size=(target_map.shape[2], target_map.shape[3]), mode='bilinear')
            boundaryS = F.interpolate(boundaryS, size=(target_boundary.shape[2], target_boundary.shape[3]), mode='bilinear')
            loss_seg1 = DiceLoss(torch.sigmoid(oS), target_map)
            loss_seg2 = mseloss(torch.sigmoid(boundaryS), target_boundary)
            loss_seg = loss_seg1 + loss_seg2

            self.running_seg_loss += loss_seg.item()
            loss_seg_data = loss_seg.data.item()
            if np.isnan(loss_seg_data):
                raise ValueError('loss is nan while training')

            loss_seg.backward()

            if self.epoch > self.warmup_epoch:
                # # 2. train generator with images from target domain
                try:
                    id_, sampleT = next(domain_t_loader)
                except:
                    domain_t_loader = enumerate(self.domain_loaderT)
                    id_, sampleT = next(domain_t_loader)

                imageT = sampleT['image'].cuda()

                T_conent, T_style, oT, boundaryT = self.model_gen(imageT)

                D_out2 = self.edge_dis(torch.sigmoid(boundaryT))
                D_out1 = self.mask_dis(oT)

                loss_adv_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())
                loss_adv_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(source_domain_label).cuda())
                loss_adv_diff = 0.01 * (loss_adv_diff1 + loss_adv_diff2)
                self.running_adv_diff_loss += loss_adv_diff.item()
                loss_adv_diff_data = loss_adv_diff.data.item()
                loss_adv_diff.backward()

                # # 3. train generator with images from generated domain
                try:
                    id_, sampleS2T = next(domain_s2t_loader)
                except:
                    domain_t_loader = enumerate(self.domain_loaderS2T)
                    id_, sampleS2T = next(domain_s2t_loader)

                imageS2T = sampleS2T['image'].cuda()
                S2T_content, S2T_style, oS2T, boundaryS2T = self.model_gen(imageS2T)
                loss1 = L1_loss(S_content, S2T_content) + L1_loss(T_style, S2T_style)

                D_gen_out2 = self.edge_dis(torch.sigmoid(boundaryS2T))
                D_gen_out1 = self.mask_dis(oS2T)
                loss_adv_1 = F.binary_cross_entropy_with_logits(D_gen_out1, torch.FloatTensor(D_gen_out1.data.size()).fill_(source_domain_label).cuda())
                loss_adv_2 = F.binary_cross_entropy_with_logits(D_gen_out2, torch.FloatTensor(D_gen_out2.data.size()).fill_(source_domain_label).cuda())
                loss_adv = 0.01 * (loss_adv_1 + loss_adv_2)
                self.running_adv_diff_loss += loss_adv.item()
                loss_adv_diff_data += loss_adv.data.item()
                loss_adv_diff_data += loss1.data.item()
                if np.isnan(loss_adv_diff_data):
                    raise ValueError('loss_adv_diff_data is nan while training')
                loss_adv.backward()

                self.optim_gen.step()

                # 4. train discriminator with images from source domain
                for param in self.edge_dis.parameters():
                    param.requires_grad = True
                for param in self.mask_dis.parameters():
                    param.requires_grad = True
                for param in self.model_gen.parameters():
                    param.requires_grad = False

                boundaryS = boundaryS.detach()
                oS = oS.detach()
                D_out2 = self.edge_dis(torch.sigmoid(boundaryS))
                D_out1 = self.mask_dis(oS)

                loss_D_same1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                    source_domain_label).cuda())
                loss_D_same = loss_D_same1+loss_D_same2

                self.running_dis_same_loss += loss_D_same.item()
                loss_D_same_data = loss_D_same.data.item()
                if np.isnan(loss_D_same_data):
                    raise ValueError('loss is nan while training')
                loss_D_same.backward()

                # 5. train discriminator with images from target domain and generated domain

                boundaryT = boundaryT.detach()
                oT = oT.detach()
                D_out2 = self.edge_dis(torch.sigmoid(boundaryT))
                D_out1 = self.mask_dis(oT)

                loss_D_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(
                    target_domain_label).cuda())
                loss_D_diff2 = F.binary_cross_entropy_with_logits(D_out2, torch.FloatTensor(D_out2.data.size()).fill_(
                    target_domain_label).cuda())

                boundaryS2T = boundaryS2T.detach()
                oS2T = oS2T.detach()
                D_out2_1 = self.edge_dis(torch.sigmoid(boundaryS2T))
                D_out1_1 = self.mask_dis(oS2T)
                loss_D_diff1 += F.binary_cross_entropy_with_logits(D_out1_1, torch.FloatTensor(D_out1_1.data.size()).fill_(target_domain_label).cuda())
                loss_D_diff2 += F.binary_cross_entropy_with_logits(D_out2_1, torch.FloatTensor(D_out2_1.data.size()).fill_(target_domain_label).cuda())

                loss_D_diff = loss_D_diff1 + loss_D_diff2
                self.running_dis_diff_loss += loss_D_diff.item()
                loss_D_diff_data = loss_D_diff.data.item()
                if np.isnan(loss_D_diff_data):
                    raise ValueError('loss is nan while training')
                loss_D_diff.backward()

                # 6. update parameters
                self.optim_edge.step()
                self.optim_mask.step()

                self.writer.add_scalar('train_dis/loss_D_same', loss_D_same_data, iteration)
                self.writer.add_scalar('train_dis/loss_D_diff', loss_D_diff_data, iteration)
            self.writer.add_scalar('train_gen/loss_seg', loss_seg_data, iteration)

        self.running_seg_loss /= len(self.domain_loaderS)
        self.running_adv_diff_loss /= len(self.domain_loaderS)
        self.running_dis_same_loss /= len(self.domain_loaderS)
        self.running_dis_diff_loss /= len(self.domain_loaderS)

        stop_time = timeit.default_timer()

        print('\n[Epoch: %d] lr:%f,  Average segLoss: %f, '
              ' Average advLoss: %f, Average dis_same_Loss: %f, '
              'Average dis_diff_Lyoss: %f,'
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss,
               self.running_adv_diff_loss,
               self.running_dis_same_loss, self.running_dis_diff_loss, stop_time - start_time))

    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch+1) % 100 == 0:
                _lr_gen = self.lr_gen * 0.2
                for param_group in self.optim_gen.param_groups:
                    param_group['lr'] = _lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.domain_loaderS)))
            if (self.epoch+1) % self.interval_validate == 0:
                self.validate(epoch=epoch)
        self.writer.close()



