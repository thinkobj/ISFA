from datetime import datetime
import os
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer
from networks.Segmentor import ISFA

# Custom includes
from dataloaders import fundus_dataloader as DL
from dataloaders import custom_transforms as tr
from networks.deeplabv3 import *
from networks.GAN import EdgeDiscriminator, MaskDiscriminator

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

here = osp.dirname(osp.abspath(__file__))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--resume', default=None, help='checkpoint path')
    parser.add_argument(
        '--datasetS', type=str, default='refuge_train', help='source folder id contain images ROIs to train'
    )
    parser.add_argument(
        '--datasetT', type=str, default='Drishti-GS', help='target folder id contain images ROIs to train'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=400, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=400, help='stop epoch'
    )
    parser.add_argument(
        '--warmup-epoch', type=int, default=-1, help='warmup epoch begin train GAN'
    )

    parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-5, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default='/path/to/ISFA/data/',
        help='data root path'
    )

    args = parser.parse_args()
    now = datetime.now()
    args.out = osp.join(here, 'logs', args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))
    os.makedirs(args.out)

    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)
    cuda = torch.cuda.is_available()

    # 1. dataset
    composed_transforms_tr = transforms.Compose([
        tr.Resize(256),
        tr.RandomRotate(),
        tr.RandomFlip(),
        tr.elastic_transform(),
        tr.add_salt_pepper_noise(),
        tr.adjust_light(),
        tr.eraser(),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    composed_transforms_ts = transforms.Compose([
        tr.Resize(256),
        tr.Normalize_tf(),
        tr.ToTensor()
    ])

    domain = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train',
                                   transform=composed_transforms_tr)
    domain_loaderS = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                drop_last=True)

    domain_S2T = DL.FundusSegmentation(base_dir=args.data_dir, dataset='refuge_train_gen', split='train',
                                       transform=composed_transforms_tr)
    domain_loaderS2T = DataLoader(domain_S2T, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True,
                                  drop_last=True)

    domain_T = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                     transform=composed_transforms_tr)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
                                drop_last=True)
    domain_val = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='test',
                                       transform=composed_transforms_ts)
    domain_loader_val = DataLoader(domain_val, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=True, drop_last=True)
    print("data load finished")
    # 2. model
    model_gen = ISFA(backbone='mobilenet').cuda()
    model_gen = torch.nn.DataParallel(model_gen)

    edge_dis = EdgeDiscriminator().cuda()
    mask_dis = MaskDiscriminator().cuda()

    start_epoch = 0
    start_iteration = 0
    print("model load finished")

    # 3. optimizer

    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )
    optim_edge = torch.optim.SGD(
        edge_dis.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optim_mask = torch.optim.SGD(
        mask_dis.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    print("optim set finished")

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        edge_dis=edge_dis,
        mask_dis=mask_dis,
        optimizer_gen=optim_gen,
        optimizer_edge=optim_edge,
        optimizer_mask=optim_mask,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        lr_decrease_rate=args.lr_decrease_rate,
        val_loader=domain_loader_val,
        domain_loaderS=domain_loaderS,
        domain_loaderS2T=domain_loaderS2T,
        domain_loaderT=domain_loaderT,
        out=args.out,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
        warmup_epoch=args.warmup_epoch,
    )

    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
