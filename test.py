import argparse
import torch.nn.functional as F

from torch.autograd import Variable
import tqdm
from dataloaders import fundus_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from utils.Utils import *
from utils.metrics import *
from datetime import datetime
import pytz
from networks.Segmentor import ISFA
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default='./logs/Drishti-GS/DGS_weights.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='Drishti-GS', help='test folder id contain images ROIs to test'
    )
    parser.add_argument('-g', '--gpu', type=int, default=1)

    parser.add_argument(
        '--data-dir',
        default='/path/to/ISFA/data/',
        help='data root path'
    )
    parser.add_argument(
        '--save-root-ent',
        type=str,
        default='./results/ent/',
        help='path to save ent',
    )
    parser.add_argument(
        '--save-root-mask',
        type=str,
        default='./results/mask/',
        help='path to save mask',
    )
    parser.add_argument('--test-prediction-save-path', type=str,
                        default='./results/baseline/',
                        help='Path root for test image and mask')
    args = parser.parse_args()
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Resize(256),
        tr.Normalize_tf(),
        tr.ToTensor(),
    ])
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='test',
                                    transform=composed_transforms_test)

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    model = ISFA(backbone='mobilenet').cuda()
    model = torch.nn.DataParallel(model)

    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('==> Evaluating with %s' % (args.dataset))

    val_cup_dice = 0.0
    val_disc_dice = 0.0

    val_cup_pa = 0
    val_disc_pa = 0
    val_cup_iou = 0
    val_disc_iou = 0

    timestamp_start = \
        datetime.now(pytz.timezone('Asia/Hong_Kong'))

    for batch_idx, (sample) in tqdm.tqdm(enumerate(test_loader),
                                         total=len(test_loader),
                                         ncols=80, leave=False):
        data = sample['image']
        target = sample['map']
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        content, style, prediction, boundary = model(data)
        predictions = prediction
        prediction = torch.sigmoid(prediction)
        boundary = torch.sigmoid(boundary)

        draw_ent(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_ent, args.dataset), img_name[0])
        draw_mask(prediction.data.cpu()[0].numpy(), os.path.join(args.save_root_mask, args.dataset), img_name[0])
        draw_boundary(boundary.data.cpu()[0].numpy(), os.path.join(args.save_root_mask, args.dataset), img_name[0])

        prediction = postprocessing(prediction.data.cpu()[0], dataset=args.dataset)
        target_numpy = target.data.cpu()[0].numpy()

        dice_cup, dice_disc = dice_coeff_2label(predictions, target)

        PA_cup, PA_disc, IOU_cup, IOU_disc = pixel_acc(predictions, target)

        val_cup_dice += dice_cup
        val_disc_dice += dice_disc

        val_cup_pa += PA_cup
        val_disc_pa += PA_disc

        val_cup_iou += IOU_cup
        val_disc_iou += IOU_disc

        imgs = data.data.cpu()

        for img, lt, lp in zip(imgs, target_numpy, [prediction]):
            img, lt = untransform(img, lt)
            save_per_img(img.numpy().transpose(1, 2, 0), os.path.join(args.test_prediction_save_path, args.dataset),
                         img_name[0],
                         lp, mask_path=None, ext="png")

    val_cup_dice /= len(test_loader)
    val_disc_dice /= len(test_loader)
    val_disc_pa /= len(test_loader)
    val_cup_pa /= len(test_loader)
    val_cup_iou /= len(test_loader)
    val_disc_iou /= len(test_loader)

    print('''\n==>val_cup_dice : {0}'''.format(val_cup_dice))
    print('''\n==>val_disc_dice : {0}'''.format(val_disc_dice))
    print('''\n==>val_disc_pa : {0}'''.format(val_disc_pa))
    print('''\n==>val_cup_pa : {0}'''.format(val_cup_pa))
    print('''\n==>val_cup_iou : {0}'''.format(val_cup_iou))
    print('''\n==>val_disc_iou : {0}'''.format(val_disc_iou))


if __name__ == '__main__':
    main()
