import argparse
import os, sys
import pandas as pd

sys.path.append("/Data/DoDNet/")

import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm

import skimage

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
#from MOTSDataset_2D_Patch_supervise_csv import MOTSValDataSet as MOTSValDataSet_joint
# from MOTSDataset_2D_Patch_supervise_normal_csv_512 import MOTSValDataSet as MOTSValDataSet_joint
from MOTSDataset_2D_Patch_supervise_normal_csv_512 import MOTSValDataSet_512_center as MOTSValDataSet_joint
from unet2D_ns import UNet2D as UNet2D_ns
from unet2D_Dodnet_scale_for_2_class import UNet2D as UNet2D_scale
#from unet2D_Dodnet_scale_v2 import UNet2D as UNet2D_scale
import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool


def one_hot_2D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def get_arguments():

    parser = argparse.ArgumentParser(description="DeepLabV3")


    parser.add_argument("--valset_dir", type=str, default='/data2/ME Project/KPMP_Patches_backup/Final_Project_data/test/data_list.csv')
    # parser.add_argument("--valset_dir", type=str, default='/Data2/KI_data_validationset_patch/')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white_scale_allpsuedo_allMatching_with_half_semi_0.05_0.05_normalwhole_0907/')

    parser.add_argument("--reload_path", type=str,
                        default='/data2/ME Project/KPMP_Patches_backup/Final_Project_data/snapshots_2D/Omniseg_H2H_glo_v2r/fold1_with_white_scale_normalwhole_1217/MOTS_DynConv_fold1_with_white_scale_normalwhole_1217_final_e199.pth')

    parser.add_argument("--best_epoch", type=int, default=199)

    parser.add_argument("--validsetname", type=str, default='normal')

    parser.add_argument("--train_list", type=str, default='./data/Mice_Patches/train/data_list.csv')

    parser.add_argument("--val_list", type=str, default='./data/Mice_Patches/val/data_list.csv')
    parser.add_argument("--edge_weight", type=float, default=1.2)

    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='512,512')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.

def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)


def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        # try:
        ###############################################################################
        hausdorff, meansurfaceDistance = surfd(preds0, labels0)
        Val_HD += hausdorff
        Val_MSD += meansurfaceDistance
        ##############################################################################3

        Val_F1 += f1_score(preds1, labels1, average='macro')

        # except:
        #     Val_DICE += 1.
        #     Val_F1 += 1.
        #     Val_HD += 0.
        #     Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)


def main():
    """Create the model and start the training."""
    parser = get_arguments()
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        #model = UNet2D_ns(num_classes=args.num_classes, weight_std = False)
        model = UNet2D_scale(num_classes=args.num_classes, weight_std=False)
        check_wo_gpu = 0

        print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))


        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        valloader = DataLoader(
            MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=1,shuffle=False,num_workers=num_worker)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        batch_size = args.batch_size
        # for epoch in range(0,args.num_epochs):

        model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()
        task0_pool_image = ImagePool(8)
        task0_pool_mask = ImagePool(8)
        task0_scale = []
        task0_name = []
        task1_pool_image = ImagePool(8)
        task1_pool_mask = ImagePool(8)
        task1_scale = []
        task1_name = []

        val_loss = np.zeros((2))
        val_F1 = np.zeros((2))
        val_Dice = np.zeros((2))
        val_HD = np.zeros((2))
        val_MSD = np.zeros((2))
        cnt = np.zeros((2))

        ####################################################################
        single_df_0 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])
        single_df_1 = pd.DataFrame(columns=['name', 'F1', 'Dice', 'HD', 'MSD'])

        with torch.no_grad():
            for iter, batch in enumerate(valloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                for ki in range(len(imgs)):
                    now_task = t_ids[ki]
                    if now_task == 0:
                        task0_pool_image.add(imgs[ki].unsqueeze(0))
                        task0_pool_mask.add(lbls[ki].unsqueeze(0))
                        task0_scale.append((s_ids[ki]))
                        task0_name.append((volumeName[ki]))
                    elif now_task == 1:
                        task1_pool_image.add(imgs[ki].unsqueeze(0))
                        task1_pool_mask.add(lbls[ki].unsqueeze(0))
                        task1_scale.append((s_ids[ki]))
                        task1_name.append((volumeName[ki]))

                output_folder = os.path.join('/data2/ME Project/KPMP_Patches_backup/Final_Project_data/DoDNet/MIDL/MIDL_github/Glo_Omniseg_FinalProject/testing_%s' % (args.validsetname),
                                             str(args.best_epoch))
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
                optimizer.zero_grad()

                #print("task0", task0_pool_image.num_imgs)




                if task0_pool_image.num_imgs >= batch_size:
                    images = task0_pool_image.query(batch_size)
                    labels = task0_pool_mask.query(batch_size)
                    now_task = torch.tensor(0)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                        filename.append(task0_name.pop(0))

                    preds, _ = model(images, torch.ones(batch_size).cuda() * 0, scales)

                    now_preds = preds[:,1,...] > preds[:,0,...]
                    now_preds_onehot = one_hot_2D(now_preds.long())

                    labels_onehot = one_hot_2D(labels.long())

                    rmin, rmax, cmin, cmax = mask_to_box(images)

                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_0)
                        single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[0] += F1
                        val_Dice[0] += DICE
                        val_HD[0] += HD
                        val_MSD[0] += MSD
                        cnt[0] += 1

                if task1_pool_image.num_imgs >= batch_size:
                    images = task1_pool_image.query(batch_size)
                    labels = task1_pool_mask.query(batch_size)
                    scales = torch.ones(batch_size).cuda()
                    filename = []
                    for bi in range(len(scales)):
                        scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                        filename.append(task1_name.pop(0))

                    preds, _ = model(images, torch.ones(batch_size).cuda() * 1, scales)
                    now_task = torch.tensor(1)

                    now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                    now_preds_onehot = one_hot_2D(now_preds.long())
                    labels_onehot = one_hot_2D(labels.long())
                    rmin, rmax, cmin, cmax = mask_to_box(images)


                    for pi in range(len(images)):
                        prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                        out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                        img = (out_image - out_image.min())/ (out_image.max() - out_image.min())
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                                   img)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                                   labels[pi, ...].detach().cpu().numpy(), cmap = cm.gray)
                        plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' %(now_task.item())),
                                   prediction.detach().cpu().numpy(), cmap = cm.gray)

                        F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                         rmin, rmax, cmin, cmax)
                        row = len(single_df_1)
                        single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                        val_F1[1] += F1
                        val_Dice[1] += DICE
                        val_HD[1] += HD
                        val_MSD[1] += MSD
                        cnt[1] += 1



            if (task0_pool_image.num_imgs < batch_size) & (task0_pool_image.num_imgs >0):
                left_size = task0_pool_image.num_imgs
                images = task0_pool_image.query(left_size)
                labels = task0_pool_mask.query(left_size)

                # last_image = torch.unsqueeze(images[-1], dim=0)
                # replicated_image = last_image.repeat(batch_size - left_size, 1, 1, 1)
                # images = torch.cat((images, replicated_image), dim=0)
                #
                # last_label = torch.unsqueeze(labels[-1], dim=0)
                # replicated_label = last_label.repeat(batch_size - left_size, 1, 1)
                # labels = torch.cat((labels, replicated_label), dim=0)

                now_task = torch.tensor(0)
                scales = torch.ones(left_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[int(len(scales) - 1 - bi)] = task0_scale.pop(0)
                    filename.append(task0_name.pop(0))

                preds, _ = model(images, torch.ones(left_size).cuda() * 0, scales)

                now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                now_preds_onehot = one_hot_2D(now_preds.long())

                labels_onehot = one_hot_2D(labels.long())

                rmin, rmax, cmin, cmax = mask_to_box(images)

                for pi in range(len(images)):
                    prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                               prediction.detach().cpu().numpy(), cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_0)
                    single_df_0.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[0] += F1
                    val_Dice[0] += DICE
                    val_HD[0] += HD
                    val_MSD[0] += MSD
                    cnt[0] += 1

            if (task1_pool_image.num_imgs < batch_size) & (task1_pool_image.num_imgs >0):
                left_size = task1_pool_image.num_imgs
                images = task1_pool_image.query(left_size)
                labels = task1_pool_mask.query(left_size)
                scales = torch.ones(left_size).cuda()
                filename = []
                for bi in range(len(scales)):
                    scales[int(len(scales) - 1 - bi)] = task1_scale.pop(0)
                    filename.append(task1_name.pop(0))

                preds, _ = model(images, torch.ones(left_size).cuda() * 1, scales)
                now_task = torch.tensor(1)

                now_preds = preds[:, 1, ...] > preds[:, 0, ...]
                now_preds_onehot = one_hot_2D(now_preds.long())
                labels_onehot = one_hot_2D(labels.long())
                rmin, rmax, cmin, cmax = mask_to_box(images)

                for pi in range(len(images)):
                    prediction = preds[pi, 1, ...] > preds[pi, 0, ...]
                    out_image = images[pi, ...].permute([1, 2, 0]).detach().cpu().numpy()
                    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_image.png'),
                               img)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_mask.png'),
                               labels[pi, ...].detach().cpu().numpy(), cmap=cm.gray)
                    plt.imsave(os.path.join(output_folder, filename[pi] + '_preds_%s.png' % (now_task.item())),
                               prediction.detach().cpu().numpy(), cmap=cm.gray)

                    F1, DICE, HD, MSD = count_score(now_preds_onehot[pi].unsqueeze(0), labels_onehot[pi].unsqueeze(0),
                                                    rmin, rmax, cmin, cmax)
                    row = len(single_df_1)
                    single_df_1.loc[row] = [filename[pi], F1, DICE.cpu().numpy(), HD, MSD]

                    val_F1[1] += F1
                    val_Dice[1] += DICE
                    val_HD[1] += HD
                    val_MSD[1] += MSD
                    cnt[1] += 1

        avg_val_F1 = val_F1 / cnt
        avg_val_Dice = val_Dice / cnt
        avg_val_HD = val_HD / cnt
        avg_val_MSD = val_MSD / cnt

        print('Validate \n 0dt_f1={:.4} 0dt_dsc={:.4} 0dt_hd={:.4} 0dt_msd={:.4}'
              ' \n 1pt_f1={:.4} 1pt_dsc={:.4} 1pt_hd={:.4} 1pt_msd={:.4}\n'

              .format(avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item(),
                      avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item()
                      ))

        df = pd.DataFrame(columns = ['task','F1','Dice','HD','MSD'])
        df.loc[0] = ['0Ahn', avg_val_F1[0].item(), avg_val_Dice[0].item(), avg_val_HD[0].item(), avg_val_MSD[0].item()]
        df.loc[1] = ['1Cap', avg_val_F1[1].item(), avg_val_Dice[1].item(), avg_val_HD[1].item(), avg_val_MSD[1].item()]
        ##########################################################################
        df.to_csv(os.path.join(output_folder,'testing_result.csv'))

        single_df_0.to_csv(os.path.join(output_folder,'testing_result_0.csv'))
        single_df_1.to_csv(os.path.join(output_folder,'testing_result_1.csv'))


    end = timeit.default_timer()
    print(end - start, 'seconds')


if __name__ == '__main__':
    main()
