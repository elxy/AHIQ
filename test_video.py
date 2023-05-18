import argparse
import json
import os
from pprint import pprint

import timm
from timm.models.vision_transformer import Block
from timm.models.resnet import Bottleneck
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from data.video import VideoPredict
from model.deform_regressor import deform_fusion, Pixel_Prediction
from script.extract_feature import get_resnet_feature, get_vit_feature
from utils.process_image import five_point_crop
from utils.util import SaveOutput


def predict_model(deform_net, resnet50, vit, regressor, opt, device, dataloader):
    save_output = SaveOutput()
    for layer in resnet50.modules():
        if isinstance(layer, Bottleneck):
            layer.register_forward_hook(save_output)
    for layer in vit.modules():
        if isinstance(layer, Block):
            layer.register_forward_hook(save_output)

    with torch.no_grad():
        epoch_preds = []
        pbar = tqdm(dataloader)
        for ref, dis, in pbar:
            ref = ref.to(device)
            dis = dis.to(device)

            pred = 0
            for i in range(opt.n_ensemble):
                b, c, h, w = ref.size()
                if opt.n_ensemble > 9:
                    new_h = opt.crop_size
                    new_w = opt.crop_size
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                    r_img = ref[:,:, top: top+new_h, left: left+new_w]
                    d_img = dis[:,:, top: top+new_h, left: left+new_w]
                elif opt.n_ensemble == 1:
                    r_img = ref
                    d_img = dis
                else:
                    d_img, r_img = five_point_crop(i, d_img=dis, r_img=ref, config=opt)

                d_img = d_img.cuda()
                r_img = r_img.cuda()

                _x = vit(d_img)
                vit_dis = get_vit_feature(save_output)
                save_output.outputs.clear()
                _y = vit(r_img)
                vit_ref = get_vit_feature(save_output)
                save_output.outputs.clear()

                B, N, C = vit_ref.shape
                if opt.patch_size == 8:
                    H, W = 28, 28
                else:
                    H, W = 14, 14
                assert H * W == N
                vit_ref = vit_ref.transpose(1, 2).view(B, C, H, W)
                vit_dis = vit_dis.transpose(1, 2).view(B, C, H, W)

                _ = resnet50(d_img)
                cnn_dis = get_resnet_feature(save_output)
                save_output.outputs.clear()
                cnn_dis = deform_net(cnn_dis, vit_ref)

                _ = resnet50(r_img)
                cnn_ref = get_resnet_feature(save_output)
                save_output.outputs.clear()
                cnn_ref = deform_net(cnn_ref, vit_ref)

                pred += regressor(vit_dis, vit_ref, cnn_dis, cnn_ref)

            pred /= opt.n_ensemble
            epoch_preds.append(pred)
            pbar.set_postfix(MOS=pred)

        epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()
        epoch_preds = epoch_preds.tolist()
        return epoch_preds


def parse_opts():

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_ref', type=str, help='Path to reference video')
    parser.add_argument('--video_dis', type=str, help='Path to distorion video')
    parser.add_argument('--output', type=str, help='Path to store predict result')
    parser.add_argument('--load_model', type=str, required=True, help='Path to load checkpoint')
    parser.add_argument('--log_file_name', default='./log/run.log', type=str, help='Path to save log')

    parser.add_argument('--patch_size', type=int, default=8, help='patch size of Vision Transformer')
    parser.add_argument('--load_epoch',
                        type=int,
                        default=-1,
                        help='which epoch to load? set to -1 to use latest cached model')
    parser.add_argument('--seed', type=int, default=1919, help='random seed')

    parser.add_argument('--n_ensemble', type=int, default=20, help='crop method for test: five points crop or nine points crop or random crop for several times')
    parser.add_argument('--crop_size', type=int, default=224, help='image size')
    parser.add_argument('--batch_size', type=int, default=10, help='input batch size')
    parser.add_argument('--multi_gpu', action='store_true', help='whether to use all GPUs')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    opt = parse_opts()

    video_ref = opt.video_ref
    video_dis = opt.video_dis
    output = opt.output
    load_model = opt.load_model
    batch_size = opt.batch_size
    MULTI_GPU_MODE = opt.multi_gpu

    video_dataset = VideoPredict(video_ref, video_dis)
    dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    deform_net = deform_fusion(opt).to(device)
    resnet50 = timm.create_model('resnet50', pretrained=True).to(device)
    if opt.patch_size == 8:
        vit = timm.create_model('vit_base_patch8_224', pretrained=True).to(device)
    else:
        vit = timm.create_model('vit_base_patch16_224', pretrained=True).to(device)
    regressor = Pixel_Prediction().to(device)

    checkpoint = torch.load(load_model)
    regressor.load_state_dict(checkpoint['regressor_model_state_dict'])
    deform_net.load_state_dict(checkpoint['deform_net_model_state_dict'])

    predicts = predict_model(deform_net, resnet50, vit, regressor, opt, device, dataloader)
    predicts = np.asfarray(predicts)
    average = np.mean(predicts)
    std = np.std(predicts)
    min = np.min(predicts)
    max = np.max(predicts)
    print(f'MOS: average {average:.2f}, std {std:.2f}, min {min:.2f}, max {max:.2f}')

    if opt.output is None:
        opt.output = 'predict_result.json'
    info = {
        'raw_mos': list(predicts),
        'mos': {
            'average': average,
            'std': std,
            'min': min,
            'max': max
        },
        'opt': vars(opt)
    }
    with open(opt.output, 'wt') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)
