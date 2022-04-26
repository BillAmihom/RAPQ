import torch
import torch.nn as nn
import argparse
import os
import random
import numpy as np
import time
import hubconf
from quant import *
from data.imagenet import build_imagenet_data


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


def get_train_samples(train_loader, num_samples):
    train_data = []
    for batch in train_loader:
        train_data.append(batch[0])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples]


ii = 0
res_w = 0
res50_w = 0
res50_a = 0
reg600_w = 0
reg600_a = 0
reg3200_w = 0
reg3200_a = 0

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='dataset name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    parser.add_argument('--data_path', default='', type=str, help='path to ImageNet data', required=True)

    # quantization parameters
    parser.add_argument('--n_bits_w', default=4, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', action='store_true', help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=4, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--act_quant', action='store_true', help='apply activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')
    parser.add_argument('--test_before_calibration', action='store_true')

    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--sym', action='store_true', help='symmetric reconstruction, not recommended')
    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')
    parser.add_argument('--step', default=20, type=int, help='record snn output per step')

    # activation calibration parameters
    parser.add_argument('--iters_a', default=1000, type=int, help='number of iteration for LSQ')
    parser.add_argument('--lr', default=4e-4, type=float, help='learning rate for LSQ')
    # parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    # p for a & w
    parser.add_argument('--p', default=2.4, type=float, help='L_p norm minimization for LSQ')
    parser.add_argument('--RAPQ', default=False,action='store_true', help='DONT USE round_ste(log2)')
    args = parser.parse_args()

    seed_all(args.seed)
    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)

    # load model
    cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))

    if 'mobilenetv2' in args.arch :
        print('~~~~~~~~~~~This is mobilenetv2~~~~~~~~~~~~~')
        if args.RAPQ:
            gamma = []

            #无截断版本
            # for name,module in cnn.named_modules():
            #     if isinstance(module, nn.BatchNorm2d):
            #         if name == 'features.0.1':
            #             print(name)
            #             print(round(((module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append( round(((module.weight.sum()) / (module.weight.shape[0])).item(),1) )
            #         elif name == 'features.1.conv.4':
            #             print(name)
            #             print(round(((module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(((module.weight.sum()) / (module.weight.shape[0])).item(), 1))
            #         elif name == 'features.18.1':
            #
            #             print(name)
            #             print(round(((module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(((module.weight.sum()) / (module.weight.shape[0])).item(), 1))
            #         elif 'conv.7' in name:
            #             print(name)
            #             print(round(((module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(((module.weight.sum()) / (module.weight.shape[0])).item(), 1))

            #截断版本
            # for name,module in cnn.named_modules():
            #     if isinstance(module, nn.BatchNorm2d):
            #         # print(name)
            #         # print('gammagammagammagammagammagammagammagammagammagammagammagammagammagammagamma')
            #         # print(module.weight.shape)
            #         # print((module.weight.sum()) / (module.weight.shape[0]))
            #         # print('running_varrunning_varrunning_varrunning_varrunning_varrunning_varrunning_var')
            #         # print((module.running_var.sum()) / (module.running_var.shape[0]))
            #         # print(module.running_var.shape)
            #         if name == 'features.0.1':
            #             # print(name)
            #             # print(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #             gamma.append(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #         elif name == 'features.1.conv.4':
            #             # print(name)
            #             # print(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #             gamma.append(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #         elif name == 'features.18.1':
            #             # print(name)
            #             # print(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #             gamma.append(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #         elif 'conv.7' in name:
            #             # print(name)
            #             # print(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))
            #             gamma.append(round(torch.clamp(((module.weight.sum()) / (module.weight.shape[0])),1,2).item(),1))

            #cosin版本 0.2618 = pi/12
            # for name,module in cnn.named_modules():
            #     if isinstance(module, nn.BatchNorm2d):
            #         if name == 'features.0.1':
            #             print(name)
            #             print(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #         elif name == 'features.1.conv.4':
            #             print(name)
            #             print(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #         elif name == 'features.18.1':
            #
            #             print(name)
            #             print(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #         elif 'conv.7' in name:
            #             print(name)
            #             print(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))
            #             gamma.append(round(-1.5+5*torch.cos(0.2618*(module.weight.sum()) / (module.weight.shape[0])).item(),1))

            # sigmoid
            for name,module in cnn.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    if name == 'features.0.1':
                        # print(name)
                        # print(round((1 + 0.7*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'features.1.conv.4':
                        # print(name)
                        # print(round((1 + 0.7*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'features.18.1':
                        # print(name)
                        # print(round((1 + 0.7*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif 'conv.7' in name:
                        # print(name)
                        # print(round((1 + 0.7*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))

            # running_var sigmoid
            # for name,module in cnn.named_modules():
            #     if isinstance(module, nn.BatchNorm2d):
            #         if name == 'features.0.1':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(1 + torch.sigmoid(-1+(module.running_var.sum()) / (module.running_var.shape[0])).item(), 1)
            #             gamma.append(sig_var)
            #         elif name == 'features.1.conv.4':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(1 + torch.sigmoid(-1+(module.running_var.sum()) / (module.running_var.shape[0])).item(), 1)
            #             gamma.append(sig_var)
            #         elif name == 'features.18.1':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(1 + torch.sigmoid(-1+(module.running_var.sum()) / (module.running_var.shape[0])).item(), 1)
            #             gamma.append(sig_var)
            #         elif 'conv.7' in name:
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(1 + torch.sigmoid(-1+(module.running_var.sum()) / (module.running_var.shape[0])).item(), 1)
            #             gamma.append(sig_var)

            # cosin running_var
            # for name,module in cnn.named_modules():
            #     if isinstance(module, nn.BatchNorm2d):
            #         if name == 'features.0.1':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(torch.clamp(1.75*torch.cos(0.2618*(module.running_var.sum()) / (module.running_var.shape[0])),1,2).item(),1)
            #             gamma.append(sig_var)
            #         elif name == 'features.1.conv.4':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(torch.clamp(1.75*torch.cos(0.2618*(module.running_var.sum()) / (module.running_var.shape[0])),1,2).item(),1)
            #             gamma.append(sig_var)
            #         elif name == 'features.18.1':
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(torch.clamp(1.75*torch.cos(0.2618*(module.running_var.sum()) / (module.running_var.shape[0])),1,2).item(),1)
            #             gamma.append(sig_var)
            #         elif 'conv.7' in name:
            #             print(name)
            #             print((module.running_var.sum()) / (module.running_var.shape[0]))
            #             sig_var = round(torch.clamp(1.75*torch.cos(0.2618*(module.running_var.sum()) / (module.running_var.shape[0])),1,2).item(),1)
            #             gamma.append(sig_var)
        else:
            gamma = [2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]

        gamma.append(2)
        print(len(gamma))
        #最后一层没有BN,所以直接取MSE/或取高p值4
        print(gamma)

    if 'resnet18' in args.arch :
        print('~~~~~~~~~~~This is resnet18~~~~~~~~~~~~~')
        if args.RAPQ:
            gamma = []
            for name, module in cnn.named_modules():
                print(name)
                if isinstance(module, nn.BatchNorm2d):
                    if 'bn2' in name:
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'bn1':
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                # gamma = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        else:
            gamma = [2,2,2,2,2,2,2,2,2]
        print(len(gamma))
        print(gamma)

    if 'resnet50' in args.arch :
        print('~~~~~~~~~~~This is resnet50~~~~~~~~~~~~~')
        if args.RAPQ:
            gamma = []
            for name, module in cnn.named_modules():
                print(name)
                if isinstance(module, nn.BatchNorm2d):
                    if 'bn3' in name:
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'bn1':
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
            # gamma = [2] * 17

        else:
            gamma = [2] * 17
        print(len(gamma))
        print(gamma)

    if 'regnetx_600m' in args.arch :
        print('~~~~~~~~~~~This is regnetx_600m~~~~~~~~~~~~~')
        if args.RAPQ:
            gamma = []
            for name, module in cnn.named_modules():
                # print(name)
                if isinstance(module, nn.BatchNorm2d):
                    if 'f.c_bn' in name:
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'stem.bn':
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
            # gamma = [2] * 17

        else:
            gamma = [2]*17
        print(len(gamma))
        print(gamma)

    if 'regnetx_3200m' in args.arch :
        print('~~~~~~~~~~~This is regnetx_600m~~~~~~~~~~~~~')
        if args.RAPQ:
            gamma = []
            for name, module in cnn.named_modules():
                print(name)
                if isinstance(module, nn.BatchNorm2d):
                    if 'f.c_bn' in name:
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
                    elif name == 'stem.bn':
                        gamma.append(round((1 + 0.1*torch.sigmoid((module.weight.sum() - 1) / (module.weight.shape[0]))).item(),1))
            # gamma = [2]*26

        else:
            gamma = [2]*26
        print(len(gamma))
        print(gamma)


    cnn.cuda()
    cnn.eval()

    # build quantization parameters
    # wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse'}
    # aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant}
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': 'mse','RAPQ': args.RAPQ }
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': 'mse', 'leaf_param': args.act_quant,'RAPQ':args.RAPQ}
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params)
    qnn.cuda()
    qnn.eval()
    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    cali_data = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Initialize weight quantization parameters
    qnn.set_quant_state(True, False)
    _ = qnn(cali_data[:64].to(device))

    if args.test_before_calibration:
        print('Quantized accuracy before RAPQ: {}'.format(validate_model(test_loader, qnn)))

    # Kwargs for weight rounding calibration
    # kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
    #               b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',p=args.p)

# mobilenetv2
    def mbv2_w_recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    print((int(name) + 18))
                    print(gamma[int(name)+18])
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[int(name)+18],RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    print(int(name))
                    print(gamma[int(name)])
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[int(name)],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                mbv2_w_recon_model(module)

    def mbv2_a_recon_model(model: nn.Module):
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[int(name)+18],RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[int(name)],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                a_recon_model(module)

# resnet18
    def resnet18_w_recon_model(model: nn.Module):
        global res_w
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    res_w = res_w+1
                    print('res_w:%d'%res_w)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[res_w],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                resnet18_w_recon_model(module)

    def resnet18_a_recon_model(model: nn.Module):
        global ii
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    ii = ii+1
                    print('ii:%d'%ii)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[ii],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                resnet18_a_recon_model(module)
# resnet50
    def resnet50_w_recon_model(model: nn.Module):
        global res50_w
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    res50_w = res50_w+1
                    print('res50_w:%d'%res50_w)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[res50_w],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                resnet50_w_recon_model(module)

    def resnet50_a_recon_model(model: nn.Module):
        global res50_a
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    res50_a = res50_a+1
                    print('res50_a:%d'%res50_a)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[res50_a],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                resnet50_a_recon_model(module)

#regnetx
    def regnetx_600m_w_recon_model(model: nn.Module):
        global reg600_w
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    reg600_w = reg600_w+1
                    print('reg600_w:%d'%reg600_w)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[reg600_w],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                regnetx_600m_w_recon_model(module)

    def regnetx_600m_a_recon_model(model: nn.Module):
        global reg600_a
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    reg600_a = reg600_a+1
                    print('reg600_a:%d'%reg600_a)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[reg600_a],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                regnetx_600m_a_recon_model(module)

    def regnetx_3200m_w_recon_model(model: nn.Module):
        global reg3200_w
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():
            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    reg3200_w = reg3200_w+1
                    print('reg3200_w:%d'%reg3200_w)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_w, weight=args.weight, asym=True,
                                  b_range=(args.b_start, args.b_end), warmup=args.warmup, act_quant=False, opt_mode='mse',
                                  p=gamma[reg3200_w],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                regnetx_3200m_w_recon_model(module)

    def regnetx_3200m_a_recon_model(model: nn.Module):
        global reg3200_a
        """
        Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for name, module in model.named_children():

            if isinstance(module, QuantModule):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of layer {}'.format(name))
                    continue
                else:
                    print('Reconstruction for layer {}'.format(name))
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=2,RAPQ=args.RAPQ)
                    layer_reconstruction(qnn, module, **kwargs)
            elif isinstance(module, BaseQuantBlock):
                if module.ignore_reconstruction is True:
                    print('Ignore reconstruction of block {}'.format(name))
                    continue
                else:
                    print('Reconstruction for block {}'.format(name))
                    reg3200_a = reg3200_a+1
                    print('reg3200_a:%d'%reg3200_a)
                    kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr,
                                  p=gamma[reg3200_a],RAPQ=args.RAPQ)
                    block_reconstruction(qnn, module, **kwargs)
            else:
                regnetx_3200m_a_recon_model(module)

    # Start calibration



    if 'mobilenetv2' in args.arch:
        mbv2_w_recon_model(qnn)
    elif 'resnet18' in args.arch:
        resnet18_w_recon_model(qnn)
    elif 'resnet50' in args.arch:
        resnet50_w_recon_model(qnn)
    elif 'regnetx_600m'in args.arch:
        regnetx_600m_w_recon_model(qnn)
    elif 'regnetx_3200m'in args.arch:
        regnetx_3200m_w_recon_model(qnn)

    qnn.set_quant_state(weight_quant=True, act_quant=False)
    print('Weight quantization accuracy: {}'.format(validate_model(test_loader, qnn)))

    if args.act_quant:
        # Initialize activation quantization parameters
        qnn.set_quant_state(True, True)
        with torch.no_grad():
            _ = qnn(cali_data[:64].to(device))
        # Disable output quantization because network output
        # does not get involved in further computation
        qnn.disable_network_output_quantization()
        # Kwargs for activation rounding calibration
        # kwargs = dict(cali_data=cali_data, iters=args.iters_a, act_quant=True, opt_mode='mse', lr=args.lr, p=args.p)
        if 'mobilenetv2' in args.arch:
            mbv2_a_recon_model(qnn)
        elif 'resnet18' in args.arch:
            resnet18_a_recon_model(qnn)
        elif 'resnet50' in args.arch:
            resnet50_a_recon_model(qnn)
        elif 'regnetx_600m' in args.arch:
            regnetx_600m_a_recon_model(qnn)
        elif 'regnetx_3200m' in args.arch:
            regnetx_3200m_a_recon_model(qnn)

        print('~~~~~~~~~~~This is a_recon_model~~~~~~~~~~~~~')
        for k, v in qnn.named_parameters():
            print(k)
        qnn.set_quant_state(weight_quant=True, act_quant=True)
        print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
                                                               validate_model(test_loader, qnn)))
