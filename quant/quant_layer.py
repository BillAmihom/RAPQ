import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from torch.autograd import Function

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


#线性非对称量化器
class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False,RAPQ:bool = True):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

        self.quantize_fn_cls = RAPQuantize
        self.log_threshold = None

        self.RAPQ = RAPQ

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                # delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)#byme
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
                # self.delta = Search_Pow2.apply(delta) #不可训练时不用data
                # self.delta = torch.nn.Parameter(delta)
                # self.delta.data = Search_Pow2.apply(delta) #可训练时用data 到下面

                # self.zero_point = torch.nn.Parameter(self.zero_point) #bywriter
            else:
                # delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                # self.delta = torch.nn.Parameter(delta)
                # self.delta.data = Search_Pow2.apply(delta) #可训练时用data 到下面
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
                # self.delta.data = Search_Pow2.apply(self.delta)
            self.inited = True

        if self.RAPQ:
            if self.leaf_param:
                self.log_threshold = nn.Parameter(torch.log2(self.delta))
                # self.log_threshold = torch.log2(self.delta)

                # print(torch.Tensor([self.n_levels])[-1])
                # print(self.zero_point.shape)
                x_dequant = self.quantize_fn_cls.apply(
                    x, self.log_threshold, torch.Tensor([self.n_levels]).cuda(), self.zero_point)

                # self.delta = 2**(round_ste(self.log_threshold))
                delta = 2 ** (self.log_threshold)
                ceil_float_range = delta.ceil()
                floor_float_range = delta.floor()
                delta[abs(ceil_float_range - delta) < abs(floor_float_range - delta)].data.copy_(ceil_float_range[
                                                                                                     abs(
                                                                                                         ceil_float_range - delta) < abs(
                                                                                                         floor_float_range - delta)].data)
                delta[abs(ceil_float_range - delta) >= abs(floor_float_range - delta)].data.copy_(floor_float_range[
                                                                                                      abs(
                                                                                                          ceil_float_range - delta) >= abs(
                                                                                                          floor_float_range - delta)].data)

                # self.zero_point = torch.round((self.zero_point * self.delta) / delta)
                self.zero_point = (self.zero_point * self.delta) / delta

                self.delta = delta
            else:
                # start quantization
                # delta = 2 ** (torch.log2(self.delta))
                # delta[delta < 0].data.copy_(torch.Tensor([2 ** -5]))
                # delta[delta > 2 ** (8 + 5)].data.copy_(torch.Tensor([2 ** (8 + 5)]))
                # ceil_float_range = delta.ceil()
                # floor_float_range = delta.floor()
                # delta[abs(ceil_float_range - delta) < abs(floor_float_range - delta)].data.copy_(ceil_float_range[
                #                                                                                      abs(
                #                                                                                          ceil_float_range - delta) < abs(
                #                                                                                          floor_float_range - delta)].data)
                # delta[abs(ceil_float_range - delta) >= abs(floor_float_range - delta)].data.copy_(floor_float_range[
                #                                                                                       abs(
                #                                                                                           ceil_float_range - delta) >= abs(
                #                                                                                           floor_float_range - delta)].data)
                #
                # self.delta = delta

                x_int = round_ste(x / self.delta) + self.zero_point
                x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
                x_dequant = (x_quant - self.zero_point) * self.delta
        else:
            self.log_threshold = round_ste(self.delta.log2())
            delta = 2 ** self.log_threshold
            self.zero_point = (self.zero_point * self.delta) / delta

            self.delta.data = delta

            x_int = round_ste(x / self.delta) + self.zero_point
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
            x_dequant = (x_quant - self.zero_point) * self.delta

        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError

        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)


class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Conv2d, nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, se_module=None):
        super(QuantModule, self).__init__()
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        self.weight = org_module.weight
        self.org_weight = org_module.weight.data.clone()
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params)
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.se_module = se_module
        self.extra_repr = org_module.extra_repr

    def forward(self, input: torch.Tensor):
        if self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias
        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        # disable act quantization is designed for convolution bef ore elemental-wise operation,
        # in that case, we apply activation function and quantization after ele-wise op.
        if self.se_module is not None:
            out = self.se_module(out)
        out = self.activation_function(out)
        if self.disable_act_quant:
            return out
        if self.use_act_quant:
            out = self.act_quantizer(out)
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

class RAPQuantize(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x, logt, domain, zero_point):
    # scale = 2**(torch.ceil(logt)) / domain

    # delta = 2**(round_ste(logt)) #这里可以改成每通道取近的

    delta = 2 ** (logt)
    ceil_float_range = delta.ceil()
    floor_float_range = delta.floor()
    delta[abs(ceil_float_range - delta) < abs(floor_float_range - delta)].data.copy_(ceil_float_range[
                                                                                            abs(ceil_float_range - delta) < abs(
                                                                                                floor_float_range - delta)].data)
    delta[abs(ceil_float_range - delta) >= abs(floor_float_range - delta)].data.copy_(floor_float_range[
                                                                                             abs(ceil_float_range - delta) >= abs(
                                                                                                 floor_float_range - delta)].data)
    quant_max = domain - 1
    quant_min = 0 * domain

    ctx.save_for_backward(x, delta, zero_point,quant_max, quant_min, logt)

    x = x.clone()

    x_int = round_ste(x / delta) + zero_point
    x_quant = torch.clamp(x_int, 0, domain[-1] - 1)
    x_dequant = (x_quant - zero_point) * delta

    #ctx.mark_dirty(x)
    # return fix_ops.NndctFixNeuron(x, x, (domain, 1 / scale), method)
    #return torch.clamp(torch.round(x/scale), quant_min.item(), quant_max.item()) * scale
    return x_dequant

  @staticmethod
  def backward(ctx, grad_output):
    x, delta,zero_point, quant_max, quant_min, logt = ctx.saved_tensors

    scaled_x = zero_point + x / delta

    # Python equivalent to NndctFixNeuron rounding implementation which is
    # consistent with hardware runtime.
    # See nndct/include/cuda/nndct_fix_kernels.cuh::_fix_neuron_v2_device
    # Round -1.5 to -1 instead of -2.
    rounded_scaled_x = torch.where(
        (scaled_x < 0) & (scaled_x - torch.floor(scaled_x) == 0.5),
        torch.ceil(scaled_x), torch.round(scaled_x))

    is_lt_min = rounded_scaled_x < quant_min
    is_gt_max = rounded_scaled_x > quant_max
    is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max

    #grad_logt = torch.ones(grad_output.shape, dtype=grad_output.dtype, device=grad_output.device) * scale * math.log(2)
    grad_logt = grad_output * delta * 0.69314718 # s * ln(2)
    grad_logt = torch.where(is_ge_min_and_le_max,
                            grad_logt * (rounded_scaled_x - scaled_x),
                            grad_logt)
    grad_logt = torch.where(is_lt_min, grad_logt * quant_min, grad_logt)
    grad_logt = torch.where(is_gt_max, grad_logt * quant_max, grad_logt)

    # grad_logt = grad_logt.sum().expand_as(logt)

    grad_x = grad_output.clone()
    grad_x = torch.where(
        is_ge_min_and_le_max, grad_x, 0 * grad_x)

    return grad_x, grad_logt, None, None



class Search_Pow2(Function):

    def forward(self, input):
        output = input
        output[output < 0].data.copy_(torch.Tensor([2 ** -5]))
        output[output > 2 ** (8 + 5)].data.copy_(torch.Tensor([2 ** (8 + 5)]))
        ceil_float_range = 2 ** output.log2().ceil()
        floor_float_range = 2 ** output.log2().floor()
        output[abs(ceil_float_range - output) < abs(floor_float_range - output)].data.copy_(ceil_float_range[
                                                                                                abs(ceil_float_range - output) < abs(
                                                                                                    floor_float_range - output)].data)
        output[abs(ceil_float_range - output) >= abs(floor_float_range - output)].data.copy_(floor_float_range[
                                                                                                 abs(ceil_float_range - output) >= abs(
                                                                                                     floor_float_range - output)].data)
        return output