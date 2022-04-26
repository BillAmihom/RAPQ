import torch
from torch import nn
from quant.quant_layer import UniformAffineQuantizer, round_ste


class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, uaq: UniformAffineQuantizer, weight_tensor: torch.Tensor, round_mode='learned_round_sigmoid,',RAPQ = True):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = uaq.n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.RAPQ = RAPQ

        # self.log_threshold = uaq.log_threshold
        # self.quantize_fn_cls = uaq.quantize_fn_cls


        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        self.init_alpha(x=weight_tensor.clone())

        self.p2_alpha = None
        self.p2_soft_targets = False
        self.init_p2_alpha(x=self.delta.detach())

    def forward(self, x):

        if self.RAPQ:
            delta_floor = torch.floor(self.delta.log2())
            if self.p2_soft_targets:
                delta_int = delta_floor + self.p2_get_soft_targets()
                delta = 2 ** delta_int
            else:
                delta_int = delta_floor + self.p2_get_soft_targets()
                delta = 2 ** delta_int
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

            # zero_point = torch.round((self.zero_point * self.delta) / delta) #fixed-point zero-point for hardware
            zero_point = (self.zero_point * self.delta) / delta #float-point zero-point for hardware

            if self.round_mode == 'nearest':
                x_int = torch.round(x / delta)
            elif self.round_mode == 'nearest_ste':
                x_int = round_ste(x / delta)
            elif self.round_mode == 'stochastic':#stochastic随机的
                x_floor = torch.floor(x / delta)
                rest = (x / delta) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print('Draw stochastic sample')
            elif self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor(x / delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                raise ValueError('Wrong rounding mode')

            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta

        else:
            if self.round_mode == 'nearest':
                x_int = torch.round(x / self.delta)
            elif self.round_mode == 'nearest_ste':
                x_int = round_ste(x / self.delta)
            elif self.round_mode == 'stochastic':  # stochastic随机的
                x_floor = torch.floor(x / self.delta)
                rest = (x / self.delta) - x_floor  # rest of rounding
                x_int = x_floor + torch.bernoulli(rest)
                print('Draw stochastic sample')
            elif self.round_mode == 'learned_hard_sigmoid':
                x_floor = torch.floor(x / self.delta)
                if self.soft_targets:
                    x_int = x_floor + self.get_soft_targets()
                else:
                    x_int = x_floor + (self.alpha >= 0).float()
            else:
                raise ValueError('Wrong rounding mode')

            x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def p2_get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.p2_alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
            # print(self.alpha.shape)
        else:
            raise NotImplementedError

    def init_p2_alpha(self, x: torch.Tensor):
        delta_floor = torch.floor(x.log2())
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init p2_alpha to be FP32')
            rest = (x.log2()) - delta_floor  # rest of rounding [0, 1)
            p2_alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(p2_alpha) = rest
            self.p2_alpha = nn.Parameter(p2_alpha)
        else:
            raise NotImplementedError
