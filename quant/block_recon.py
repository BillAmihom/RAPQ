import torch
import linklink as link
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
from quant.quant_model import QuantModel
from quant.quant_block import BaseQuantBlock
from quant.adaptive_rounding import AdaRoundQuantizer
from quant.data_utils import save_grad_data, save_inp_oup_data


def block_reconstruction(model: QuantModel, block: BaseQuantBlock, cali_data: torch.Tensor,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         multi_gpu: bool = False,RAPQ:bool = True):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
    :param cali_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    """
    model.set_quant_state(False, False)
    block.set_quant_state(True, act_quant)
    round_mode = 'learned_hard_sigmoid'

    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        # Replace weight quantizer to AdaRoundQuantizer
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            weight_tensor=module.org_weight.data,RAPQ=RAPQ)
                module.weight_quantizer.soft_targets = True
                module.weight_quantizer.p2_soft_targets = True

        # Set up optimizer
        opt_params = []
        p2_opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                opt_params += [module.weight_quantizer.alpha]
                p2_opt_params += [module.weight_quantizer.p2_alpha]
                # p2_opt_params += [module.weight_quantizer.p2_alpha]
                # if module.weight_quantizer.delta is not None:
                #     opt_params += [module.weight_quantizer.delta]
                # if module.weight_quantizer.zero_point is not None:
                #     opt_params += [module.weight_quantizer.zero_point]
                # if module.weight_quantizer.log_threshold is not None:
                #     opt_params += [module.weight_quantizer.log_threshold]
        # optimizer = torch.optim.Adam(opt_params)
        param_groups = [
        {'params': opt_params, 'name': 'opt_params'},
        {'params': p2_opt_params, 'name': 'p2_opt_params'},
        ]
        optimizer = torch.optim.Adam(param_groups)
        scheduler = None
    else:
        # Use UniformAffineQuantizer to learn delta
        # if hasattr(block.act_quantizer, 'delta'):
        #     opt_params = [block.act_quantizer.delta]
        # else:
        #     opt_params = []
        if hasattr(block.act_quantizer, 'log_threshold'):
            opt_params = [block.act_quantizer.log_threshold]
        else:
            opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantModule):
                # if module.act_quantizer.delta is not None:
                #     opt_params += [module.act_quantizer.delta]
                # if module.act_quantizer.zero_point is not None:
                #     opt_params += [module.act_quantizer.zero_point]
                if module.act_quantizer.log_threshold is not None:
                    opt_params += [module.act_quantizer.log_threshold]
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p,act_quant=act_quant)

    # Save data before optimizing the rounding
    cached_inps, cached_outs = save_inp_oup_data(model, block, cali_data, asym, act_quant, batch_size)
    if opt_mode != 'mse':
        cached_grads = save_grad_data(model, block, cali_data, act_quant, batch_size=batch_size)
    else:
        cached_grads = None
    device = 'cuda'
    for i in range(iters):

        # if (i == iters * warmup) and (not act_quant):
        if (i == 500) and (not act_quant):
            for name, module in block.named_modules():
                if isinstance(module, QuantModule):
                    module.weight_quantizer.p2_soft_targets = False

        # if (i >= iters * warmup) and (not act_quant):
        if (i >= 500) and (not act_quant):
            for param_group in optimizer.param_groups:
                if param_group['name'] == 'p2_opt_params':
                    param_group['lr'] = 0

        idx = torch.randperm(cached_inps.size(0))[:batch_size]
        cur_inp = cached_inps[idx].to(device)
        cur_out = cached_outs[idx].to(device)
        cur_grad = cached_grads[idx].to(device) if opt_mode != 'mse' else None

        optimizer.zero_grad()
        out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        err.backward(retain_graph=True)
        if multi_gpu:
            for p in opt_params:
                link.allreduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for name, module in block.named_modules():
        if isinstance(module, QuantModule):
            module.weight_quantizer.soft_targets = False
            #module.weight_quantizer.p2_soft_targets = False

    # Reset original activation function
    if not include_act_func:
        block.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 block: BaseQuantBlock,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 act_quant:bool = False):

        self.block = block
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.act_quant = act_quant

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':
            rec_loss = lp_loss(pred, tgt, p=self.p)
        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)

        p2_round_loss = 0
        #用于计算p2_round_loss的b值是没有归零的
        # if (self.count < self.loss_start) and (not self.act_quant):
        if (self.count < 500) and (not self.act_quant):
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    p2_round_vals = module.weight_quantizer.p2_get_soft_targets()
                    p2_round_loss += self.weight * 10 * (1 - ((p2_round_vals - .5).abs() * 2).pow(b)).sum()

        # if (self.count < self.loss_start or self.round_loss == 'none') or (self.count < 20000):
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            for name, module in self.block.named_modules():
                if isinstance(module, QuantModule):
                    round_vals = module.weight_quantizer.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError


        total_loss = rec_loss + round_loss + p2_round_loss
        if self.count % 500 == 0:
            print('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f},p2_round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), float(p2_round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))
