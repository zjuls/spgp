import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.optim as optim

class Rectangle(torch.autograd.Function):
    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        return inpt.gt(0).float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = (torch.abs(inpt) < 0.5).float()
        return grad_input * sur_grad

class PDF(torch.autograd.Function):

    alpha = 0.1
    beta = 0.1

    @staticmethod
    def forward(self, inpt):
        self.save_for_backward(inpt)
        p = torch.ones(inpt.size(), device=device) - torch.exp(-1 * inpt)
        p2 = torch.rand(inpt.size(),device = device)
        x = p > p2
        return x.float()

    @staticmethod
    def backward(self, grad_output):
        inpt, = self.saved_tensors
        grad_input = grad_output.clone()
        sur_grad = PDF.alpha * torch.exp(-PDF.beta * torch.abs(inpt))
        return sur_grad * grad_input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Act = Linear.apply
Act = Rectangle.apply
Act_spgp = PDF.apply
steps = 4 # timestep

def lr_scheduler(optimizer, epoch, init_lr=0.1, lr_decay_epoch=30):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    if epoch % lr_decay_epoch == 0 and epoch > 1:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * init_lr
    return optimizer

def state_update(u, o, i, decay, Vth):
    u = decay * u + i - o * Vth
    o = Act(u - Vth)
    return u, o

def accumulated_state(u, o):
    u_ = 0.5 * u + o
    return u_

class LIF(nn.Module):
    def __init__(self):
        super(LIF, self).__init__()
        init_decay = 0.2
        ini_v = 0.5

        #self.nrom = torch.norm(w.detach().cpu(), None, dim=None)
        self.decay = nn.Parameter(torch.tensor(init_decay, dtype=torch.float), requires_grad=True)
        self.decay.data.clamp_(0., 1.)
        self.vth = ini_v

    def forward(self, x, output = False, vmem = False):
        if output:
            if not vmem:
                u = torch.zeros(x.shape[:-1], device=x.device)
                for step in range(steps):
                    u = accumulated_state(u, x[..., step])
                return u
            else:
                u = torch.zeros(x.shape[:-1], device=x.device)
                out = torch.zeros(x.shape, device=x.device)
                u_trace = torch.zeros(x.shape, device=x.device)
                for step in range(steps):
                    u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], self.decay, self.vth)
                    u_trace[..., step] = u
                return u_trace

        else:
            u = torch.zeros(x.shape[:-1], device=x.device)
            out = torch.zeros(x.shape, device=x.device)
            for step in range(steps):
                u, out[..., step] = state_update(u, out[..., max(step - 1, 0)], x[..., step], self.decay, self.vth)
            return out



class tdLayer(nn.Module):
    def __init__(self, layer,):
        super(tdLayer, self).__init__()
        self.layer = layer

    def forward(self, x):
        x_ = torch.zeros(self.layer(x[..., 0]).shape + (steps,), device = device)
        for step in range(steps):
            x_[..., step] = self.layer(x[..., step])
        return x_

class TemporalBN(nn.Module):
    def __init__(self, in_channels, nb_steps):
        super(TemporalBN, self).__init__()
        self.nb_steps = nb_steps
        self.bns = nn.ModuleList([nn.BatchNorm2d(in_channels) for t in range(self.nb_steps)])

    def forward(self, x):
        out = []
        stack_dim = len(x.shape) - 1
        for t in range(self.nb_steps):
            out.append(self.bns[t](x[..., t]))
        out = torch.stack(out, dim=stack_dim)
        return out

