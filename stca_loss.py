from layer import *

thresh = 0.5
sub_thresh = 0.05
C =1
class STCA_ClassifyLoss(nn.Module):
    def __init__(self):
        super(STCA_ClassifyLoss, self).__init__()
        print('Use Classification Loss in TCA ')

    def forward(self, vmem, labels):
        batch_size = vmem.shape[0]
        num_neuron = vmem.shape[1]
        num_time = vmem.shape[2]
        loss = torch.tensor(0.0, device=device)
        for ibatch in range(batch_size):
            for ineuron in range(num_neuron):
                v = vmem[ibatch, ineuron, :]
                pos_spike = torch.nonzero(v >= thresh)
                pos_spike = torch.reshape(pos_spike, [pos_spike.numel()])
                end_list = []
                beg_list = [0]
                for ispike in range(1, pos_spike.shape[0]):
                    differ = pos_spike[ispike] - pos_spike[ispike - 1]
                    if differ > C:
                        end_list.append(ispike - 1)
                        beg_list.append(ispike)
                end_list.append(pos_spike.shape[0] - 1)
                num_cluster = len(beg_list)
                if end_list[-1] < 0: num_cluster = 0
                end_list = torch.tensor(end_list, device=device)
                beg_list = torch.tensor(beg_list, device=device)
                vmax = v.max()
                if labels[ibatch, ineuron] > 0 and (num_cluster == 0 or thresh + sub_thresh > vmax):
                    loss += thresh + sub_thresh - vmax
                if labels[ibatch, ineuron] == 0 and num_cluster > 0:
                    idx_cluster = torch.argmin(end_list - beg_list)
                    loss += v[pos_spike[end_list[idx_cluster]]] - thresh + sub_thresh
                if labels[ibatch, ineuron] == 0 and thresh - sub_thresh <= vmax < thresh:
                    loss += vmax - thresh + sub_thresh
        return loss / batch_size

