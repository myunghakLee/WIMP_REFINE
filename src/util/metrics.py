import numpy as np
import torch

# +
# calc_dist = lambda A,B : ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)**0.5
# repeat = lambda tensor, num_reps, k = 6 : torch.cat([a.unsqueeze(1).repeat(1,k).flatten() for a in tensor], axis=0)#.reshape(len(tensor), num_reps**2)


def compute_metrics(prediction, truth, mean=True, on_gpu=True, miss_threshold=2.0):
    """Compute the required evaluation metrics: ADE, FDE, and MR
        Args:
            prediction (array): predicted trajectories
            truth (array): ground truth trajectory
        Returns:
            ade (float): Average Displacement Error
            fde (float): Final Displacement Error
            mr (float): Miss Rate
    """
    if on_gpu:
        truth = truth.unsqueeze(1)
        l2_all = torch.sqrt(torch.sum((prediction - truth)**2, dim=-1))
        ade_all = torch.sum(l2_all, dim=-1) / prediction.size(-2)
        fde_all = l2_all[..., -1]
        min_fde = torch.argmin(fde_all, dim=-1)
        indices = torch.arange(prediction.shape[0], device=min_fde.get_device())
        fde = fde_all[indices, min_fde]
        ade = ade_all[indices, min_fde]
        miss = (fde > miss_threshold).float()
        
        if mean:
            return torch.mean(ade), torch.mean(fde), torch.mean(miss)
        else:
            return ade, fde, miss
    else:
        truth = np.expand_dims(truth, 1)
        l2_all = np.sqrt(np.sum((prediction - truth)**2, axis=-1))
        ade_all = np.sum(l2_all, axis=-1) / prediction.shape[-2]
        fde_all = l2_all[..., -1]
        min_fde = np.argmin(fde_all, axis=-1)
        indices = np.arange(prediction.shape[0])
        fde = fde_all[indices, min_fde]
        ade = ade_all[indices, min_fde]
        miss = (fde > miss_threshold).astype(np.float32)
        if mean:
            return np.mean(ade), np.mean(fde), np.mean(miss)
        else:
            return ade, fde, miss


# -

# import numpy as np
# import torch
torch.mean(torch.tensor([2.0,3]))
# A = np.full((1,1,30,2), 1)
# B = np.full((1,1,30,2), 2)
# C = np.full((1,1,30,2), 3)
# D = np.full((1,1,30,2), 4)
# E = np.full((1,1,30,2), 5)
# F = np.full((1,1,30,2), 6)

# +
# all = np.concatenate((A,B,C,D,E,F), axis=1)
# all2 = np.concatenate((A*2,B*2,C*2,D*2,E*2,F*2), axis=1)
# all = np.concatenate((all, all2))
# all = torch.from_numpy(all)[:,:,-1,:]
# all.shape

# +

# calc_dist = lambda A,B : ((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)**0.5
# repeat = lambda tensor, num_reps, k = 6 : torch.cat([a.unsqueeze(1).repeat(1,k).flatten() for a in tensor], axis=0)#.reshape(len(tensor), num_reps**2)


# ((repeat(all[:,:,0],6) - all[:,:,0].repeat(1,6).flatten()) ** 2 + (repeat(all[:,:,1],6) - all[:,:,1].repeat(1,6).flatten()) ** 2) ** 0.5
# -


