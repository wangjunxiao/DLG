import torch
import numpy as np

'''
valid for ConvNet and LeNet, 1-batch, single image
'''
def fc_perturb(parameters, model, ground_truth, pruning_rate, setup):
    feature_fc1_graph = model.extract_feature()
    deviation_f1_target = torch.zeros_like(feature_fc1_graph)
    deviation_f1_x_norm = torch.zeros_like(feature_fc1_graph)
    for f in range(deviation_f1_x_norm.size(1)):
        deviation_f1_target[:,f] = 1
        feature_fc1_graph.backward(deviation_f1_target, retain_graph=True)
        deviation_f1_x = ground_truth.grad.data
        deviation_f1_x_norm[:,f] = torch.norm(deviation_f1_x.view(deviation_f1_x.size(0), -1), dim=1)/(feature_fc1_graph.data[:,f] + 0.1)
        model.zero_grad()
        ground_truth.grad.data.zero_()
        deviation_f1_target[:,f] = 0
        
    deviation_f1_x_norm_sum = deviation_f1_x_norm.sum(axis=0)
    thresh = np.percentile(deviation_f1_x_norm_sum.flatten().cpu().numpy(), pruning_rate)
    mask = np.where(abs(deviation_f1_x_norm_sum.cpu()) < thresh, 0, 1).astype(np.float32)
    parameters[-2] = parameters[-2] * torch.Tensor(mask).to(**setup)
    return parameters

