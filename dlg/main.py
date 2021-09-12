# -*- coding: utf-8 -*-
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from _utils import label_to_onehot, cross_entropy_for_onehot

# config
Iteration = 100
LogInterval = Iteration/10
ImgIndex = 25 # the index for leaking images on CIFAR100
perturb = False

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Running on %s" % device)

torch.cuda.empty_cache()

dst = datasets.CIFAR100("~/.torch", download=True)
tt = transforms.ToTensor()
tp = transforms.ToPILImage()

gt_data = tt(dst[ImgIndex][0]).to(device)

print ("image index:", ImgIndex, "image size:", *gt_data.size())
gt_data = gt_data.view(1, *gt_data.size())

gt_label = torch.Tensor([dst[ImgIndex][1]]).long().to(device)
gt_label = gt_label.view(1, )
gt_onehot_label = label_to_onehot(gt_label)

plt.imshow(tp(gt_data[0].cpu()))

from models.vision import LeNet, weights_init
net = LeNet().to(device)

torch.manual_seed(1234)

net.apply(weights_init)
criterion = cross_entropy_for_onehot

# compute original gradient 
pred = net(gt_data)
y = criterion(pred, gt_onehot_label)
dy_dx = torch.autograd.grad(y, net.parameters())

original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# perturb
if perturb:
    for i in range(len(original_dy_dx)):
        gradient_tensor = original_dy_dx[i].cpu().numpy() 
        gradient_tensor += np.random.laplace(0, 1e-4, gradient_tensor.shape)
        gradient_tensor = torch.Tensor(gradient_tensor).to(device)
        original_dy_dx[i] = gradient_tensor

# generate dummy data and label
dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

plt.imshow(tp(dummy_data[0].cpu()))

optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

history = []
for iters in range(Iteration):
    def closure():
        optimizer.zero_grad()

        dummy_pred = net(dummy_data) 
        dummy_onehot_label = F.softmax(dummy_label, dim=-1)
        dummy_loss = criterion(dummy_pred, dummy_onehot_label) 
        dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)
        
        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx): 
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()
        
        return grad_diff
    
    optimizer.step(closure)
    if iters % LogInterval == 0: 
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tp(dummy_data[0].cpu()))

plot_num = Iteration/LogInterval
plot_row = 2
plot_col = plot_num/plot_row
fig, axs = plt.subplots(plot_row, int(plot_col), figsize=(12.6, 4))
for r in range(plot_row):
    for i in range(int(plot_col)):
        axs[r][i].imshow(history[int(r*plot_col+i)])
        axs[r][i].set_title("iter=%d" % ((r*plot_col+i) * LogInterval))
        axs[r][i].set_xticks([])
        axs[r][i].set_yticks([])
fig.savefig((time.strftime("%m-%d %Hh%Mm%Ss", time.localtime())+'.pdf').replace('\\','/'))