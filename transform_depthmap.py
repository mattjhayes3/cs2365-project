# load and show an image with Pillow
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
# Open the image form working directory
target = Image.open('/Users/matthewhayes/Downloads/depth_test.png').convert('L')
hand = Image.open('/Users/matthewhayes/Downloads/adjust hand for rotation only test.png').convert('LA')
hand.show()

# summarize some details about the image
print(hand.format)
print(hand.size)
print(hand.mode)
# show the image
# array = np.asarray(hand)
# hand.show()
# print(array.shape)
torch.set_printoptions(threshold=10_000_000)
tensor = torchvision.transforms.functional.to_tensor(hand).unsqueeze(0)
target = torchvision.transforms.functional.to_tensor(target).unsqueeze(0)
# print(tensor[0, 0])
print("tensor shape",tensor.shape, "type", tensor.type())
print("target shape",target.shape, "type", target.type())

# t = torch.nn.Parameter(torch.tensor(torch.pi))
t = torch.nn.Parameter(torch.tensor(0.))
t.requires_grad = True
scale = torch.tensor(1)

def apply(tensor):
    print(f"t=", t.item())
    theta = torch.tensor([[scale * torch.cos(t), -torch.sin(t), 0],
                         [torch.sin(t), torch.cos(t), 0]]).repeat(tensor.shape[0], 1, 1)
    theta.requires_grad = True
    # theta2 = torch.stack([
    #             torch.stack([
    #                 scale * torch.cos(t).unsqueeze(dim=0), 
    #                 -torch.sin(t).unsqueeze(dim=0), 
    #                 torch.zeros(1)]), 
    #             torch.stack([
    #                 torch.sin(t).unsqueeze(dim=0), 
    #                 scale * torch.cos(t).unsqueeze(dim=0), 
    #                 torch.zeros(1)])]).squeeze().repeat(tensor.shape[0], 1, 1)
    # print('theta = ', theta)
    # print('theta2 = ', theta2)
    grid = F.affine_grid(theta, tensor.size())
    return F.grid_sample(tensor, grid, mode = "bilinear")

def loss(tensor, target):
    alpha = tensor[0,0]
    diff = alpha * (target[0, 0] - tensor[0, 0])
    return torch.linalg.vector_norm(diff)

optimizer = torch.optim.SGD([t], lr=0.1)
for step in range(5000):
    optimizer.zero_grad()
    new_tensor = apply(tensor)
    l = loss(new_tensor, target)
    print('loss=', l.item())
    # l.backward(retain_graph=True)
    l.backward()
    optimizer.step()
    tensor = new_tensor.detach()


print("result", t.item())


pil = torchvision.transforms.functional.to_pil_image(tensor[0])
pil.show()