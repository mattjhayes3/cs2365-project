# load and show an image with Pillow
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
# Open the image form working directory
# target = Image.open('/Users/matthewhayes/Downloads/depth_test.png').convert('L')
# hand = Image.open('/Users/matthewhayes/Downloads/hand_depth_1.png').convert('LA')
# hand.show()

# summarize some details about the image
# print(hand.format)
# print(hand.size)
# print(hand.mode)
# # show the image
# # array = np.asarray(hand)
# # hand.show()
# # print(array.shape)
# torch.set_printoptions(threshold=10_000_000)
# tensor = torchvision.transforms.functional.to_tensor(hand).unsqueeze(0)
# target = torchvision.transforms.functional.to_tensor(target).unsqueeze(0)
# # print(tensor[0, 0])
# print("tensor shape",tensor.shape, "type", tensor.type())
# print("target shape",target.shape, "type", target.type())

# # t = torch.nn.Parameter(torch.tensor(torch.pi))
# t = torch.nn.Parameter(torch.tensor(np.pi))
# scale = torch.tensor(1)

# theta = torch.tensor([[scale * torch.cos(t), -torch.sin(t), 0, 0],
#                         [torch.sin(t), torch.cos(t), 0, 0],
#                         [0, 0, 1, 0]]).repeat(tensor.shape[0], 1, 1)
# print("theta shape", theta.shape)
#.repeat(tensor.shape[0], 1, 1)
import cv2
target = cv2.imread('/Users/matthewhayes/Downloads/depth_test.png')
hand = cv2.imread('/Users/matthewhayes/Downloads/hand_depth_1.png')

cv2.imshow('source', hand)



theta = np.array([[-9.02781766e-01, -7.89939748e-02,  3.15610357e+02],
                      [-4.49835471e-01, -3.93708525e-02,  1.57236146e+02],
                      [-2.86039819e-03, -2.50269682e-04,  1.00000000e+00]])#.repeat(tensor.shape[0], 1, 1)

# theta = np.array([
#     [1., 0., 0.],
#     [0, 1., 0.],
#     [0, 0, 1.]])

result = cv2.warpPerspective(hand, theta, (target.shape[1], target.shape[0]))
cv2.imshow('result', result)
cv2.waitKey(0)

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
# grid = F.affine_grid(theta, tensor.size())
# tensor =  F.grid_sample(tensor, grid, mode = "bilinear")
# # tensor = theta * tensor

# pil = torchvision.transforms.functional.to_pil_image(tensor[0])
# pil.show()