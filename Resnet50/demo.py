## test linearprobing
import torchvision
import torch.nn as nn
import torch
# torch.manual_seed(10)

device = torch.device('cuda:0')
src1 = './checkpoints/resnet50-11ad3fa6.pth'
src2 = '/mnt/media_nvme/Process/cellSeg/ResNet50_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/ResNet50_DDP/LineProb/linProb_Group1/ratio_100-100/BN-one-fc-blr-1.5e-3-Resize-Totensor-SGD-epoch100/checkpoint_099.pth'
# resnet_pretrained = torch.load(src1)
# fc_checkpoint = torch.load(src2, map_location='cpu')
# lp_checkpoint = fc_checkpoint['state_dict']
# for key, val in list(resnet_pretrained.items()):
#     print(key)
# print('OK')





#
model = torchvision.models.resnet50(weights=None)  # num_classes =2
resnet_pretrained = torch.load(src1)
model.load_state_dict(resnet_pretrained)
# print(model)
fc_infeature = model.fc.in_features
model.fc = nn.Sequential(nn.BatchNorm1d(fc_infeature, ), nn.Linear(fc_infeature, 2))

fc_checkpoint = torch.load(src2, map_location='cpu')
model.load_state_dict(fc_checkpoint['state_dict'])
model.to(device)

a = torch.rand(2, 3, 224, 224)
a = a.to(device)
print(a)

b = model(a)
print(b)



pretrained ='../ResNet50_3090_output/output_dir/sampler-80K-40K-9groups_Remote-lsj3090/Group1/ResNet50_DDP/Finetune/epoch_50/Finetune_Group1/ratio_100-100/one-fc-blr-5e-4-Resize-Totensor-SGD-epoch50/checkpoint_049.pth'
model = torchvision.models.resnet50(weights=None)  # num_classes =2
model.fc = nn.Identity()
checkpoint_model = torch.load(pretrained, map_location='cpu')['state_dict']

for keys, values in checkpoint_model.items():
    print(keys)
# model.to(device)



#
# for key, val in list(resnet_pretrained.items()):
#     print(key)
# print('OK')
