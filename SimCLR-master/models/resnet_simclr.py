import torch.nn as nn
import torchvision.models as models

from exceptions.exceptions import InvalidBackboneError


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features

        # add mlp projection head
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ResNetSimCLR_linearProbing(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_linearProbing, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.dim_mlp= dim_mlp

        # add mlp projection head
        # self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        self.backbone.fc = nn.Identity()
        # print(self.backbone)
        # self.backbone.fc = nn.Sequential(nn.BatchNorm1d(dim_mlp,), nn.Linear(dim_mlp, 2))
        self.head = nn.Linear(dim_mlp, 2)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        y = self.backbone(x)
        out = self.head(y)
        return out
class ResNetSimCLR_feature_representation(nn.Module):

    def __init__(self, base_model, out_dim=2):
        super(ResNetSimCLR_feature_representation, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.dim_mlp = dim_mlp
        self.backbone.fc = nn.Identity()

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        y = self.backbone(x)
        return y




class ResNetSimCLR_linearProbing1(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR_linearProbing1, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)}

        backbone = self._get_basemodel(base_model)
        dim_mlp = backbone.fc.in_features
        del backbone.fc
        self.dim_mlp= dim_mlp
        self.backbone = backbone

        self.head = nn.Sequential(nn.BatchNorm1d(dim_mlp,), nn.Linear(dim_mlp, 2))

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        x = self.backbone(x)
        cls = self.head(x)
        return cls




if __name__ == '__main__':
    import torch
    device = torch.device('cuda:0')
    a = torch.randn((32,3, 224, 224))
    a = a.to(device)

    model =ResNetSimCLR_feature_representation('resnet50', 128)
    # model = models.resnet50(weights=None, num_classes=128)
    # print(model.fc)
    model.to(device)
    b = model(a)
    print(model)
    print(b.shape)


        # print(a)

