import torch,pdb
import torchvision
import torchvision.models as tv_models
import torch.nn.modules


def _tv_model(model_fn, pretrained, weights_enum_cls):
    """
    torchvision >= 0.13 deprecates pretrained= in favor of weights=.
    Keep backward compatibility with older torchvision.
    """
    if not pretrained:
        try:
            return model_fn(weights=None)
        except TypeError:
            return model_fn(pretrained=False)
    if weights_enum_cls is None:
        return model_fn(pretrained=True)
    try:
        return model_fn(weights=weights_enum_cls.IMAGENET1K_V1)
    except (TypeError, AttributeError):
        return model_fn(pretrained=True)


class vgg16bn(torch.nn.Module):
    def __init__(self,pretrained = False):
        super(vgg16bn,self).__init__()
        w_cls = getattr(tv_models, "VGG16_BN_Weights", None)
        vgg = _tv_model(tv_models.vgg16_bn, pretrained, w_cls)
        model = list(vgg.features.children())
        model = model[:33]+model[34:43]
        self.model = torch.nn.Sequential(*model)
        
    def forward(self,x):
        return self.model(x)
class resnet(torch.nn.Module):
    def __init__(self,layers,pretrained = False):
        super(resnet,self).__init__()
        if layers == '18':
            w_cls = getattr(tv_models, "ResNet18_Weights", None)
            model = _tv_model(tv_models.resnet18, pretrained, w_cls)
        elif layers == '34':
            w_cls = getattr(tv_models, "ResNet34_Weights", None)
            model = _tv_model(tv_models.resnet34, pretrained, w_cls)
        elif layers == '50':
            w_cls = getattr(tv_models, "ResNet50_Weights", None)
            model = _tv_model(tv_models.resnet50, pretrained, w_cls)
        elif layers == '101':
            w_cls = getattr(tv_models, "ResNet101_Weights", None)
            model = _tv_model(tv_models.resnet101, pretrained, w_cls)
        elif layers == '152':
            w_cls = getattr(tv_models, "ResNet152_Weights", None)
            model = _tv_model(tv_models.resnet152, pretrained, w_cls)
        elif layers == '50next':
            w_cls = getattr(tv_models, "ResNeXt50_32X4D_Weights", None)
            model = _tv_model(tv_models.resnext50_32x4d, pretrained, w_cls)
        elif layers == '101next':
            w_cls = getattr(tv_models, "ResNeXt101_32X8D_Weights", None)
            model = _tv_model(tv_models.resnext101_32x8d, pretrained, w_cls)
        elif layers == '50wide':
            w_cls = getattr(tv_models, "Wide_ResNet50_2_Weights", None)
            model = _tv_model(tv_models.wide_resnet50_2, pretrained, w_cls)
        elif layers == '101wide':
            w_cls = getattr(tv_models, "Wide_ResNet101_2_Weights", None)
            model = _tv_model(tv_models.wide_resnet101_2, pretrained, w_cls)
        elif layers == '34fca':
            model = torch.hub.load('cfzd/FcaNet', 'fca34' ,pretrained=True)
        else:
            raise NotImplementedError
        
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2,x3,x4
