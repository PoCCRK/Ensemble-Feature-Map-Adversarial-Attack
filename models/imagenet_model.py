import torch, torchvision
from pytorchcv.model_provider import get_model as ptcv_get_model

from data.imagenet_data import imagenet_normalize


def get_imagenet_pretrained_models_with_features():
    # https://pypi.org/project/pytorchcv/
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inceptionv3 = ptcv_get_model('inceptionv3', pretrained=True).to(device).eval()
    inceptionv4 = ptcv_get_model('inceptionv4', pretrained=True).to(device).eval()
    inceptionresnetv2 = ptcv_get_model('inceptionresnetv2', pretrained=True).to(device).eval()
    resnet101 = ptcv_get_model('resnet101', pretrained=True).to(device).eval()
    resnet152 = ptcv_get_model('resnet152', pretrained=True).to(device).eval()

    models = [
        inceptionv3,
        inceptionv4,
        inceptionresnetv2,
        #resnet152,
        ]

    print(f"number of imagenet pretrained models with features: {len(models)}")
    return models

def get_imagenet_transfer_models():
    # https://pypi.org/project/pytorchcv/
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    inceptionv3 = ptcv_get_model('inceptionv3', pretrained=True).to(device).eval()
    inceptionv4 = ptcv_get_model('inceptionv4', pretrained=True).to(device).eval()
    inceptionresnetv2 = ptcv_get_model('inceptionresnetv2', pretrained=True).to(device).eval()
    resnet101 = ptcv_get_model('resnet101', pretrained=True).to(device).eval()
    resnet152 = ptcv_get_model('resnet152', pretrained=True).to(device).eval()

    transfer_models = [
        resnet152,
        ]

    print(f"number of imagenet transfer models: {len(transfer_models)}")
    return transfer_models

class ImageNet_Model(torch.nn.Module):
    def __init__(self, model, name=''):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, x: torch.Tensor):
        x = imagenet_normalize(x)
        output = self.model.forward(x)

        return output

    def features(self, x: torch.Tensor):
        x = imagenet_normalize(x)
        features = self.model.features(x)

        return features


if __name__ == "__main__":
    models = get_imagenet_pretrained_models_with_features()
    transfer_models = get_imagenet_transfer_models()
    for model in models:
        print(model.name)
    for model in transfer_models:
        print(model.name)