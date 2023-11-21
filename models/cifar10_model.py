import torch
from pytorchcv.model_provider import get_model as ptcv_get_model

from data.cifar10_data import cifar10_normalize


def get_cifar10_pretrained_models_with_features():
    # https://pypi.org/project/pytorchcv/
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resnet20_cifar10 = CIFAR10_Model(ptcv_get_model('resnet20_cifar10', pretrained=True).to(device).eval(), 'resnet20_cifar10')
    diaresnet20_cifar10 = CIFAR10_Model(ptcv_get_model('diaresnet20_cifar10', pretrained=True).to(device).eval(), 'diaresnet20_cifar10')
    resnext29_32x4d_cifar10 = CIFAR10_Model(ptcv_get_model('resnext29_32x4d_cifar10', pretrained=True).to(device).eval(), 'resnext29_32x4d_cifar10')
    densenet40_k12_cifar10 = CIFAR10_Model(ptcv_get_model('densenet40_k12_cifar10', pretrained=True).to(device).eval(), 'densenet40_k12_cifar10')
    pyramidnet110_a48_cifar10 = CIFAR10_Model(ptcv_get_model('pyramidnet110_a48_cifar10', pretrained=True).to(device).eval(), 'pyramidnet110_a48_cifar10')

    models = [
        resnet20_cifar10,
        diaresnet20_cifar10,
        resnext29_32x4d_cifar10,
        densenet40_k12_cifar10,
        pyramidnet110_a48_cifar10,
        ]

    print(f"number of cifar10 pretrained models with features: {len(models)}")
    return models

def get_cifar10_transfer_models():
    # https://pypi.org/project/pytorchcv/
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nin_cifar10 = CIFAR10_Model(ptcv_get_model('nin_cifar10', pretrained=True).to(device).eval(), 'nin_cifar10')
    preresnet20_cifar10 = CIFAR10_Model(ptcv_get_model('preresnet20_cifar10', pretrained=True).to(device).eval(), 'preresnet20_cifar10')
    wrn16_10_cifar10 = CIFAR10_Model(ptcv_get_model('wrn16_10_cifar10', pretrained=True).to(device).eval(), 'wrn16_10_cifar10')
    ror3_56_cifar10 = CIFAR10_Model(ptcv_get_model('ror3_56_cifar10', pretrained=True).to(device).eval(), 'ror3_56_cifar10')
    
    transfer_models = [
        nin_cifar10,
        preresnet20_cifar10,
        wrn16_10_cifar10,
        ror3_56_cifar10,
        ]

    print(f"number of cifar10 transfer models: {len(transfer_models)}")
    return transfer_models

class CIFAR10_Model(torch.nn.Module):
    def __init__(self, model, name=''):
        super().__init__()
        self.model = model
        self.name = name

    def forward(self, x: torch.Tensor):
        x = cifar10_normalize(x)
        output = self.model.forward(x)

        return output

    def features(self, x: torch.Tensor):
        x = cifar10_normalize(x)
        features = self.model.features(x)

        return features


if __name__ == "__main__":
    models = get_cifar10_pretrained_models_with_features()
    transfer_models = get_cifar10_transfer_models()
    for model in models:
        print(model.name)
    for model in transfer_models:
        print(model.name)