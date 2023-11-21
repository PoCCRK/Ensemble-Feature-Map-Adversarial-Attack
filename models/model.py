import torch


def test_model(model, dataloader):
    total,correct = 0,0
    name = model.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, batch in enumerate (dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of {name} : {100 * correct / total}')
    return correct / total


def test_models(models, dataloader):
    for model in models:
        test_model(model, dataloader)
