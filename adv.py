from argparse import ArgumentParser
from xmlrpc.server import resolve_dotted_attribute
import numpy as np
import torch, torchvision
import os, shutil
import pytorchcv, torchattacks
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from data import *
from models import *
from attacks.model_based_ensembling_attack import MBEL2, MBELINF
from attacks.approach import APPROACH, APPROACH2
from attacks.MEMIFGSM import MEMIFGSM

global_target_images_list = []

def get_next_label(images, labels):
    global global_target_images_list
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(global_target_images_list)
    target_labels = torch.zeros_like(labels)
    for j, label in enumerate(labels):
        index = label
        while True:
            index = (index+1)%n_classes
            if global_target_images_list[index] is not None:
                target_labels[j] = index
                break

    return target_labels.long().to(device)


def data_init(save_path, models, dataloader, classes):
    # selected images that can be classified correctly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(classes)
    count = 0
    
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    
    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        for i, model in enumerate(models):
            outputs = model(images)
            confidence, predicted = torch.max(outputs.data, 1)
            
            if i==0:
                results = torch.eq(labels, predicted)
            else:
                new_results = torch.eq(labels, predicted)
                results = torch.logical_and(results, new_results)

        for j, result in enumerate(results):
            if result:
                selected_image = images[j].cpu()
                selected_label = labels[j].cpu()
                count += 1
                name = f"{save_path}/{count}.pt"
                
                torch.save([selected_image, selected_label], name)
    
    print(f"n_selected {count}")

def anchor_init(save_path, models, dataloader, classes):
    # save the anchor images
    # please use the dataloader in seleced_images
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(classes)
    count = 0
    anchor_images = [None] * n_classes
    anchor_labels = [None] * n_classes
    anchor_confidences = [0] * n_classes

    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)

    for batch in dataloader:
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        
        for i, model in enumerate(models):
            output = model(images)
            confidence, predicted = torch.max(output.data, 1)
            confidences = confidence if i==0 else confidences + confidence

        for j, confidence in enumerate(confidences):
            label = labels[j]
            image = images[j]
            if anchor_confidences[label] < confidence:
                anchor_confidences[label] = confidence
                anchor_images[label] = image
                anchor_labels[label] = label

    for image, label in zip(anchor_images, anchor_labels): 
        if image!=None and label!=None:
            anchor_image = image.cpu()
            anchor_label = label.cpu()
            count += 1
            name = f"{save_path}/{count}.pt"
                
            torch.save([anchor_image, anchor_label], name)

    print(f"n_anchor {count}")

def load_selected(save_path, batch_size=1, n_images=None):
    selected_dataset = FolderDataset(data_dir=save_path)
    
    dataset_size = len(selected_dataset)
    dataset_indices = list(range(dataset_size))
    if n_images is None:
        n_images = dataset_size

    # takes only n_images images to use
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    subset_idx = dataset_indices[:n_images]
    subset = torch.utils.data.Subset(selected_dataset, subset_idx)

    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        )

    print(f'num_images : {subset.__len__()}')
    return dataloader

def load_anchor(save_path, classes):
    global global_target_images_list
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_classes = len(classes)
    target_image = [None] * n_classes

    files = os.listdir(save_path)
    for f in files:
        image, label = torch.load(f"{save_path}/{f}")
        target_image[label] = image

    global_target_images_list = target_image
    return target_image


def adv_epoch(name, data_path, attack, dataloader, classes, models, transfer_models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = attack.__class__.__name__ + name
    print(f"\nadv_epoch of {name}")
    MSELoss = torch.nn.MSELoss(reduction='none')
    Flatten = torch.nn.Flatten()

    image_list,adv_image_list,label_list,target_list = [],[],[],[]
    for i, batch in enumerate(dataloader):
        images, labels = batch
        image_list.append(images.detach().cpu())
        images, labels = images.to(device), labels.to(device)
        target_labels = attack._get_target_label(images, labels)
        adv_images = attack(images, labels)

        adv_images = adv_images.clamp(0, 1)
        adv_images = (adv_images * 255).type(torch.uint8).clamp(0, 255)
        adv_images = (adv_images / 255).type(torch.get_default_dtype()).clamp(0, 1)

        adv_image_list.append(adv_images.detach().cpu())
        label_list.append(labels.detach().cpu())
        target_list.append(target_labels.detach().cpu())
    image_list_cat = torch.cat(image_list, 0)
    adv_image_list_cat = torch.cat(adv_image_list, 0)
    label_list_cat = torch.cat(label_list, 0)
    target_list_cat = torch.cat(target_list, 0)

    #print(adv_image_list_cat.shape, image_list_cat.shape)
    #print(Flatten(adv_image_list_cat).shape, Flatten(image_list_cat).shape)
    L2 = MSELoss(Flatten(image_list_cat), Flatten(adv_image_list_cat))
    #print(L2.shape)
    L2 = L2.sum(dim=1)
    #print(L2.shape)
    #print(L2)

    torch.save((adv_image_list_cat, label_list_cat, target_list_cat), f'{data_path}/adv_data/{name}.pt')
    adv_images, labels, targets = torch.load(f'{data_path}/adv_data/{name}.pt')
    adv_data = ADVTensorDatasetWithTargetLabel(adv_images, labels, targets)
    adv_loader = DataLoader(adv_data, batch_size=8, shuffle=False)
    save_dataloader_to_images(f'{data_path}/adv_data/{name}', adv_loader, len(classes), classes)
    
    total = 0
    untarget_success, target_success = [0]*len(models), [0]*len(models)
    transfer_untarget_success, transfer_target_success = [0]*len(transfer_models), [0]*len(transfer_models)
    for i, batch in enumerate(adv_loader):
        adv_images, labels, targets = batch
        adv_images, labels, targets = adv_images.to(device), labels.to(device), targets.to(device)
        total += labels.size(0)

        for j, model in enumerate(models):
            outputs = model(adv_images)
            confidence, predicted = torch.max(outputs.data, 1)
            
            untarget_success[j] += (predicted != labels).sum().item()
            target_success[j] += (predicted == targets).sum().item()

        for j, model in enumerate(transfer_models):
            outputs = model(adv_images)
            confidence, predicted = torch.max(outputs.data, 1)

            transfer_untarget_success[j] += (predicted != labels).sum().item()
            transfer_target_success[j] += (predicted == targets).sum().item()

    for j, model in enumerate(models):
        print(f"model: {model.name} "
              f"untarget_success: {100 * untarget_success[j] / total} "
              f"target_success: {100 * target_success[j] / total} ")
    for j, model in enumerate(transfer_models):
        print(f"tansfer model: {model.name} "
              f"untarget_success: {100 * transfer_untarget_success[j] / total} "
              f"target_success: {100 * transfer_target_success[j] / total} ")

def untargeted_adv_epoch(name, data_path, attack, dataloader, classes, models, transfer_models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    name = attack.__class__.__name__ + name
    print(f"\nadv_epoch of {name}")
    MSELoss = torch.nn.MSELoss(reduction='none')
    Flatten = torch.nn.Flatten()

    image_list,adv_image_list,label_list = [],[],[]
    for i, batch in enumerate(dataloader):
        images, labels = batch
        image_list.append(images.detach().cpu())
        images, labels = images.to(device), labels.to(device)
        adv_images = attack(images, labels)

        adv_images = adv_images.clamp(0, 1)
        adv_images = (adv_images * 255).type(torch.uint8).clamp(0, 255)
        adv_images = (adv_images / 255).type(torch.get_default_dtype()).clamp(0, 1)

        adv_image_list.append(adv_images.detach().cpu())
        label_list.append(labels.detach().cpu())
    image_list_cat = torch.cat(image_list, 0)
    adv_image_list_cat = torch.cat(adv_image_list, 0)
    label_list_cat = torch.cat(label_list, 0)

    #print(adv_image_list_cat.shape, image_list_cat.shape)
    #print(Flatten(adv_image_list_cat).shape, Flatten(image_list_cat).shape)
    L2 = MSELoss(Flatten(image_list_cat), Flatten(adv_image_list_cat))
    #print(L2.shape)
    L2 = L2.sum(dim=1)
    #print(L2.shape)
    #print(L2)

    torch.save((adv_image_list_cat, label_list_cat), f'{data_path}/adv_data/untargeted_{name}.pt')
    adv_images, labels = torch.load(f'{data_path}/adv_data/untargeted_{name}.pt')
    adv_data = ADVTensorDataset(adv_images, labels)
    adv_loader = DataLoader(adv_data, batch_size=8, shuffle=False)
    save_dataloader_to_images(f'{data_path}/adv_data/untargeted_{name}', adv_loader, len(classes), classes)
    
    total = 0
    untarget_success = [0]*len(models)
    transfer_untarget_success = [0]*len(transfer_models)
    for i, batch in enumerate(adv_loader):
        adv_images, labels = batch
        adv_images, labels = adv_images.to(device), labels.to(device)
        total += labels.size(0)

        for j, model in enumerate(models):
            outputs = model(adv_images)
            confidence, predicted = torch.max(outputs.data, 1)
            
            untarget_success[j] += (predicted != labels).sum().item()

        for j, model in enumerate(transfer_models):
            outputs = model(adv_images)
            confidence, predicted = torch.max(outputs.data, 1)

            transfer_untarget_success[j] += (predicted != labels).sum().item()

    for j, model in enumerate(models):
        print(f"model: {model.name} "
              f"untarget_success: {100 * untarget_success[j] / total} ")
    for j, model in enumerate(transfer_models):
        print(f"tansfer model: {model.name} "
              f"untarget_success: {100 * transfer_untarget_success[j] / total} ")

def cifar_main(args):
    data_path = args[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    cifar10_models = get_cifar10_pretrained_models_with_features()
    transfer_models = get_cifar10_transfer_models()
    cifar10_loader, cifar10_classes = get_cifar10_test_dataloader(data_dir=f'{data_path}/cifar10/', n_images=None, batch_size=8)
    test_models(cifar10_models, cifar10_loader)

    if os.path.exists(f"{data_path}/cifar10_selected") is False:
        data_init(f"{data_path}/cifar10_selected", cifar10_models, cifar10_loader, cifar10_classes)
    selected_loader = load_selected(f"{data_path}/cifar10_selected", batch_size=8, n_images=100)
    if os.path.exists(f"{data_path}/cifar10_anchor") is False:
        anchor_init(f"{data_path}/cifar10_anchor", cifar10_models, selected_loader, cifar10_classes)
    load_anchor(f"{data_path}/cifar10_anchor", cifar10_classes)
    test_models(cifar10_models, selected_loader)

    save_dataloader_to_images(f'{data_path}/adv_data/cifar10_benign', selected_loader, len(cifar10_classes))


    #epss = [0, 1/255, 4/255, 8/255, 16/255]
    #for eps in epss:
    #    mbea = MBELINF(cifar10_models, rho=1/255, eps=eps, steps=20, alpha=None)
    #    mbea.set_mode_targeted_by_function(get_next_label)
    #    adv_epoch(f"eps{eps}", data_path, mbea, selected_loader, cifar10_classes, cifar10_models, transfer_models)
        
    #    mbmifgsm = MEMIFGSM(cifar10_models, eps=eps, alpha=1/255, steps=20)
    #    mbmifgsm.set_mode_targeted_by_function(get_next_label)
    #    adv_epoch(f"eps{eps}", data_path, mbmifgsm, selected_loader, cifar10_classes, cifar10_models, transfer_models)

    #    approach = APPROACH(cifar10_models, eta=1, rho=1/255, eps=eps, steps=20, alpha=None, target_images_list=global_target_images_list)
    #    approach.set_mode_targeted_by_function(get_next_label)
    #    adv_epoch(f"eps{eps}", data_path, approach, selected_loader, cifar10_classes, cifar10_models, transfer_models)



def imagenet_main(args):
    data_path = args[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    imagenet_models = get_imagenet_pretrained_models_with_features()
    transfer_models = get_imagenet_transfer_models()
    imagenet_loader, imagenet_classes = get_imagenet_val_dataloader(data_dir=f'{data_path}/ImageNet/', n_images=None, batch_size=8)
    test_models(imagenet_models, imagenet_loader)

    if os.path.exists(f"{data_path}/imagenet_selected") is False:
        data_init(f"{data_path}/imagenet_selected", imagenet_models, imagenet_loader, imagenet_classes)
    selected_loader = load_selected(f"{data_path}/imagenet_selected", batch_size=8, n_images=100)
    if os.path.exists(f"{data_path}/imagenet_anchor") is False:
        anchor_init(f"{data_path}/imagenet_anchor", imagenet_models, selected_loader, imagenet_classes)
    load_anchor(f"{data_path}/imagenet_anchor", imagenet_classes)
    test_models(imagenet_models, selected_loader)

    save_dataloader_to_images(f'{data_path}/adv_data/imagenet_benign', selected_loader, len(imagenet_classes))

    #mbea = MBELINF(imagenet_models, rho=4/255, eps=32/255, steps=10, alpha=None)
    #mbea.set_mode_targeted_by_function(get_next_label)
    #adv_epoch(f"eps32", data_path, mbea, selected_loader, imagenet_classes, imagenet_models, transfer_models)

    #etas = [0, 0.001, 0.1, 0.5, 0.7, 1, 1.2, 2]
    #for eta in etas:
    #    approach = APPROACH(imagenet_models, eta=eta, rho=4/255, eps=32/255, steps=10, alpha=None, target_images_list=global_target_images_list)
    #    approach.set_mode_targeted_by_function(get_next_label)
    #    adv_epoch(f"eps32eta{eta}", data_path, approach, selected_loader, imagenet_classes, imagenet_models, transfer_models)

    #etas = [0, 0.001, 0.1, 0.5, 0.7, 1, 1.2, 2]
    #for eta in etas:
    #    approach = APPROACH(imagenet_models, eta=eta, rho=4/255, eps=16/255, steps=10, alpha=None, target_images_list=global_target_images_list)
    #    approach.set_mode_targeted_by_function(get_next_label)
    #    adv_epoch(f"eps16eta{eta}", data_path, approach, selected_loader, imagenet_classes, imagenet_models, transfer_models)


if __name__ == "__main__":
    #parser = ArgumentParser()
    #parser.add_argument('data_path', type=str, nargs='+', default='./data'
    #                help='data path of image (cifar10 and ImageNet)')

    #args = parser.parse_args()
    cifar_main(["./data"])
    imagenet_main(["./data"])