import numpy as np
import torch, torchvision


imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

def imagenet_normalize(images):
    return torchvision.transforms.functional.normalize(images, imagenet_mean, imagenet_std)

def get_imagenet_val_dataloader(data_dir, n_images=None, batch_size=1):
    '''
        ImageNet 2012 Classification Dataset.
        see example in:
        https://github.com/pytorch/examples/blob/main/imagenet/main.py
        https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a
        https://github.com/PatrickHua/EasyImageNet
    '''
    val_transform = torchvision.transforms.Compose([
            #torchvision.transforms.Resize(256),
            #torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Resize((224,224)),
            torchvision.transforms.ToTensor(),
            #transforms.Normalize(imagenet_mean, imagenet_std),
        ])

    imagenet_dataset = torchvision.datasets.ImageNet(root=data_dir,
                                                     split='val',
                                                     transform=val_transform)

    dataset_size = len(imagenet_dataset)
    dataset_indices = list(range(dataset_size))
    if n_images is None:
        n_images = dataset_size

    # takes only n_images images to use
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    subset_idx = dataset_indices[:n_images]
    subset = torch.utils.data.Subset(imagenet_dataset, subset_idx)

    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        )

    print(f'imagenet num_images : {subset.__len__()}')
    return dataloader, imagenet_dataset.classes


if __name__ == "__main__":
    dataloader, classes = get_imagenet_val_dataloader(
        "D:/workspace/data/ImageNet/",
        batch_size=4)
    for x,y in dataloader:
        print(x.shape, y)
        x = imagenet_normalize(x)
        for label in y:
            print(classes[label])
