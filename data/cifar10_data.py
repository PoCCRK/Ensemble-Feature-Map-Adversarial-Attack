import numpy as np
import torch, torchvision


cifar10_mean = [0.4914, 0.4822, 0.4465]
cifar10_std = [0.2471, 0.2435, 0.2616]

def cifar10_normalize(images):
    return torchvision.transforms.functional.normalize(images, cifar10_mean, cifar10_std)

def get_cifar10_test_dataloader(data_dir, n_images=None, batch_size=1):
    transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.ToTensor(),
            #transforms.Normalize(cifar10_mean, cifar10_std),
        ])

    cifar10_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    dataset_size = len(cifar10_dataset)
    dataset_indices = list(range(dataset_size))
    if n_images is None:
        n_images = dataset_size

    # takes only n_images images to use
    np.random.seed(0)
    np.random.shuffle(dataset_indices)
    subset_idx = dataset_indices[:n_images]
    subset = torch.utils.data.Subset(cifar10_dataset, subset_idx)

    dataloader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
    )

    print(f'cifar10 num_images : {subset.__len__()}')
    return dataloader, cifar10_dataset.classes


if __name__ == "__main__":
    dataloader, classes = get_cifar10_test_dataloader(
        'D:/workspace/data/cifar10/',
        batch_size=4)
    for x,y in dataloader:
        print(x.shape, y)
        x = cifar10_normalize(x)
        for label in y:
            print(classes[label])
