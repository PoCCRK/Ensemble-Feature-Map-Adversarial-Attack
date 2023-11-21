import os, re
import numpy as np
import torch
from PIL import Image

def robust_accuracy(model, loader, loader2=None, loss_fn=torch.nn.CrossEntropyLoss()):
    #print(f'num of image:{len(loader.dataset)}')
    #model.eval()
    train_acc, train_loss = 0.0, 0.0
    l2_distance = 0.0
    max_distance = 0.0
    device = next(model.parameters()).device
    model_name = str(model).split("(")[0]
    print(model_name)

    if loader2 is None:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            #yp = model(normalize(x))
            yp = model(x)
            loss = loss_fn(yp, y)
            train_acc += (yp.argmax(dim=1) == y).sum().item()
            # loss use mean (not sum), loss manually sum here
            train_loss += loss.item() * x.shape[0]
        return train_acc / len(loader.dataset), train_loss / len(loader.dataset)
    else:
        # loader as adv, loader as origin
        for (x, y), (x2, y2) in zip(loader, loader2):
            x, y = x.to(device), y.to(device)
            x2 = x2.to(device)
            #yp = model(normalize(x))
            yp = model(x)
            loss = loss_fn(yp, y)
            train_acc += (yp.argmax(dim=1) == y).sum().item()
            # loss use mean (not sum), loss manually sum here
            train_loss += loss.item() * x.shape[0]
            # dist use l2 distance
            for i in range(x.shape[0]):
                # dis = torch.dist(x[i], x2[i], p=2)
                l2 = torch.sqrt(torch.sum((torch.square(x[i]-x2[i]))))
                # l2_ = torch.sqrt(torch.sum(((x[i]-x2[i])**2)))
                l2_distance += l2.item()
                if (l2.item() > max_distance):
                    max_distance = l2.item()
        return train_acc / len(loader.dataset), train_loss / len(loader.dataset), l2_distance / len(loader.dataset), max_distance

def save_tensors_to_images(data_dir, images, labels, n_classes, classes=None):
    r"""
        create directory which stores images
        Shape:
        - images: :math:`(N, C, H, W)` where `N = number of images`, `C = number of channels`,
                        `H = height` and `W = width`. It must have a range [0, 1].
    """
    if os.path.exists(data_dir) is not True:
        os.makedirs(data_dir)
    dir_list = [0] * n_classes

    for image, label in zip(images, labels):
        label = label.detach().cpu().numpy()
        image = image.detach().cpu().clamp(0, 1)
        image = image.mul(255).type(torch.uint8).clamp(0, 255)
        image = image.numpy().transpose((1,2,0))

        dir_list[label] = dir_list[label] + 1
        if classes is not None:
            if isinstance(classes[label], tuple) or isinstance(classes[label], list):
                _dir = re.sub(r'(?u)[^-\w.]', '', str(classes[label]).strip().replace(' ', '__'))
            else:
                _dir = f"{classes[label]}"
            if os.path.exists(os.path.join(data_dir, _dir)) is not True:
                os.makedirs(os.path.join(data_dir, _dir))
            name = f"{_dir}/{_dir}_{dir_list[label]}.jpg"
        else:
            _dir = f"{label}"
            if os.path.exists(os.path.join(data_dir, _dir)) is not True:
                os.makedirs(os.path.join(data_dir, _dir))
            name = f"{label}/{dir_list[label]}.jpg"
        im = Image.fromarray(image.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(data_dir, name))

def save_dataloader_to_images(data_dir, dataloader, n_classes, classes=None):
    image_list,label_list = [],[]
    for i, batch in enumerate(dataloader):
        images, labels = batch[0], batch[1]
        image_list.append(images.detach().cpu())
        label_list.append(labels.detach().cpu())
    image_list_cat = torch.cat(image_list, 0)
    label_list_cat = torch.cat(label_list, 0)
    save_tensors_to_images(data_dir, image_list_cat, label_list_cat, n_classes, classes)

def save_dataloader_to_files(data_dir, dataloader):
    if os.path.exists(data_dir) is not True:
        os.makedirs(data_dir)
    idx = 0
    bs = dataloader.batch_size
    for batch in dataloader:
        x, y = batch
        for i in range(bs):
            torch.save([x[i], y[i]], f"{data_dir}/tensor{idx}.pt")
            idx += 1

class FolderDataset(torch.utils.data.Dataset):
   def __init__(self, data_dir):
       self.files = os.listdir(data_dir)
       self.data_dir = data_dir
   def __len__(self):
       return len(self.files)
   def __getitem__(self, idx):
       return torch.load(f"{self.data_dir}/{self.files[idx]}")

class ADVTensorDatasetWithTargetLabel(torch.utils.data.Dataset):
    def __init__(self, adv_images, labels, target_labels, transform=None):
        assert adv_images.shape[0] == labels.shape[0] and adv_images.shape[0] == target_labels.shape[0]
        self.adv_images = adv_images
        self.labels = labels
        self.target_labels = target_labels

    def __getitem__(self, index):
        adv_image = self.adv_images[index]
        label = self.labels[index]
        target_label = self.target_labels[index]

        return adv_image,label,target_label

    def __len__(self):
        return self.adv_images.size(0)

class ADVTensorDataset(torch.utils.data.Dataset):
    def __init__(self, adv_images, labels, transform=None):
        assert adv_images.shape[0] == labels.shape[0]
        self.adv_images = adv_images
        self.labels = labels

    def __getitem__(self, index):
        adv_image = self.adv_images[index]
        label = self.labels[index]

        return adv_image,label

    def __len__(self):
        return self.adv_images.size(0)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.transform = transform
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            class_id = int(class_dir)
            for image_name in os.listdir(class_path):
                image = os.path.join(class_path, image_name)
                self.images.append(image)
                self.labels.append(class_id)
        self.labels = torch.as_tensor([lbl for lbl in self.labels])
        #self.labels = np.array(self.labels)
        #self.labels = torch.from_numpy(self.labels)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __len__(self):
        return len(self.images)

if __name__ == "__main__":
    from imagenet_data import imagenet_normalize, get_imagenet_val_dataloader
    from cifar10_data import cifar10_normalize, get_cifar10_test_dataloader

    dataloader, classes = get_cifar10_test_dataloader('D:/workspace/data/cifar10/', batch_size=4)
    save_dataloader_to_images('D:/workspace/data/adv_data/cifar10_benign', dataloader, len(classes), classes)

    dataloader, classes = get_imagenet_val_dataloader('D:/workspace/data/ImageNet/', batch_size=4)
    save_dataloader_to_images('D:/workspace/data/adv_data/imagenet_benign', dataloader, len(classes), classes)