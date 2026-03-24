import torchvision.transforms as transforms

train_transform = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((142, 142)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    
    # transforms.RandomChoice(
    #     [transforms.AutoAugment(),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
    # ),
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop((128, 128), scale=(0.3, 1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_transform = transforms.Compose([
    transforms.Resize((142, 142)),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])