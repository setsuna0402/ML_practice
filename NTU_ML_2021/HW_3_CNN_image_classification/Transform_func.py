import torchvision.transforms as transforms

train_transform = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((256, 256)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    
    # transforms.RandomChoice(
    #     [transforms.AutoAugment(),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
    #     transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
    # ),
    transforms.RandomResizedCrop((224, 224), scale=(0.5, 1)),
    # transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ColorJitter(0.2, 0.2),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize the image with mean and std of ImageNet dataset.
])