import os
from torchvision import transforms,datasets
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloader(
        train_dir:str,
        test_dir:str,
        batch_size:int = 32,
        transform:transforms.Compose  = None,
        num_workers:int=NUM_WORKERS,
):
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    test_data = datasets.ImageFolder(test_dir, transform=transform)

    class_names = train_data.classes

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader,class_names