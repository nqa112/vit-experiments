import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from augmentation import *

def prepare_ds(dataset, img_size, batch_size, nprocs):
    # Tensor transformation for train and test
    train_transforms_group = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(img_size, padding=4),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2770, 0.2691, 0.2821])
    ])

    val_transforms_group = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                             std=[0.2770, 0.2691, 0.2821])
        ])

    def transform_dataset(examples, transforms_group):
        examples["img_tensor"] = []

        for img in examples["image"]:
            transformed_img = transforms_group(img)
            examples["img_tensor"].append(transformed_img)

        return examples

    # Filter out gray images
    ds_train = dataset["train"].filter(lambda example: example["image"].mode == "RGB")
    ds_val = dataset["val"].filter(lambda example: example["image"].mode == "RGB")
    ds_test = dataset["test"].filter(lambda example: example["image"].mode == "RGB")

    ds_train = ds_train.map(transform_dataset, 
                            batched=True, 
                            batch_size=batch_size, 
                            num_proc=nprocs,
                            fn_kwargs={"transforms_group": train_transforms_group})
    ds_val = ds_val.map(transform_dataset, 
                        batched=True, 
                        batch_size=batch_size, 
                        num_proc=nprocs,
                        fn_kwargs={"transforms_group": val_transforms_group})
    ds_test = ds_test.map(transform_dataset, 
                          batched=True, 
                          batch_size=batch_size, 
                          num_proc=nprocs,
                          fn_kwargs={"transforms_group": val_transforms_group})

    ds_train = ds_train.with_format("torch",
                                    columns=["label", "img_tensor"],
                                    dtype = torch.float32)
    ds_val = ds_val.with_format("torch",
                                columns=["label", "img_tensor"],
                                dtype = torch.float32)
    ds_test = ds_test.with_format("torch",
                                  columns=["label", "img_tensor"],
                                  dtype = torch.float32)
    
    import pdb; pdb.set_trace()

    train_dataloader = DataLoader(dataset=ds_train,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=nprocs,
                                drop_last=True)
    val_dataloader = DataLoader(dataset=ds_val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=nprocs,
                                drop_last=False)
    test_dataloader = DataLoader(dataset=ds_test,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=nprocs,
                                drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader