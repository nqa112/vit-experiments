import torch
import torch.optim as optim
import torch.nn as nn
from datasets import load_dataset, DatasetDict
from calflops import calculate_flops
import wandb

from train import *
from vit import ViT
from dataset import prepare_ds

proj_name = "vit_tinyimagenet"
ds_name = "zh-plus/tiny-imagenet"
n_classes = 200
img_size = 64
n_channels = 3
patch_size = 8
batch_size = 32
nprocs = 4
lr = 1e-3
n_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hidden_dim = 192
mlp_dim = 384
n_layers = 9
n_heads = 12
head_dim = hidden_dim // n_heads
# head_dim = 64
dropout = 0.
emb_dropout = 0.

# Prepare dataset
ds_train_valid = load_dataset(ds_name)["train"].train_test_split(test_size=0.2, stratify_by_column="label")
ds_test = load_dataset(ds_name)["valid"]

dataset = DatasetDict({
    "train": ds_train_valid["train"],
    "val": ds_train_valid["test"],
    "test": ds_test
})

train_loader, val_loader, test_loader = prepare_ds(dataset=dataset, 
                                                   img_size=img_size,
                                                   batch_size=batch_size, 
                                                   nprocs=nprocs)

# Define model
model = ViT(image_size=img_size, 
            patch_size=patch_size, 
            num_classes=n_classes,
            dim=hidden_dim,
            depth=n_layers,
            heads=n_heads,
            dim_head=head_dim,
            mlp_dim=mlp_dim,
            channels=n_channels,
            dropout=dropout,
            emb_dropout=emb_dropout)

model = model.to(device)
# import pdb; pdb.set_trace()

flops, _, params = calculate_flops(model=model,
                                   input_shape=(batch_size, n_channels, img_size, img_size),
                                   print_results=False)
print(f"Model information: FLOPS: {flops} | Params: {params}")

# Train model
criterion = nn.CrossEntropyLoss()
# opt = optim.Adam(model.parameters(), lr=lr)
opt = optim.AdamW(model.parameters(), lr=lr)

# scheduler = WarmupCosineSchedule(optimizer=opt,
#                                  warmup_steps=n_epochs*len(train_loader)*0.05,
#                                  t_total=n_epochs*len(train_loader))

scheduler = CosineAnnealingWarmupRestarts(optimizer=opt,
                                          first_cycle_steps=n_epochs*len(train_loader),
                                          cycle_mult=1.,
                                          max_lr=lr,
                                          min_lr=1e-6,
                                          warmup_steps=n_epochs*len(train_loader)*0.1,
                                          gamma=1.)

# Init wandb logging
run = wandb.init(project=proj_name)
assert run is wandb.run

for epoch in range(n_epochs):

    train_acc, train_loss = train_step(train_dataloader=train_loader,
                                       model=model,
                                       criterion=criterion,
                                       optimizer=opt,
                                       scheduler=scheduler,
                                       device=device)

    val_acc, val_loss = eval_step(val_dataloader=val_loader,
                                  model=model,
                                  criterion=criterion,
                                  device=device)
        
    run.log({"train_acc": train_acc,
             "train_loss": train_loss,
             "val_acc": val_acc,
             "val_loss": val_loss,
             "lr": scheduler.get_lr()[0]})

    print(f"[Epoch {epoch+1}/{n_epochs}] train_acc = {train_acc: .4f} | train_loss = {train_loss:.4f} | val_acc = {val_acc: .4f} | val_loss = {val_loss:.4f}\n")

run.finish()
print("Finish training")