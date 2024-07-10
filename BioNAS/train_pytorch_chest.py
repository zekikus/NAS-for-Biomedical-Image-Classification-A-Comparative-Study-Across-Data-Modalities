import io
import os
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import medmnist
from medmnist import INFO
from torchinfo import summary
import matplotlib.pyplot as plt
import torch.utils.data as data
import torchvision.transforms as transforms
from torchmetrics import Accuracy
from utils.save_best_model import BestModelCheckPoint



import warnings
warnings.filterwarnings("ignore")

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cuda')
        else:
            return super().find_class(module, name)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


modelNo = 78
path = "results"
download = True
data_flag = 'chestmnist'

info = INFO[data_flag]
NUM_CLASSES = len(info['label'])
BATCH_SIZE = 128
info = INFO[data_flag]

DataClass = getattr(medmnist, info['python_class'])

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
val_dataset = DataClass(split='val', transform=data_transform, download=download)

# encapsulate data into dataloader form
train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device('cuda:0')

print(train_dataset.__len__())

for seed in [0, 42, 143, 1234, 3074]:
    log = ""
    seed_torch(seed)

    checkpoint = BestModelCheckPoint(modelNo, path=f"{path}/rs_{data_flag}")

    # Loss Function
    loss_fn = torch.nn.BCEWithLogitsLoss()
    metric_fn = Accuracy(task='multilabel', num_labels=NUM_CLASSES)

    # Load Model
    model = None
    with open(f"{path}/rs_{data_flag}/model_{modelNo}.pkl", "rb") as f:
        model = GPU_Unpickler(f).load()

    print("\nModel No:", model.solNo, "Seed:", seed)
    #summary(model, input_size=(1, 3, 28, 28))

    model.reset()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # 1e-3


    for epoch in range(200):
        train_loss = []
        train_acc = []

        # Train Phase
        model.train()
        for inputs, labels in train_loader:
            #labels = labels.type(torch.LongTensor)
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.set_grad_enabled(True):
                output = model(inputs)
                error = loss_fn(output.float(), labels.float())
                train_loss.append(error.item())
                train_acc.append(metric_fn(output.to('cpu'), labels.to('cpu')).item())
                optimizer.zero_grad()
                error.backward()
                optimizer.step()
                del output
                del error
            
            del inputs
            del labels

        torch.cuda.empty_cache()

        # Validation Phase
        val_loss = []
        val_acc = []
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                #labels = labels.type(torch.LongTensor)
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                error = loss_fn(output, labels.float())
                val_acc.append(metric_fn(output.to('cpu'), labels.to('cpu')).item())
                val_loss.append(error)

        avg_tr_loss = sum(train_loss) / len(train_loss)
        avg_tr_score = sum(train_acc) / len(train_acc)
        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val_score = sum(val_acc) / len(val_acc)
        txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_acc_score: {avg_tr_score}, val_loss: {avg_val_loss}, val_acc: {avg_val_score}"
        log += txt
        print(txt)
        checkpoint.check(avg_val_score, model, seed)
        torch.cuda.empty_cache()

    del checkpoint

    # Write Log
    with open(f"{path}/rs_{data_flag}/log_{modelNo}_seed_{seed}.txt", "w") as f:
        f.write(log)
