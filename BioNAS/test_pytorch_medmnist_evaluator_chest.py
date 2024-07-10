import io
import os
import time
import torch
import pickle
import random
import medmnist
import numpy as np
import pandas as pd
from medmnist import INFO, Evaluator
import torch.utils.data as data
from torchmetrics import Accuracy
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")

def write_file(fname, data):
    with open(fname, "w") as f:
        f.write(data)

class GPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location=device)
        else:
            return super().find_class(module, name)

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    modelNo = 78

    path = "results"
    pickle_path = "rs_chestmnist"
    device = torch.device('cuda:0')

    data_flag = pickle_path.replace("rs_", "")
    download = True

    BATCH_SIZE = 1
    info = INFO[data_flag]
    NUM_CLASSES = len(info['label'])
    
    print("#Classes:", NUM_CLASSES)

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    test_dataset = DataClass(split='test', transform=data_transform, download=download)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(test_dataset.__len__())

    metric_fn = Accuracy(task='multilabel', num_labels=NUM_CLASSES)

    txt = ""
    for seed in [0, 42, 143, 1234, 3074]:
        start_time = time.time()
        print("Seed:", seed)

        # Load Model
        model = None
        with open(f"{path}/{pickle_path}/model_{modelNo}.pkl", "rb") as f:
            model = GPU_Unpickler(f).load()

        # Load pre-trained weights
        print("Model No:", model.solNo, "Seed:", seed)
        print("Load Model...")
        model.load_state_dict(torch.load(f"{path}/{pickle_path}/model_{modelNo}_seed_{seed}.pt", map_location=device))
        model.to(device)

        model.eval()

        y_score = torch.tensor([]).to(device)
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                targets = targets.to(torch.float32)
                outputs = outputs.softmax(dim=-1)
                y_score = torch.cat((y_score, outputs), 0)

            y_score = y_score.detach().cpu().numpy()
            
            evaluator = Evaluator(data_flag, 'test')
            metrics = evaluator.evaluate(y_score)
        
            print('%s  auc: %.4f  acc:%.4f' % ('test', *metrics))
            txt += '%s  auc: %.4f  acc:%.4f\n' % ('test', *metrics)
        
    write_file(f"{path}/{pickle_path}/model_{modelNo}.txt", txt)
