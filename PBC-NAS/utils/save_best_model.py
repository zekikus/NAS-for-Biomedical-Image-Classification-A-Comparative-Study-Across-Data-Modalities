import torch

class BestModelCheckPoint:

    def __init__(self, model_name, path=".", ablation=None):
        self.path = path
        self.best_score = 0
        self.model_name = model_name
        self.ablation = ablation
    
    def check(self, score, model, seed):
        if score > self.best_score:
            print("Best Score:", score)
            self.best_score = score
            torch.save(model.state_dict(), f"{self.path}/model_{self.model_name}_seed_{seed}{'_ablation_' + self.ablation if self.ablation is not None else ''}.pt")

        