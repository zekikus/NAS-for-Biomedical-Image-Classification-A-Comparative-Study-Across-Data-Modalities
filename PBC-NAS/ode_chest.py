import os
import copy
import torch
import random
import pickle
import medmnist
import numpy as np
from model_chest import Model
from ops import OPS_Keys
from medmnist import INFO
from utils.distances import *
import torch.utils.data as data
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

"""
    - Opposition-Based Differential Evolution
"""

class DE():
    
    def __init__(self, pop_size = None, 
                 mutation_factor = None, 
                 crossover_prob = None, 
                 boundary_fix_type = 'random', 
                 seed = None,
                 mutation_strategy = 'rand1',
                 crossover_strategy = 'bin'):

        # DE related variables
        self.pop_size = pop_size
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.mutation_strategy = mutation_strategy
        self.crossover_strategy = crossover_strategy
        self.boundary_fix_type = boundary_fix_type

        # Global trackers
        self.population = []
        self.P0 = [] # P0 population
        self.OP0 = [] # Opposite of the P0 population
        self.history = []
        self.allModels = dict()
        self.best_arch = None
        self.seed = seed

        # CONSTANTS
        self.MAX_SOL = 500
        self.NUM_EDGES = 9
        self.NUM_VERTICES = 7
        self.DIMENSIONS = 29
        self.MAX_STACK = 3
        self.MAX_NUM_CELL = 3
        self.JUMPING_RATE = 0.3
        self.STACKS = [i for i in range(1, self.MAX_STACK + 1)] # 1, 2, 3
        self.CELLS = [i for i in range(1, self.MAX_NUM_CELL + 1)] # 1, 2, 3
        self.NBR_FILTERS = [2**i for i in range(5, 8)] # 32, 64, 128
        #self.NBR_FILTERS = [2**i for i in range(3, 6)] # 8, 16, 32
        self.OPS = copy.deepcopy(OPS_Keys)
    
    def reset(self):
        self.best_arch = None
        self.population = []
        self.P0 = []
        self.OP0 = []
        self.allModels = dict()
        self.history = []
        self.init_rnd_nbr_generators()
    
    def init_rnd_nbr_generators(self):
        # Random Number Generators
        self.crossover_rnd = np.random.RandomState(self.seed)
        self.sample_pop_rnd = np.random.RandomState(self.seed)
        self.init_pop_rnd = np.random.RandomState(self.seed)
        self.jumping_rnd = np.random.RandomState(self.seed)

    def seed_torch(self, seed=42):
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
    
    def writePickle(self, data, name):
        # Write History
        with open(f"results/model_{name}.pkl", "wb") as pkl:
            pickle.dump(data, pkl)

    # Generate uniformly distributed random population P_0
    def init_P0_population(self, pop_size = None):
        i = 0
        while i < pop_size:
            chromosome = self.init_pop_rnd.uniform(low=0.0, high=1.0, size=self.DIMENSIONS)
            config = self.vector_to_config(chromosome)
            model = Model(chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES)

            # Same Solution Check
            isSame, _ = self.checkSolution(model)
            if not isSame:
                model.solNo = self.solNo
                self.solNo += 1
                self.allModels[model.solNo] = {"org_matrix": model.org_matrix.astype("int8"), 
                                               "org_ops": model.org_ops,
                                               "chromosome": model.chromosome,
                                               "fitness": model.fitness}                                               
                self.P0.append(model)
                self.writePickle(model, model.solNo)
                i += 1
    
    def get_opposite_model(self, model, a = 0, b = 1):

        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            opposite_chromosome = np.array([a[idx] + b[idx] - c for idx, c in enumerate(model.chromosome)])
        else:
            opposite_chromosome = np.array([a + b - c for c in model.chromosome])
        
        config = self.vector_to_config(opposite_chromosome)
        opposite_model = Model(opposite_chromosome, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES)
        
        return opposite_model

    def init_OP0_population(self):
        counter = 0
        while counter < len(self.P0):
            opposite_model = self.get_opposite_model(self.P0[counter])
            # Same Solution Check
            isSame, _ = self.checkSolution(opposite_model)
            if not isSame:
                self.solNo += 1
                opposite_model.solNo = self.solNo
                self.allModels[opposite_model.solNo] = {"org_matrix": opposite_model.org_matrix.astype("int8"), 
                                                        "org_ops": opposite_model.org_ops,
                                                        "chromosome": opposite_model.chromosome,
                                                        "fitness": opposite_model.fitness}
                self.OP0.append(opposite_model)
                self.writePickle(opposite_model, opposite_model.solNo)
            counter += 1

    def checkSolution(self, model):
        model_dict = {"org_matrix": model.org_matrix.astype("int8"), 
                      "org_ops": model.org_ops}
        for i in self.allModels.keys():
            model_2 = self.allModels[i]
            D = jackard_distance_caz(model_dict, model_2)
            if D == 0:
                return True, model_2
        
        return False, None          
    
    def sample_population(self, size = None):
        '''Samples 'size' individuals'''

        selection = self.sample_pop_rnd.choice(np.arange(len(self.population)), size, replace=False)
        return self.population[selection]
    
    def boundary_check(self, vector):
        '''
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        projection == The invalid value is truncated to the nearest limit
        random == The invalid value is repaired by computing a random number between its established limits
        reflection == The invalid value by computing the scaled difference of the exceeded bound multiplied by two minus

        '''
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        
        if self.boundary_fix_type == 'projection':
            vector = np.clip(vector, 0.0, 1.0)
        elif self.boundary_fix_type == 'random':
            vector[violations] = np.random.uniform(low=0.0, high=1.0, size=len(violations))
        elif self.boundary_fix_type == 'reflection':
            vector[violations] = [0 - v if v < 0 else 2 - v if v > 1 else v for v in vector[violations]]

        return vector

    def get_param_value(self, value, step_size):
        ranges = np.arange(start=0, stop=1, step=1/step_size)
        return np.where((value < ranges) == False)[0][-1]

    def vector_to_config(self, vector):
        '''Converts numpy array to discrete values'''

        try:
            config = np.zeros(self.DIMENSIONS, dtype='uint8')
            
            max_edges = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)
            # Edges
            for idx in range(max_edges):
                config[idx] = self.get_param_value(vector[idx], 2)

            # Vertices - Ops
            for idx in range(max_edges, max_edges + self.NUM_VERTICES - 2):
                config[idx] = self.get_param_value(vector[idx], len(self.OPS))

            # Number of Cells
            idx = max_edges + self.NUM_VERTICES - 2
            config[idx] = self.get_param_value(vector[idx], len(self.CELLS))
            
            # Number of Stacks
            config[idx + 1] = self.get_param_value(vector[idx + 1], len(self.STACKS))

            # Number of Filters
            config[idx + 2] = self.get_param_value(vector[idx + 2], len(self.NBR_FILTERS))
        except:
            print("HATA...", vector)

        return config

    def f_objective(self, model):
        if model.isFeasible == False: # Feasibility Check
            return -1, -1
        
        # Else  
        fitness, cost, log = model.evaluate(train_loader, val_loader, loss_fn, metric_fn, device)
        if fitness != -1:
            self.totalTrainedModel += 1
            self.allModels.setdefault(model.solNo, dict())
            self.allModels[model.solNo]["fitness"] = fitness
            with open(f"results/model_{model.solNo}.txt", "w") as f:
                f.write(log)
        return fitness, cost

    def init_eval_pop(self):
        '''
            Creates new population of 'pop_size' and evaluates individuals.
        '''
        print("Start Initialization...")

        self.init_P0_population(self.pop_size)
        self.init_OP0_population()

        for model in self.P0:
            model.fitness, cost = self.f_objective(model)
            self.writePickle(model, model.solNo)
        
        for model in self.OP0:
            model.fitness, cost = self.f_objective(model)
            self.writePickle(model, model.solNo)
        
        self.P0.extend(self.OP0)
        self.population = sorted(self.P0, key = lambda x: x.fitness, reverse=True)[:self.pop_size]
        self.best_arch = self.population[0]

        del self.P0
        del self.OP0
        
        return np.array(self.population)

    def mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation(self, current=None, best=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_rand1(r1.chromosome, r2.chromosome, r3.chromosome)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self.sample_population(size=5)
            mutant = self.mutation_rand2(r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome, r5.chromosome)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_rand1(best, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self.sample_population(size=4)
            mutant = self.mutation_rand2(best, r1.chromosome, r2.chromosome, r3.chromosome, r4.chromosome)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self.sample_population(size=2)
            mutant = self.mutation_currenttobest1(current, best.chromosome, r1.chromosome, r2.chromosome)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self.sample_population(size=3)
            mutant = self.mutation_currenttobest1(r1.chromosome, best.chromosome, r2.chromosome, r3.chromosome)

        return mutant

    def crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.crossover_rnd.rand(self.DIMENSIONS) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.crossover_rnd.randint(0, self.DIMENSIONS)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        '''
            Performs the exponential crossover of DE
        '''
        n = self.crossover_rnd.randint(0, self.DIMENSIONS)
        L = 0
        while ((self.crossover_rnd.rand() < self.crossover_prob) and L < self.DIMENSIONS):
            idx = (n+L) % self.DIMENSIONS
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''
            Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self.crossover_exp(target, mutant)
        return offspring
    
    def readPickleFile(self, file):
        with open(f"results/model_{file}.pkl", "rb") as f:
            data = pickle.load(f)
        
        return data

    def evolve_generation(self):
        '''
            Performs a complete DE evolution: mutation -> crossover -> selection
        '''
        trials = []
        Pnext = [] # Next population

        generationBest = max(self.population, key=lambda x: x.fitness)

        # mutation -> crossover
        for j in range(self.pop_size):
            target = self.population[j].chromosome
            mutant = copy.deepcopy(target)
            mutant = self.mutation(current=target, best=generationBest)
            trial = self.crossover(target, mutant)
            trial = self.boundary_check(trial)
            config = self.vector_to_config(trial)
            model = Model(trial, config, self.CELLS[config[-3]], self.STACKS[config[-2]], self.NBR_FILTERS[config[-1]], NUM_CLASSES)
            self.solNo += 1
            model.solNo = self.solNo
            trials.append(model)
        
        trials = np.array(trials)

        # selection
        for j in range(self.pop_size):
            target = self.population[j]
            mutant = trials[j]

            isSameSolution, sol = self.checkSolution(mutant)
            if isSameSolution:
                print("SAME SOLUTION")
                cfg = self.vector_to_config(sol["chromosome"])
                mutant = Model(sol["chromosome"], cfg, self.CELLS[cfg[-3]], self.STACKS[cfg[-2]], self.NBR_FILTERS[cfg[-1]], NUM_CLASSES)
                mutant.fitness = sol["fitness"]
            else:
                self.f_objective(mutant)
                self.writePickle(mutant, mutant.solNo)
                self.allModels[mutant.solNo] = {"org_matrix": mutant.org_matrix.astype("int8"), 
                                                "org_ops": mutant.org_ops,
                                                "chromosome": mutant.chromosome,
                                                "fitness": mutant.fitness}

            # Check Termination Condition
            if self.totalTrainedModel > self.MAX_SOL: 
                return
            #######

            if mutant.fitness >= target.fitness:
                Pnext.append(mutant)
                del target

                # Best Solution Check
                if mutant.fitness >= self.best_arch.fitness:
                    self.best_arch = mutant
            else:
                Pnext.append(target)
                del mutant

        self.population = Pnext

        ## Opposition-Based Generation Jumping
        if self.jumping_rnd.uniform() < self.JUMPING_RATE:
            chromosomes = []
            for model in self.population:
                chromosomes.append(model.chromosome)
            
            min_p_j = np.min(chromosomes, 0)
            max_p_j = np.max(chromosomes, 0)

            counter = 0
            while counter < self.pop_size:
                opposite_model = self.get_opposite_model(self.population[counter], a = min_p_j, b = max_p_j)
                # Same Solution Check
                isSame, _ = self.checkSolution(opposite_model)
                if not isSame:
                    self.solNo += 1
                    opposite_model.solNo = self.solNo
                    self.f_objective(opposite_model)
                    self.allModels[opposite_model.solNo] = {"org_matrix": opposite_model.org_matrix.astype("int8"), 
                                                            "org_ops": opposite_model.org_ops,
                                                            "chromosome": opposite_model.chromosome,
                                                            "fitness": opposite_model.fitness}
                    self.population.append(opposite_model)
                    self.writePickle(opposite_model, opposite_model.solNo)
                counter += 1
            
            self.population = sorted(self.population, key = lambda x: x.fitness, reverse=True)[:self.pop_size]
            if self.population[0].fitness >= self.best_arch.fitness:
                self.best_arch = self.population[0]

        self.population = np.array(self.population)

    def run(self, seed):
        self.seed = seed
        self.solNo = 0
        self.generation = 0
        self.totalTrainedModel = 0
        print(self.mutation_strategy)
        self.reset()
        self.seed_torch()
        self.population = self.init_eval_pop()

        while self.totalTrainedModel < self.MAX_SOL:
            self.evolve_generation()
            print(f"Generation:{self.generation}, Best: {self.best_arch.fitness}, {self.best_arch.solNo}")
            self.generation += 1     
        

if __name__ == "__main__":
    device = torch.device('cuda:1')

    random.seed(42)
    data_flag = 'chestmnist'
    download = True
    BATCH_SIZE = 128
    SPLIT_VALUE = 13
    info = INFO[data_flag]
    task = info['task']
    NUM_CLASSES = len(info['label'])
    
    print("#Classes:", NUM_CLASSES)

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the half of the train data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    rnd_train_indexes = random.sample(range(train_dataset.__len__()), int(train_dataset.__len__() / SPLIT_VALUE))
    train_dataset_ode = torch.utils.data.Subset(train_dataset, rnd_train_indexes)
    
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    rnd_val_indexes = random.sample(range(val_dataset.__len__()), int(val_dataset.__len__() / SPLIT_VALUE))
    val_dataset_ode = torch.utils.data.Subset(val_dataset, rnd_val_indexes)

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset_ode, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = data.DataLoader(dataset=val_dataset_ode, batch_size=BATCH_SIZE, shuffle=False)

    if task == "multi-label, binary-class":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        task_type = "multilabel"
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        task_type = "multiclass"
    
    print(task_type)
    #metric_fn = Evaluator(data_flag, "train")
    metric_fn = Accuracy(task=task_type, num_labels=NUM_CLASSES)

    de = DE(pop_size=20, mutation_factor=0.5, crossover_prob=0.5, seed=42)
    de.run(42)