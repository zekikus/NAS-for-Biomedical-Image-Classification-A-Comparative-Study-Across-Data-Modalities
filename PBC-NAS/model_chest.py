from cmath import nan
import copy
import torch
import timeit
import torch.nn as nn
import numpy as np
from cell import Cell
import torch.optim as optim
from ops import OPS as ops_dict
from utils.early_stopping import EarlyStopping

class Model(nn.Module):

    def __init__(self, chromosome = None, config = None, nbr_cell = None, nbr_stack = None, nbr_filters = None, num_classes = None, C_in = 1):
        super(Model, self).__init__()
        # CONSTANT
        self.NUM_VERTICES = 7
        self.MAX_EDGE_NBR = int(((self.NUM_VERTICES) * (self.NUM_VERTICES - 1)) / 2)
        
        self.solNo = None
        self.C_in = C_in
        self.fitness = 0
        self.config = config
        self.nbr_stack = nbr_stack
        self.nbr_cell = nbr_cell
        self.nbr_init_filters = nbr_filters
        self.chromosome = chromosome

        self.org_matrix = self.create_matrix()
        self.org_ops = list(self.config[self.MAX_EDGE_NBR: self.MAX_EDGE_NBR + self.NUM_VERTICES - 2])        

        self.stem_conv = nn.Sequential(
            nn.Conv2d(C_in, self.nbr_init_filters, kernel_size=3, padding='same'),
            nn.BatchNorm2d(self.nbr_init_filters),
            nn.ReLU(),
        )

        embedding_size = self.nbr_init_filters
        for i in range(self.nbr_stack - 1):
            embedding_size *= 2
        
        self.classifier = torch.nn.Linear(
            embedding_size, # embedding_size
            num_classes, # class
            bias=True)
        
        self.mp = nn.MaxPool2d((2, 2), stride=2)
        self.cells = nn.ModuleList([self.stem_conv])
        
        self.matrix, self.ops, self.isFeasible = self.prune(self.org_matrix, self.org_ops)
        if self.isFeasible:
            # Get ops except for Input and output nodes
            self.ops = self.ops[1:-1]
            self.compile()
        
            for param in self.parameters():
                param.requires_grad = True
                if len(param.shape) > 1:
                    torch.nn.init.xavier_uniform_(param)
    
    def create_matrix(self):
        """ Convert encoding vector to adjacency matrix """
        triu_indices = np.triu_indices(self.NUM_VERTICES, k = 1)
        self.matrix = np.zeros((self.NUM_VERTICES, self.NUM_VERTICES))
        self.matrix[triu_indices] = self.config[0: self.MAX_EDGE_NBR]
        
        return self.matrix

    def compile(self):
        """ Build Model """
        C_in = self.C_in # Input channel
        C_out = self.nbr_init_filters

        # Stack
        for stack_idx in range(self.nbr_stack):
            # Cell - num cell per stack
            for cell_idx in range(self.nbr_cell):
                cell = Cell(self.matrix, self.ops, C_in, C_out)
                C_in = C_out
                self.cells.append(cell)
            C_out = C_out * 2


    def forward(self, inputs):
        outputs = [0] * (self.nbr_stack) # Stack outputs
        outputs[0] = self.cells[0](inputs) # Apply Stem Convolution

        # Encoder Cells
        _input = outputs[0]
        
        for stack_idx in range(self.nbr_stack): # Stack 
            for cell_idx in range(self.nbr_cell): # Cell - num cell per stack
                cell_output = self.cells[(stack_idx * self.nbr_cell) + cell_idx + 1](_input, stack_idx) # Feed forward input to Cell
                _input = cell_output
            
            if stack_idx == self.nbr_stack - 1:
                outputs[stack_idx] = cell_output
            else:
                outputs[stack_idx] = self.mp(cell_output) # Apply Max Pooling - Downsampling stage
            _input = outputs[stack_idx]

        return self.classifier(torch.mean(outputs[-1], dim=[2, 3])) # Linear -> Global Average Pooling


    def evaluate(self, train_loader, val_loader, loss_fn, metric_fn, device):
        
        try:
            print(f"Model {self.solNo} Training...")
            self.to(device) # cuda start

            train_loss = []
            train_acc = []
            log = f"Model No: {self.solNo}\n"
            early_stopping = EarlyStopping(patience=30)

            startTime = timeit.default_timer()
            optimizer = optim.Adam(self.parameters(), lr=0.001)
            for epoch in range(120):

                # Train Phase
                self.train()
                for inputs, labels in train_loader:
                    #labels = labels.type(torch.LongTensor)
                    inputs, labels = inputs.to(device), labels.to(device)
        
                    with torch.set_grad_enabled(True):
                        output = self.forward(inputs)
                        error = loss_fn(output.float(), labels.float())
                        train_loss.append(error.item())
                        train_acc.append(metric_fn(output.to('cpu'), labels.to('cpu')).item())
                        optimizer.zero_grad()
                        error.backward()
                        optimizer.step()
                
                torch.cuda.empty_cache()
		
                # Validation Phase
                val_loss = []
                val_dice = []
                self.eval()
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        #labels = labels.type(torch.LongTensor)
                        inputs, labels = inputs.to(device), labels.to(device)
                        output = self.forward(inputs)
                        error = loss_fn(output.float(), labels.float())
                        val_dice.append(metric_fn(output.to('cpu'), labels.to('cpu')).item())
                        val_loss.append(error)
                
                torch.cuda.empty_cache()
                
                # Log
                avg_tr_loss = sum(train_loss) / len(train_loss)
                avg_tr_score = sum(train_acc) / len(train_acc)
                avg_val_loss = sum(val_loss) / len(val_loss)
                avg_val_score = sum(val_dice) / len(val_dice)
                txt = f"\nEpoch: {epoch}, tr_loss: {avg_tr_loss}, tr_acc: {avg_tr_score}, val_loss: {avg_val_loss}, val_acc: {avg_val_score}"
                log += txt
                print(txt)

                # Early Stopping Check
                if early_stopping.stopTraining(epoch, avg_val_loss, avg_val_score):
                    self.fitness = early_stopping.best_score
                    self.cost = timeit.default_timer() - startTime
                    print(f"Stop Training - Model {self.solNo} , {self.fitness}, {self.cost}")
                    break
            
        except Exception as e: # Memory Problems
            torch.cuda.empty_cache()
            print(e)
            return -1, -1, None

        torch.cuda.empty_cache()

        self.fitness = early_stopping.best_score
        self.cost = timeit.default_timer() - startTime
        
        log += f"\nElapsed Time: {self.cost}, Fitness: {self.fitness}"
        
        return self.fitness, self.cost, log

    def prune(self, original_matrix, original_ops):

        """Prune the extraneous parts of the graph.

        General procedure:
        1) Remove parts of graph not connected to input.
        2) Remove parts of graph not connected to output.
        3) Reorder the vertices so that they are consecutive after steps 1 and 2.

        These 3 steps can be combined by deleting the rows and columns of the
        vertices that are not reachable from both the input and output (in reverse).
        """
        matrix = copy.deepcopy(original_matrix)
        ops = copy.deepcopy(original_ops)
        num_vertices = np.shape(original_matrix)[0]

        ops.insert(0, 'input')
        ops.append('output')

        # DFS forward from input
        visited_from_input = set([0])
        frontier = [0]
        while frontier:
            top = frontier.pop()
            for v in range(top + 1, num_vertices):
                if original_matrix[top, v] and v not in visited_from_input:
                    visited_from_input.add(v)
                    frontier.append(v)

        # DFS backward from output
        visited_from_output = set([num_vertices - 1])
        frontier = [num_vertices - 1]
        while frontier:
            top = frontier.pop()
            for v in range(0, top):
                if original_matrix[v, top] and v not in visited_from_output:
                    visited_from_output.add(v)
                    frontier.append(v)

        # Any vertex that isn't connected to both input and output is extraneous to
        # the computation graph.
        extraneous = set(range(num_vertices)).difference(
            visited_from_input.intersection(visited_from_output))

        # If the non-extraneous graph is less than 2 vertices, the input is not
        # connected to the output and the spec is invalid.
        if len(extraneous) > num_vertices - 2:
            return matrix, ops, False

        matrix = np.delete(matrix, list(extraneous), axis=0)
        matrix = np.delete(matrix, list(extraneous), axis=1)
        for index in sorted(extraneous, reverse=True):
            del ops[index]   

        if len(ops) < 3:
            return matrix, ops, False

        if np.sum(matrix) > 9:
            return matrix, ops, False

        """
        # Pooling Operasyonlarının fazla olması infeasible model üretiyor
        nbr_pooling = np.array(ops[1:-1])[np.where(matrix[1:-1, -1] == 1)]
        if len(nbr_pooling) > 0:
            cond_1 = (list(nbr_pooling).count(2) / len(nbr_pooling)) > 0.5 # Max Pooling Operations (going edge to output node)
            cond_2 = ops.count(2) / len(ops) > 0.5 # Number of Max Pooling Operations

            if cond_1 or cond_2:
                return matrix, ops, False
        """

        return matrix, ops, True

    def reset(self):
        for param in self.parameters():
            param.requires_grad = True
            if len(param.shape) > 1:
                torch.nn.init.xavier_uniform_(param)        
