import copy
import numpy as np
import torch.nn as nn
from ops import OPS, OPS_Keys

# Encoder operations
CELL_OPS = copy.deepcopy(OPS_Keys)

class Cell(nn.Module):
    def __init__(self, matrix, ops, prev_C, currrent_C):
        super(Cell, self).__init__()
        
        self.ops = ops # Discrete values
        self.matrix = matrix
        self.prev_C = prev_C
        self.current_C = currrent_C # Number of filters

        self.NBR_OP = self.matrix.shape[0] - 1
        self.stem_conv = nn.Conv2d(self.prev_C, self.current_C, kernel_size=1, padding='same')
        self.compile()

    def compile(self):
        self.ops_list = nn.ModuleList([self.stem_conv])        

        # Iterate each operation
        for op_idx in range(1, self.NBR_OP):
            op = CELL_OPS[self.ops[op_idx - 1]]
            self.ops_list.append(OPS[op](self.current_C, self.current_C))

    def forward(self, inputs, stack_id):

        outputs = [0] * len(self.ops_list) # Store output of each operation
        
        if stack_id == 0:
            outputs[0] = inputs
        else:
            outputs[0] = self.ops_list[0](inputs) # Stem Convolution - Equalize channel count

        # Feed forward - Input to output
        for op_idx in range(1, self.NBR_OP):
            op = self.ops_list[op_idx]
            # Get input nodes/edges to the operation
            in_nodes = list(np.where(self.matrix[:, op_idx] == 1)[0])
            
            # Sum and process if there is more than one input node/edge
            if len(in_nodes) > 1:
                _input = sum([outputs[i] for i in in_nodes])
                outputs[op_idx] = op(_input)
            else:
                outputs[op_idx] = op(outputs[in_nodes[0]])
        
        # Get input nodes/edges to the output node
        in_nodes = list(np.where(self.matrix[:, self.NBR_OP] == 1)[0])
        return outputs[0] + sum([outputs[out] for out in in_nodes])# Output




        
            
