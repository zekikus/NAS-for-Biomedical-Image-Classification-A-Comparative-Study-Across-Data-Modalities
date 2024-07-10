import copy
import numpy as np
import torch.nn as nn
from ops import OPS, OPS_Keys

# Encoder operations
CELL_OPS = copy.deepcopy(OPS_Keys)

class Cell(nn.Module):
    def __init__(self, matrix, ops, prev_C, currrent_C):
        super(Cell, self).__init__()
        
        self.ops = ops
        self.matrix = matrix
        self.prev_C = prev_C
        self.current_C = currrent_C # Number of filters

        self.NBR_HIDDEN_NODE = self.matrix.shape[0] - 1
        self.stem_conv = nn.Conv2d(self.prev_C, self.current_C, kernel_size=1, padding='same')
        self.compile()

    def compile(self):
        self.ops_list = nn.ModuleList([self.stem_conv])        

        # Iterate each operation
        #for hidden_node_idx in range(self.NBR_HIDDEN_NODE + 1):
        for op_idx in np.transpose(self.ops)[np.transpose(self.ops).nonzero()]:
            op = CELL_OPS[int(op_idx - 1)]
            self.ops_list.append(OPS[op](self.current_C, self.current_C))

    def forward(self, inputs, stack_id):

        outputs = [0] * (self.NBR_HIDDEN_NODE + 1) # Store output of each operation
        
        if stack_id == 0:
            outputs[0] = inputs
        else:
            outputs[0] = self.ops_list[0](inputs) # Stem Convolution - Equalize channel count

        op_idx = 1
        for hidden_node_idx in range(1, self.NBR_HIDDEN_NODE + 1):
            # Get input nodes/edges to the hidden_node
            in_nodes = list(np.where(self.matrix[:, hidden_node_idx] == 1)[0])
            outputs[hidden_node_idx] = sum([self.ops_list[op_idx + i](outputs[in_nodes[i]]) 
                                                                    for i in range(len(in_nodes))])

            """
            test = []
            for i in range(len(in_nodes)):
                test.append(self.ops_list[op_idx + i](outputs[in_nodes[i]])) 
            outputs[hidden_node_idx] = sum(test) 
            """
            
            op_idx = op_idx + len(in_nodes)

        if self.matrix[0, self.NBR_HIDDEN_NODE] == 1:
            return outputs[-1]
        else:
            return outputs[0] + outputs[-1]