from intepreter import Intepreter
import numpy as np          

class Reflex():
    def __init__(self, genome):
        self.ninternal = 4
            
        self.enabled_inputs = np.zeros(len(Intepreter.InputNodes),dtype=bool)
        self.enabled_outputs = np.zeros(len(Intepreter.OutputNodes),dtype=bool)
        self.connections = self.get_connections(genome)

    def get_connections(self, genome):
        connections = dict([(k,[]) for k in ["io","is","ss","so"]]) # 4 types connections
        for gene in genome:
            in_type = gene>>24 &0x80 >>7
            in_id = gene>>24 & 0x7f
            out_type = gene>>16 &0x0080 >>7
            out_id = gene>>16 &0x007f
            weight = gene &0x0000ffff 
            weight = (weight - 2**15)/2**15*4
            
            if in_type and out_type:
                # from input to output
                in_node = in_id % len(Intepreter.InputNodes)
                out_node = out_id % len(Intepreter.OutputNodes)
                self.enabled_inputs[in_node]=True
                self.enabled_outputs[out_node]=True
                connections["io"].append((in_node, out_node, weight))
            elif in_type and not out_type:
                # from input to internal
                in_node = in_id % len(Intepreter.InputNodes)
                out_node = out_id % self.ninternal
                self.enabled_inputs[in_node]=True
                connections["is"].append((in_node, out_node, weight))
            elif not in_type and out_type:
                # from internal to output
                in_node = in_id % self.ninternal
                out_node = out_id % len(Intepreter.OutputNodes)
                self.enabled_outputs[out_node]=True
                connections["so"].append((in_node, out_node, weight))
            else:
                # from internal to internal
                in_node = in_id % self.ninternal
                out_node = out_id % self.ninternal
                connections["ss"].append((in_node, out_node, weight))
        return connections
    
    def forward(self, inputs):
        outputs = np.zeros(len(Intepreter.OutputNodes))
        internals = np.zeros(self.ninternal)

        for in_node, out_node, weight in self.connections["is"]:
            internals[out_node] += inputs[in_node] * weight
        for in_node, out_node, weight in self.connections["ss"]:
            internals[out_node] += internals[in_node] * weight
        internals = np.tanh(internals)
        
        for in_node, out_node, weight in self.connections["so"]:
            outputs[out_node] += internals[in_node] * weight
        for in_node, out_node, weight in self.connections["io"]:
            outputs[out_node] += inputs[in_node] * weight
            
        return outputs
    
    __call__=forward
                
    