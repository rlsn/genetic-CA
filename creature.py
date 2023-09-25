import numpy as np
from intepreter import Intepreter

mutation_table = np.array([0x1<<n for n in range(32)])

base_cpm = 0.005
genome_mass = 0.1

class Reflex():
    def __init__(self, genome, ninternal, nmem):
        self.ninternal = ninternal
        self.nmem = nmem

        self.mem = np.zeros(nmem)
        self.enabled_inputs = []
        self.enabled_outputs = []
        self.connections = self.get_connections(genome)

    def get_connections(self, genome):
        connections = dict([(k,[]) for k in ["io","is","ss","so"]]) # 4 types connections
        inputs_enabled = dict()
        outputs_enabled = dict()

        for gene in genome:
            # 32 bits: [1 in_type] [7 in_id] [1 out_type] [7 out_id] [16 weight]
            in_type = gene>>24 &0x80 >>7
            in_id = gene>>24 & 0x7f
            out_type = gene>>16 &0x80 >>7
            out_id = gene>>16 &0x7f
            weight = gene &0xffff
            weight = (weight - 2**15)/2**15*4
            
            if in_type and out_type:
                # from input to output
                in_node = Intepreter.InputNodes[in_id % len(Intepreter.InputNodes)]
                if in_node not in inputs_enabled:
                    in_pos = len(self.enabled_inputs)
                    self.enabled_inputs.append(in_node)
                    inputs_enabled[in_node] = in_pos
                else:
                    in_pos = inputs_enabled[in_node]
                out_node = Intepreter.OutputNodes[out_id % len(Intepreter.OutputNodes)]
                if out_node not in outputs_enabled:
                    out_pos = len(self.enabled_outputs)
                    self.enabled_outputs.append(out_node)
                    outputs_enabled[out_node] = out_pos
                else:
                    out_pos = outputs_enabled[out_node]
                connections["io"].append((in_pos, out_pos, weight))
            elif in_type and not out_type:
                # from input to internal
                in_node = Intepreter.InputNodes[in_id % len(Intepreter.InputNodes)]
                if in_node not in inputs_enabled:
                    in_pos = len(self.enabled_inputs)
                    self.enabled_inputs.append(in_node)
                    inputs_enabled[in_node] = in_pos
                else:
                    in_pos = inputs_enabled[in_node]
                out_pos = out_id % (self.ninternal+self.nmem)
                connections["is"].append((in_pos, out_pos, weight))
            elif not in_type and out_type:
                # from internal to output
                in_pos = in_id % (self.ninternal+self.nmem)
                out_node = Intepreter.OutputNodes[out_id % len(Intepreter.OutputNodes)]
                if out_node not in outputs_enabled:
                    out_pos = len(self.enabled_outputs)
                    self.enabled_outputs.append(out_node)
                    outputs_enabled[out_node] = out_pos
                else:
                    out_pos = outputs_enabled[out_node]
                connections["so"].append((in_pos, out_pos, weight))
            else:
                # from internal to internal
                in_pos = in_id % (self.ninternal+self.nmem)
                out_pos = out_id % self.ninternal
                connections["ss"].append((in_pos, out_pos, weight))
        
        # clean useless connections
        i_reachable = set()
        o_reachable = set()
        for conn in connections["is"]:
            i_reachable.add(conn[1])
        for conn in connections["so"]:
            o_reachable.add(conn[0])
        for conn in connections["ss"]:
            if conn[0] in i_reachable:
                i_reachable.add(conn[1])
            if conn[1] in o_reachable:
                o_reachable.add(conn[0])
                
        valid = i_reachable.intersection(o_reachable)

        connections["is"] = [conn for conn in connections["is"] if conn[1] in valid]
        connections["ss"] = [conn for conn in connections["ss"] if conn[0] in valid and conn[1] in valid]
        connections["so"] = [conn for conn in connections["so"] if conn[0] in valid]

        return connections
    
    def forward(self, inputs):
        outputs = np.zeros(len(self.enabled_outputs))
        cells = np.concatenate([self.mem, np.zeros(self.ninternal)])

        for in_node, out_node, weight in self.connections["is"]:
            cells[out_node] += inputs[in_node] * weight
        for in_node, out_node, weight in self.connections["ss"]:
            cells[out_node] += cells[in_node] * weight
        cells = np.tanh(cells)
        self.mem = cells[:len(self.mem)]
        for in_node, out_node, weight in self.connections["so"]:
            outputs[out_node] += cells[in_node] * weight
        for in_node, out_node, weight in self.connections["io"]:
            outputs[out_node] += inputs[in_node] * weight
            
        return outputs
    
    __call__=forward
                

class Creature():
    def __init__(self, genome):
        
        self.genome = genome

        # traits (constant during lifetime)
        self.solve_traits(genome[0])
        self.reflex = Reflex(genome[1:], self.ninternal, self.nmem)
        self.mutation_rate=0.002
        self.shift_rate=0.001


        # states (variable during lifetime)
        self.age = 0
        self.rp = self.max_resource*0.8
        self.hp = self.max_health*0.8

        self.loc = 0 # coordinate x,y
        self.r = 0 # orientation 0-7, 0 towards north, increment clockwise

        # statistics for performance tracking
        self.generation = 0
        self.recent_input_values = []
        self.recent_output_values = []
        self.n_children = 0
    def solve_traits(self, gene):
        # 32 bits:
        # 8: [1?] [3 life_expectency(log)] [2 ninternal(log)] [2 nmem(log)] 
        # 16: [3 attack] [3 defense] [3 max_health] [3 max_resource] [4 repl_resource] (all log)
        # 8: [3 Osc1_period (log)] [3 Osc2_period (log)] [2 spawn_range(log)]

        self.life_expectency = 2**((gene>>24 &0x70 >>4)+2)
        self.ninternal = 2**(gene>>24 &0xc >> 2)
        self.nmem = 2**(gene>>24 &0x3)

        self.attack = 2**(gene>>8 &0xe000 >> 13)
        self.defense = 2**(gene>>8 &0x1c00 >> 10)
        self.max_health = 2**(gene>>8 &0x380 >> 7)
        self.max_resource = 2**(gene>>8 &0x70 >> 4)
        self.repl_resource = 0.5**(gene>>8 &0xf)

        self.osc1_period = 2**(gene &0xe0>>5)
        self.osc2_period = 2**(gene &0x1c>>2)
        self.spawn_range = 2**(gene &0x3)
                
        self.mass = np.log2(self.attack * self.defense * self.max_health * self.max_resource) + len(self.genome) * genome_mass

    def reproduce(self):
        self.n_children+=1
        repl = []
        for g in self.genome:
            # point mutation
            repl.append(g^sum(mutation_table[np.random.rand(32)<self.mutation_rate]))
        repl = np.array(repl)

        if np.random.rand()<self.shift_rate:
            if np.random.rand()<0.5:
                # insertion
                np.insert(repl,
                          np.random.randint(len(repl)+1),
                          np.random.randint(2**32,dtype=np.uint32))
            else:
                if len(repl)>1:
                    # delete
                    np.delete(repl,np.random.randint(len(repl)))
        return Creature(repl)
    
    def step(self, inputs):
        self.rp -= base_cpm * self.mass
        self.age+=1

        self.recent_input_values = inputs
        outputs = self.recent_output_values = self.reflex(inputs)
        return outputs
        
    def __str__(self):
        return f"Creature @ {self.loc} twds {self.r} w/ genome: {[hex(g) for g in self.genome]}"
        
    __repr__=__str__