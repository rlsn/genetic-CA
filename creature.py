from reflex import Reflex
import numpy as np

mutation_table = np.array([0x1<<n for n in range(32)])

class Creature():
    def __init__(self, genome):
        self.genome = genome
        
        # states (variable during lifetime)
        self.age = 0
        
        self.loc = 0 # coordinate x,y
        self.r = 0 # orientation 0-7, 0 towards north, increment clockwise
        
        # traits (constant during lifetime)
        self.reflex = Reflex(genome)
        
                
    def reproduce(self, mutation_rate=0.001):
        repl = []
        for g in self.genome:
            for mut in mutation_table[np.random.rand(32)<mutation_rate]:
                g^=mut
            repl.append(g)
        return repl
    
    def step(self, inputs):
        self.age+=1
        return self.reflex(inputs)
        
    def __str__(self):
        return f"Creature @ {self.loc} twds {self.r} w/ genome: {[hex(g) for g in self.genome]}"
        
    __repr__=__str__