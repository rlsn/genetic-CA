import numpy as np
from creature import Creature
from intepreter import Intepreter
from operator import methodcaller
import pickle

class GridWorldSimulator():
    
    def __init__(self, size=128, wrap=False):
        self.size = size
        self.wrap = wrap
        self.clear_world()
                
    def get_cid(self, c):
        return self.map[c.loc[0],c.loc[1]]
            
    def move_creature(self,c,loc):
        if type(c) != Creature:
            cid = c
            c = self.creatures[cid]
        else:
            cid = self.map[c.loc[0],c.loc[1]]
        self.map[c.loc[0],c.loc[1]] = 0
        self.map[loc[0],loc[1]] = cid
        c.loc = np.array(loc)
        
    def add_creature(self,c):
        self.new_id+=1
        self.creatures[self.new_id] = c
        self.map[c.loc[0],c.loc[1]]=self.new_id
    
    def remove_creature(self,c):
        if type(c) != Creature:
            cid = c
            c = self.creatures[cid]
        else:
            cid = self.map[c.loc[0],c.loc[1]]
        
        self.map[c.loc[0],c.loc[1]] = 0
        del self.creatures[cid]
    
    def clear_world(self):
        self.new_id = 0
        self.map = np.zeros((self.size,self.size),dtype=int)
        self.creatures = {}
    
    def populate_density(self, density=0.1, genome_size = 8):
        positions = np.logical_and(np.random.rand(self.size,self.size)<density, self.map<1)
        positions = np.argwhere(positions==True)
        np.random.shuffle(positions)
        for x,y in positions:
            genome = np.random.randint(2**32,size=genome_size,dtype=np.uint32)
            c = Creature(genome)
            c.loc = np.array([x,y])
            c.r = np.random.randint(8)
            self.add_creature(c)
        
    def populate_number(self, n_creatures=100, genome_size = 8, genomes=None):
        if genomes is None:
            genomes = np.random.randint(2**32,size=(n_creatures,genome_size),dtype=np.uint32)

        rotations = np.random.randint(8, size=len(genomes))
        for i,c in enumerate(range(len(genomes))):
            x,y = np.random.randint(self.size,size=2)
            n_try = 0
            c = Creature(genomes[i])
            c.r=rotations[i]
            while self.map[x,y]>0:
                if n_try>10:
                    raise Exception("failed to populate, world is too crowded")
                x,y = np.random.randint(self.size,size=2)
            c.loc = np.array([x,y])
            self.add_creature(c)
        
    def step(self):
        for cid,c in self.creatures.items():
            inputs = [methodcaller(fn, c, self)(Intepreter) for fn in Intepreter.InputNodes]
            outputs = Intepreter.aggregate(c.step(inputs))
            enabled_actions = [(Intepreter.OutputNodes[i],outputs[i]) for i in range(len(Intepreter.OutputNodes)) if c.reflex.output_enabled[i]]
            np.random.shuffle(enabled_actions)
            for action, value in enabled_actions:
                methodcaller(action, c, self, value)(Intepreter)
            

    def save(self, filename):
        with open(filename,'wb') as wf:
            pickle.dump(self, wf)

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as rf:
            w=pickle.load(rf)
        return w