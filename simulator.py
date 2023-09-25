import numpy as np
from scipy.signal import convolve2d
from creature import Creature
from intepreter import Intepreter
from operator import methodcaller
from functools import partial
import pickle

class GridWorldSimulator():
    
    def __init__(self, size=128, wrap=False, diffusion_map=None):
        self.size = size
        self.wrap = wrap

        if diffusion_map is None:
            diffusion_map = np.one(1)
        self.diffusion_map = diffusion_map

        self.step_fn = []
        self.new_creature_fn = []

        self.allow_repr = True
        
        self.clear_world()

        # statistics
        self.step_cnt = 0
        self.new_added_cnt = 0
                
    def __len__(self):
        return len(self.creatures)

    def get_creature_at(self, loc):
        cid = self.map[loc[0],loc[1]]
        if cid>0:
            return self.creatures[cid]
        else:
            return None
    
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
        assert self.map[c.loc[0],c.loc[1]]==0, "cannot add, place is occupied"
        self.new_id+=1
        self.creatures[self.new_id] = c
        self.map[c.loc[0],c.loc[1]]=self.new_id
        for fn in self.new_creature_fn:
            fn(self, c)
        self.new_added_cnt += 1

    def remove_creature(self,c):
        if type(c) != Creature:
            cid = c
            c = self.creatures[cid]
        else:
            cid = self.map[c.loc[0],c.loc[1]]
        
        self.map[c.loc[0],c.loc[1]] = 0
        self.res[c.loc[0],c.loc[1]] += max(c.rp,0)
        c.alive=False
        del self.creatures[cid]
    
    def clear_world(self):
        self.new_id = 0
        self.map = np.zeros((self.size,self.size),dtype=int)
        self.res = np.zeros((self.size,self.size),dtype=float)
        self.creatures = {}
    
    def init_loc(self,c):
        x,y = np.random.randint(self.size,size=2)
        n_try = 0
        while self.map[x,y]>0:
            if n_try>10:
                raise Exception("hard to find an empty space, world is too crowded")
            x,y = np.random.randint(self.size,size=2)
        c.loc = np.array([x,y])
        c.r = np.random.randint(8)
        return c
    
    def populate_density(self, density=0.1, genome_size = 8):
        positions = np.logical_and(np.random.rand(self.size,self.size)<density, self.map<1)
        positions = np.argwhere(positions==True)
        np.random.shuffle(positions)
        for x,y in positions:
            genome = np.random.randint(2**32,size=genome_size,dtype=np.uint32)
            # genome[1] = genome[1] & 0x0000ffff | 0x80800000 # make sure it's capable to repl
            c = Creature(genome)
            c.loc = np.array([x,y])
            c.r = np.random.randint(8)
            self.add_creature(c)
        
    def populate_number(self, n_creatures=100, genome_size = 8, genomes=None):
        if genomes is None:
            genomes = np.random.randint(2**32,size=(n_creatures,genome_size),dtype=np.uint32)

        for i,c in enumerate(range(len(genomes))):
            c = Creature(genomes[i])
            self.init_loc(c)
            self.add_creature(c)
        
    def step(self):
        self.step_cnt+=1
        self.new_added_cnt=0
        # diffuse resource
        self.res=convolve2d(self.res, self.diffusion_map, boundary='wrap', mode='same')

        # step creatures
        tmp = [(cid,c) for cid,c in self.creatures.items()]
        for cid,c in tmp:
            if not c.alive:
                continue
            # actions
            inputs = []
            for fn in c.reflex.enabled_inputs:
                inputs.append(methodcaller(fn, c, self)(Intepreter))

            outputs = Intepreter.aggregate(c.step(inputs))
            enabled_actions = [(fn,v) for fn,v in zip(c.reflex.enabled_outputs, outputs)]
            np.random.shuffle(enabled_actions)
            for action, value in enabled_actions:
                methodcaller(action, c, self, value)(Intepreter)

        # other step functions
        for fn in self.step_fn:
            fn(self)

    def save(self, filename):
        with open(filename,'wb') as wf:
            pickle.dump(self, wf)

    @classmethod
    def load(cls, filename):
        with open(filename,'rb') as rf:
            w=pickle.load(rf)
        return w