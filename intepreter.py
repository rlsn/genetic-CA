import numpy as np

orientation = np.array([[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]])
agpx = np.array([-1, -0.5 ,0, 0.5, 1, 0.5, 0, -0.5])
agpy = np.array([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5])

trace_ratio = 0.5
overrest_return = 0.75
minimum_rp_to_repr = 0.8

movement_cpm = 0.3
rotate_cpm = 0.1
attack_cpm = 1

def sigmoid(x):
    return 1/(1 + np.exp(-x))

class Intepreter():
    InputNodes = [
        "Age", # age [0,1)
        "BlFw", # blockage forward
        "BlLF", # blockage left forward
        "BlRF", # blockage right forward

        "RsFw", # resource forward
        "RsLF", # resource left forward
        "RsRF", # resource right forward
        "RsGd", # resource gradient forward

        "Rnd", # random input
        "Cnst", # Constant 
        # "Lx", # loc x [0,1]
        # "Ly", # loc y [0,1]
        "AgPx", # angle wrt +x [-1,1] with +x:1, -x:-1
        "AgPy", # angle wrt +y [-1,1] with +y:1, -y:-1
        "NNgh", # number of neighbors [0,8]/8
        
        "Osc1s", # Oscillator 1 sin
        "Osc1c", # Oscillator 1 cos
        "Osc2s", # Oscillator 2 sin
        "Osc2c", # Oscillator 2 cos

        "Rp", # resource
        "Hp", # health
    ]
    OutputNodes = [
        "MvFw", # move forward
        "MvBw", # move backward
        "MvRn", # move randomly

        "RtLF", # rotate left forward
        "RtRF", # rotate right forward

        # "RtPx", # rotate towards +x
        # "RtNx", # rotate towards -x
        # "RtPy", # rotate towards +y
        # "RtNy", # rotate towards -y

        # "MvPx", # move towards +x
        # "MvNx", # move towards -x
        # "MvPy", # move towards +y
        # "MvNy", # move towards -y

        "Rest", # take a rest
        "AtkFw", # attack forward

        "Repr", # reproduce

        "ESFw", # emit resource forward
        "ESBw", # emit resource backward
        "ESAr", # emit resource around
    ]


    @staticmethod
    def block(loc,world):
        if not world.wrap and (loc[0]>=world.size or loc[0]<0 or loc[1]>=world.size or loc[1]<0):
            return 1
        if world.map[loc[0]%world.size,loc[1]%world.size]>0:
            return 1
        return 0
    
    # inputs
    @staticmethod
    def Age(creature, world):
        return creature.age/creature.life_expectency
    @staticmethod
    def Hp(creature, world):
        return creature.hp/creature.max_health
    @staticmethod
    def Rp(creature, world):
        return creature.rp/creature.max_resource
    @staticmethod
    def NNgh(creature, world):
        locs = creature.loc + orientation
        locs%=world.size
        return sum(world.map[locs[:,0],locs[:,1]]>0)/8    
    @staticmethod
    def Lx(creature, world):
        return creature.loc[0]/world.size
    @staticmethod
    def Ly(creature, world):
        return creature.loc[1]/world.size
    @staticmethod
    def AgPx(creature, world):
        return agpx[creature.r]
    @staticmethod
    def AgPy(creature, world):
        return agpy[creature.r]
    @staticmethod
    def Cnst(creature, world):
        return 1
    @staticmethod
    def Rnd(creature, world):
        return np.random.randn()
    @staticmethod
    def Osc1s(creature, world):
        return np.sin(creature.age/creature.osc1_period*2*np.pi)
    @staticmethod
    def Osc1c(creature, world):
        return np.cos(creature.age/creature.osc1_period*2*np.pi)
    @staticmethod
    def Osc2s(creature, world):
        return np.sin(creature.age/creature.osc2_period*2*np.pi)
    @staticmethod
    def Osc2c(creature, world):
        return np.cos(creature.age/creature.osc2_period*2*np.pi)
    @staticmethod
    def BlFw(creature, world):
        loc = creature.loc + orientation[creature.r]
        return Intepreter.block(loc,world)
    @staticmethod
    def BlLF(creature, world):
        loc = creature.loc + orientation[(creature.r-1)%8]
        return Intepreter.block(loc,world)
    @staticmethod
    def BlRF(creature, world):
        loc = creature.loc + orientation[(creature.r+1)%8]
        return Intepreter.block(loc,world)
    @staticmethod
    def RsFw(creature, world):
        loc = creature.loc + orientation[creature.r]
        loc%=world.size
        return world.res[loc[0],loc[1]]/creature.max_resource
    @staticmethod
    def RsLF(creature, world):
        loc = creature.loc + orientation[(creature.r-1)%8]
        loc%=world.size
        return world.res[loc[0],loc[1]]/creature.max_resource
    @staticmethod
    def RsRF(creature, world):
        loc = creature.loc + orientation[(creature.r+1)%8]
        loc%=world.size
        return world.res[loc[0],loc[1]]/creature.max_resource
    @staticmethod
    def RsGd(creature, world):
        loc1 = (creature.loc + orientation[creature.r])%world.size
        loc2 = (loc1 + orientation[creature.r])%world.size
        return (world.res[loc2[0],loc2[1]]-world.res[loc1[0],loc1[1]])/creature.max_resource

    # outputs 
    @staticmethod
    def Repr(creature, world, v):
        if np.random.rand()>v or not world.allow_repr:
            return
        if creature.rp<minimum_rp_to_repr*creature.max_resource:
            return

        loc = (creature.loc + np.random.randint(creature.spawn_range,size=2))%world.size
        if world.get_creature_at(loc) is None:
            r = creature.repl_resource * creature.rp
            creature.rp -= r
            c = creature.reproduce()
            c.loc = loc
            c.r = np.random.randint(8)
            c.rp = r
            c.generation = creature.generation + 1
            world.add_creature(c)


    @staticmethod
    def gather(creature, world, v):
        x,y = creature.loc
        creature.rp += world.res[x,y]
        if creature.rp>creature.max_resource:
            world.res[x,y] = creature.rp-creature.max_resource
            creature.rp=creature.max_resource

    @staticmethod
    def Rest(creature, world, v):
        recv = max(creature.rp, creature.rp * v)
        creature.rp -= recv
        creature.hp += recv
        if creature.hp>creature.max_health:
            creature.rp+=(creature.hp-creature.max_health)*overrest_return
            creature.hp=creature.max_health

        Intepreter.gather(creature, world, 0)

    @staticmethod
    def MvFw(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + orientation[creature.r]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            r = max(creature.rp, creature.mass * movement_cpm)
            world.res[creature.loc[0],creature.loc[1]] += r * trace_ratio
            creature.rp -= r
            world.move_creature(creature, loc)
            Intepreter.gather(creature, world, 0)
    @staticmethod
    def MvBw(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc - orientation[creature.r]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            r = max(creature.rp, creature.mass * movement_cpm)
            world.res[creature.loc[0],creature.loc[1]] += r * trace_ratio
            creature.rp -= r
            world.move_creature(creature, loc)
            Intepreter.gather(creature, world, 0)
    @staticmethod
    def MvRn(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + orientation[np.random.randint(len(orientation))]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            r = max(creature.rp, creature.mass * movement_cpm)
            world.res[creature.loc[0],creature.loc[1]] += r * trace_ratio
            creature.rp -= r
            world.move_creature(creature, loc)
            Intepreter.gather(creature, world, 0)
    @staticmethod
    def RtLF(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r-=1
        creature.r = creature.r%8
        
        creature.rp -= creature.mass * rotate_cpm
    @staticmethod
    def RtRF(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r+=1
        creature.r = creature.r%8
        creature.rp -= creature.mass * rotate_cpm

    @staticmethod
    def MvNx(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + np.array([-1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvPx(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + np.array([1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvPy(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + np.array([0,1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvNy(creature, world, v):
        if np.random.rand()>v:
            return
        loc = creature.loc + np.array([0,-1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)

    @staticmethod
    def RtPx(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r = 4
    @staticmethod
    def RtNx(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r = 0
    @staticmethod
    def RtPy(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r = 2
    @staticmethod
    def RtNy(creature, world, v):
        if np.random.rand()>v:
            return
        creature.r = 6

    @staticmethod
    def ESFw(creature, world, v):
        loc = creature.loc + orientation[creature.r]
        loc%=world.size
        r = max(creature.rp, v * creature.mass)
        creature.rp -= r
        world.res[loc[0],loc[1]]+= r
    @staticmethod
    def ESBw(creature, world, v):
        loc = creature.loc + orientation[(creature.r+4)%8]
        loc%=world.size
        r = max(creature.rp, v * creature.mass)
        creature.rp -= r
        world.res[loc[0],loc[1]]+= r
    @staticmethod
    def ESAr(creature, world, v):
        loc=creature.loc+orientation
        loc%=world.size
        r = max(creature.rp, v * creature.mass/8)
        creature.rp -= r
        world.res[loc[:,0],loc[:,1]]+= r
    @staticmethod
    def AtkFw(creature, world, v):
        loc = creature.loc + orientation[creature.r]
        loc%=world.size
        atk = max(creature.rp, creature.mass * creature.attack * v)
        creature.rp -= atk
        victim = world.get_creature_at(loc)
        if victim is not None:
            victim.hp -= atk - victim.defense

    # aggregation
    @staticmethod
    def aggregate(outputs):
        outputs = sigmoid(outputs)
        return outputs



if __name__=="__main__":
    unimpl = []
    for fn in Intepreter.InputNodes+Intepreter.OutputNodes:
        try:
            getattr(Intepreter, fn)
        except AttributeError:
            unimpl.append(fn)
    
    if len(unimpl)>0:
        print(f"unimplemented intepreters: {unimpl}")
    else:
        print("congrats! all intepreters are implemented")