import numpy as np

class Intepreter():
    InputNodes = [
        "Age", # age (0-1)
        "BlFw", # blockage forward
        "BlLF", # blockage left forward
        "BlRF", # blockage right forward
        "Rnd", # random input
        "Osc", # Oscillator

    ]
    OutputNodes = [
        "MvFw", # move forward
        "MvBw", # move backward
        "RtLF", # rotate left forward
        "RtRF", # rotate right forward
        "MvRn", # move randomly
        "MvN", # move north
        "MvS", # move south
        "MvE", # move east
        "MvW", # move west
    ]

    orientation = np.array([[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]])
    
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
        return creature.age*0.005
    @staticmethod
    def Rnd(creature, world):
        return np.random.randn()
    @staticmethod
    def Osc(creature, world):
        return np.sin(creature.age/10*2*np.pi)
    @staticmethod
    def BlFw(creature, world):
        loc = creature.loc + Intepreter.orientation[creature.r]
        return Intepreter.block(loc,world)
    @staticmethod
    def BlLF(creature, world):
        loc = creature.loc + Intepreter.orientation[(creature.r-1)%8]
        return Intepreter.block(loc,world)
    @staticmethod
    def BlRF(creature, world):
        loc = creature.loc + Intepreter.orientation[(creature.r+1)%8]
        return Intepreter.block(loc,world)

    # outputs 
    @staticmethod
    def MvFw(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + Intepreter.orientation[creature.r]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvBw(creature, world, v):
        if v!=1:
            return
        loc = creature.loc - Intepreter.orientation[creature.r]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)

    @staticmethod
    def MvRn(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + Intepreter.orientation[np.random.randint(len(Intepreter.orientation))]
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)

    @staticmethod
    def RtLF(creature, world, v):
        if v!=1:
            return
        creature.r-=1
        creature.r = creature.r%8
    @staticmethod
    def RtRF(creature, world, v):
        if v!=1:
            return
        creature.r+=1
        creature.r = creature.r%8

    @staticmethod
    def MvN(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([-1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvS(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvE(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([0,1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvW(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([0,-1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)

    # aggregation

    @staticmethod
    def aggregate(outputs):
        outputs = outputs > 0.05
        return outputs
