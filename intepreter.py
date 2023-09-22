import numpy as np

class Intepreter():
    InputNodes = [
        "Age", # age [0,inf)
        "BlFw", # blockage forward
        "BlLF", # blockage left forward
        "BlRF", # blockage right forward
        "Rnd", # random input
        "Osc", # Oscillator
        "Cnst", # Constant 1
        "AgPx", # angle wrt +x [-1,1] with +x:1, -x:-1
        "AgPy", # angle wrt +y [-1,1] with +y:1, -y:-1
    ]
    OutputNodes = [
        "MvFw", # move forward
        "MvBw", # move backward
        "RtLF", # rotate left forward
        "RtRF", # rotate right forward

        "RtPx", # rotate towards +x
        "RtNx", # rotate towards -x
        "RtPy", # rotate towards +y
        "RtNy", # rotate towards -y

        "MvRn", # move randomly
        "MvPx", # move towards +x
        "MvNx", # move towards -x
        "MvPy", # move towards +y
        "MvNy", # move towards -y
    ]

    orientation = np.array([[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1]])
    agpx = np.array([-1, -0.5 ,0, 0.5, 1, 0.5, 0, -0.5])
    agpy = np.array([0, 0.5, 1, 0.5, 0, -0.5, -1, -0.5])

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
    def AgPx(creature, world):
        return Intepreter.agpx[creature.r]
    @staticmethod
    def AgPy(creature, world):
        return Intepreter.agpy[creature.r]
    @staticmethod
    def Cnst(creature, world):
        return 1
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
    def MvNx(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([-1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvPx(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([1,0])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvPy(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([0,1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)
    @staticmethod
    def MvNy(creature, world, v):
        if v!=1:
            return
        loc = creature.loc + np.array([0,-1])
        if not Intepreter.block(loc, world):
            loc = loc%world.size
            world.move_creature(creature, loc)

    @staticmethod
    def RtPx(creature, world, v):
        if v!=1:
            return
        creature.r = 4
    @staticmethod
    def RtNx(creature, world, v):
        if v!=1:
            return
        creature.r = 0
    @staticmethod
    def RtPy(creature, world, v):
        if v!=1:
            return
        creature.r = 2
    @staticmethod
    def RtNy(creature, world, v):
        if v!=1:
            return
        creature.r = 6

    # aggregation
    @staticmethod
    def aggregate(outputs):
        outputs = outputs > 0.05
        return outputs


if __name__=="__main__":
    for fn in Intepreter.InputNodes+Intepreter.OutputNodes:
        try:
            getattr(Intepreter, fn)
        except AttributeError:
            raise NotImplementedError(f"intepreter '{fn}' is not implemented")
    print("congrats! all intepreters are implemented")