from simulator import GridWorldSimulator
from intepreter import stats
from creature import Creature
import argparse
import numpy as np
from functools import partial
from utils import MovingSupply
wrap=True
world_size = 96
diffusion = 0.05
dissapation = 0.995

init_population = 800
init_genome_size = [4,8,12,16]

report_intv = 1

ideal_pop = 250
boost_pop = 20
pop_limit = 800

ms = [
    MovingSupply(world_size,[(2000,0.6,0.3),(160,0,0.8)], 10, 4, 2, 8),
    MovingSupply(world_size,[(500,-0.5,1.5)], 10, 3, 2, 8),
    MovingSupply(world_size,[(700,-0.6,1.8)], 10, 2, 1, 16),
    MovingSupply(world_size,[(50,0.5,0.5)], 8, 4, 5, 4),
    MovingSupply(world_size,[(1200,0.2,0.4)], 10, 5, 1, 4),
    MovingSupply(world_size,[(80,0.7,0.4)], 8, 4, 3, 4),
    MovingSupply(world_size,[(60,0.8,0.8)], 4, 20, 1, 4),
]

save_name = "world.pkl"

def spawn(world: GridWorldSimulator):
    if np.random.rand()<(ideal_pop-len(world))/ideal_pop:
        world.populate_number(int((ideal_pop-len(world))/10)*0+1, genome_size = np.random.choice(init_genome_size))
        # print("added 1 crt")
    if len(world)<boost_pop:
        for gsize in init_genome_size:
            world.populate_number(int(init_population/len(init_genome_size)), genome_size = gsize)
        print("boosted pop")

def supply(world: GridWorldSimulator):
    for m in ms:
        m.step(world)
def select(world: GridWorldSimulator):
    delc = []
    for c in world.creatures.values():
        if c.rp<=c.max_resource*0.05 or c.hp<=0 or c.age>c.life_expectency:
            delc.append(c)
    for c in delc:
        world.remove_creature(c)

    world.allow_repr=len(world)<=pop_limit


def end_world(world):
    if len(world)<=10:
        print("dead world, game over")
        exit()

    if len(world)>2000:
        print("over crowded, game over")
        exit()


def report(world, printout = True):
    if world.step_cnt % report_intv==0:
        res = int(world.res.sum())
        rp = int(sum([c.rp for c in world.creatures.values()]))
        total_res = res+rp

        all_mass = [c.mass for c in world.creatures.values()]
        mass = int(sum(all_mass))
        max_mass = round(max(all_mass),1)
        min_mass = round(min(all_mass),1)

        rep = f"step={world.step_cnt}, pop={len(world)}, rp={rp}, res={res}, total_res={total_res}, mass={mass}, max_mass={max_mass}, min_max={min_mass}, new_born={world.new_added_cnt}"
        if printout:
            print(rep)
    return rep

def save(world):
    if world.step_cnt % report_intv==0:
        world.save(save_name)

def init_world(args):
    if not args.continue_sim:
        world = GridWorldSimulator(size=world_size,wrap=wrap,diffusion=diffusion,dissapation=dissapation)
        for gsize in init_genome_size:
            world.populate_number(int(init_population/len(init_genome_size)), genome_size = gsize)
        print("finished initialization")
    else:
        world = GridWorldSimulator.load(save_name)
        print("continue sim")


    world.step_fn = [
            select,
            spawn,
            end_world,
            supply,
            report,
            save,
        ]
    return world

def run(args):

    world = init_world(args)
    # last_v = 50000000
    while True:
        world.step()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='simulator')
 
    parser.add_argument('--continue_sim','-c', action='store_true')
 
    args = parser.parse_args()

    run(args)