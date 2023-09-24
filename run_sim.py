from simulator import GridWorldSimulator
from creature import Creature
import argparse
import numpy as np
from functools import partial

wrap=True
world_size = 128
diffusion = 0.98
dissipation = 0.03

init_population = 800
init_genome_size = [4,8,12,16]

report_intv = 1

spawn_rate = 0.1
ideal_pop = 250
boost_pop = 50
pop_limit = 800

res_period = 250
avg_res = 0.5
amp_res = 0.5

save_name = "world.pkl"

diffusion_map = np.ones((3,3))*(1-diffusion)/8
diffusion_map[1,1]=diffusion-dissipation
avg_res*=world_size**2
amp_res*=world_size**2

def spawn(world: GridWorldSimulator):
    if np.random.rand()<(ideal_pop-len(world))/ideal_pop:
        world.populate_number(1, genome_size = np.random.choice(init_genome_size))
    if len(world)<boost_pop:
        for gsize in init_genome_size:
            world.populate_number(int(init_population/len(init_genome_size)), genome_size = gsize)

def supply(world: GridWorldSimulator, nlocs=100):
    locs = np.random.randint(world_size, size=(nlocs,2))
    sup = avg_res+amp_res*np.sin(world.step_cnt/res_period*np.pi*2)
    tot = world.res.sum() + sum([c.rp for c in world.creatures.values()])

    world.res[locs[:,0],locs[:,1]]+=max(0,sup-tot)/nlocs


def select(world: GridWorldSimulator):
    delc = []
    for c in world.creatures.values():
        if c.rp<=0 or c.hp<=0 or c.age>c.life_expectency:
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


def report(world):
    if world.step_cnt % report_intv==0:
        res = int(world.res.sum())
        rp = int(sum([c.rp for c in world.creatures.values()]))
        total_res = res+rp

        mass = int(sum([c.mass for c in world.creatures.values()]))
        max_mass = int(max([c.mass for c in world.creatures.values()]))
        min_mass = int(min([c.mass for c in world.creatures.values()]))

        print(f"step={world.step_cnt}, pop={len(world)}, res={res}, rp={rp}, total_res={total_res}, mass={mass}, max_mass={max_mass}, min_max={min_mass}")
        world.save(save_name)

def init_world(args):
    if not args.continue_sim:
        world = GridWorldSimulator(size=world_size,wrap=wrap,diffusion_map=diffusion_map)
        for gsize in init_genome_size:
            world.populate_number(int(init_population/len(init_genome_size)), genome_size = gsize)

        world.step_fn = [
             select,
             end_world,
             spawn,
             supply,
             report
        ]        
        print("finished initialization")
    else:
        world = GridWorldSimulator.load(save_name)
        print("continue sim")
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