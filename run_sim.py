from simulator import GridWorldSimulator
from creature import Creature
import argparse
import numpy as np
from tqdm import tqdm

wrap=True
size = 128
diffusion = 0.98
population = 1000
step_per_gen = 200
n_generation = 500
genome_size = 8
mutation = 0.002
save_name = "world.pkl"

new_ratio = 0.05

def select(world: GridWorldSimulator):
    delc = []
    for c in world.creatures.values():
        if c.rp<=0 or c.hp<=0:
            delc.append(c)
    for c in delc:
        world.remove_creature(c)

def repopulate(world: GridWorldSimulator):
    if len(world.creatures)<1:
        raise Exception("the population died out")
    parents = np.random.choice(list(world.creatures.values()),min(population*2,int(population*(1-new_ratio))),replace=True)
    world.clear_world()
    for p in parents:
        c = p.reproduce(mutation)
        c.generation = p.generation+1
        world.init_loc(c)
        world.add_creature(c)
    
    world.populate_number(population-len(world.creatures), genome_size = genome_size)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='simulator')
 
    parser.add_argument('--continue_sim','-c', action='store_true')
 
    args = parser.parse_args()
    if not args.continue_sim:
        world = GridWorldSimulator(size=size,wrap=wrap,diffusion=diffusion)
        world.populate_number(population, genome_size = genome_size)
        print("finished initialization")
    else:
        world = GridWorldSimulator.load(save_name)
        print("continue sim")


    for gen in range(n_generation):
        n_start = len(world.creatures)
        print(f"gen:{gen}, population:{len(world.creatures)}")
        for s in tqdm(range(step_per_gen)):
            world.step()

        select(world)
        print(f"finished gen:{gen}, population:{len(world.creatures)}, surviving ratio = {len(world.creatures)/n_start}")

        repopulate(world)
        world.save(save_name)