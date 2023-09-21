from simulator import GridWorldSimulator

import numpy as np
from tqdm import tqdm

size = 128
population = 1000
step_per_gen = 200
n_generation = 500
genome_size = 8
mutation = 0.002
new_ratio = 0.05
save_name = "world.pkl"

world = GridWorldSimulator(size)
world.populate_number(population, genome_size = genome_size)
print("finished initialization")

def select(world: GridWorldSimulator):
    delc = []
    for c in world.creatures.values():
        if c.loc[0]<64:
            delc.append(c)
    for c in delc:
        world.remove_creature(c)

def repopulate(world: GridWorldSimulator):
    genomes = []
    parents = np.random.choice(list(world.creatures.values()),int(population*(1-new_ratio)),replace=True)
    for c in parents:
        genome = c.reproduce(mutation)
        genomes.append(genome)
    world.clear_world()
    world.populate_number(genomes=genomes)
    world.populate_number(int(population*new_ratio), genome_size = genome_size)

for gen in range(n_generation):
    n_start = len(world.creatures)
    print(f"gen:{gen}, population:{len(world.creatures)}")
    for s in tqdm(range(step_per_gen)):
        world.step()

    select(world)
    print(f"finished gen:{gen}, population:{len(world.creatures)}, surviving ratio = {len(world.creatures)/n_start}")

    repopulate(world)
    world.save(save_name)