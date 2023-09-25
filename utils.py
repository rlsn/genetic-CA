from graphviz import Digraph
import numpy as np

def color_mapping(genome):
    n = [0,0,0]
    rgb = [0,0,0]
    for i,g in enumerate(genome):
        rgb[i%3]+=g//2**24
        n[i%3]+=1
    return tuple([max(min(c//m,255),0) for c,m in zip(rgb,n)])

def visualize_reflex(reflex, graphname='reflex.tmp'):
    g = Digraph(graphname)

    with g.subgraph(name='cluster_0') as c:
        c.attr(color='lightblue2')
        c.node_attr['style'] = 'filled'
        c.node_attr['color'] = 'lightblue2'
        for n in reflex.enabled_inputs:
            c.node(n)

    # with g.subgraph(name='cluster_1') as c:
    #     c.attr(style='filled', color='lightgrey')
    #     c.node_attr.update(style='filled', color='white')
        
    #     for i in range(reflex.ninternal):
    #         c.node(f'i{i}')
        
    with g.subgraph(name='cluster_1') as c:
        c.attr(color='lightgoldenrod1')
        c.node_attr['style'] = 'filled'
        c.node_attr['color'] = 'lightgoldenrod1'
        
        for n in reflex.enabled_outputs:
            c.node(n)

    for i,o,v in reflex.connections["io"]:
        g.edge(reflex.enabled_inputs[i], reflex.enabled_outputs[o],
            color='lightgreen' if v>0 else 'lightpink', penwidth=str(max(0.8,min(5,abs(v)))), label=str(round(v,2)))
    for i,o,v in reflex.connections["is"]:
        g.edge(reflex.enabled_inputs[i], f'i{o}', 
            color='lightgreen' if v>0 else 'lightpink', penwidth=str(max(0.8,min(5,abs(v)))), label=str(round(v,2)))
    for i,o,v in reflex.connections["ss"]:
        g.edge(f'i{i}', f'i{o}', 
            color='lightgreen' if v>0 else 'lightpink', penwidth=str(max(0.8,min(5,abs(v)))), label=str(round(v,2)))
    for i,o,v in reflex.connections["so"]:
        g.edge(f'i{i}', reflex.enabled_outputs[o],
            color='lightgreen' if v>0 else 'lightpink', penwidth=str(max(0.8,min(5,abs(v)))), label=str(round(v,2)))
    
    return g

class MovingSupply():
    def __init__(self, world_size, Tma_list, size, move_freq, move_step, n_scatter):
        self.Tma_list=Tma_list # (peroid, mean, amp)
        self.size = size
        self.radius=size//2
        self.move_freq=move_freq
        self.move_step=move_step
        self.n_scatter=n_scatter
        self.ncells = (self.size)**2

        self.loc = np.random.randint(world_size,size=2)

    def step(self, world):
        locs = (self.loc+np.random.randint(-self.radius,self.radius, size=(self.n_scatter,2)))%world.size
        if world.step_cnt%self.move_freq==0:
            self.loc=(self.loc+np.random.randint(self.move_step, size=2))%world.size
        v = 0
        for t,m,a in self.Tma_list:
            v+=(self.ncells*m+ self.ncells*a*np.sin(world.step_cnt/t*np.pi*2))/t*np.pi
        world.res[locs[:,0],locs[:,1]]+=max(0,v)