import pygame as pg
from simulator import GridWorldSimulator
from intepreter import Intepreter
import numpy as np
from utils import *
from run_sim import init_world
import argparse

screen_size = (2048, 960)
map_offset = (150,100)
map_size = 768
margin = 20
boundary_thickness = 5

reflex_filename = "reflex.tmp.gv.png"
reflex_png = None

play_button_rect = (600,40,120,32)
showres_button_rect = (200,40,120,32)

step_text_loc = (800,40)
info_text_loc = (1000,60)
reflex_png_loc = (1000,500)

###############################################################################

parser = argparse.ArgumentParser(description='simulator gui')
parser.add_argument('--continue_sim','-c', action='store_true')
args = parser.parse_args()

###############################################################################
def draw_reflex(c):
    global reflex_png
    g=visualize_reflex(c.reflex, graphname='reflex.tmp')
    g.render(format='png')
    reflex_png = pg.image.load(reflex_filename)
    reflex_png.convert()

def assign_color(world, creature):
    creature.color = color_mapping(creature.genome)
def get_world():
    world = init_world(args)
    for c in world.creatures.values():
        c.color = color_mapping(c.genome)
    world.new_creature_fn.append(assign_color)
    return world

world = get_world()
scale = map_size//world.size
cell_size = map_size//world.size//2

# pygame setup
pg.init()
screen = pg.display.set_mode(screen_size)
clock = pg.time.Clock()
running = True

pg.font.init()
font = pg.font.Font('freesansbold.ttf', 32)


playing = False
highlight = None
show_res = False
def init():
    global step, playing, highlight
    playing = False
    highlight = None
def set_highlight(c):
    global highlight
    highlight = c
    draw_reflex(highlight)

###############################################################################

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                playing=not playing
            if event.key == pg.K_r:
                    show_res=not show_res
            if highlight is not None:
                if event.key == pg.K_RIGHT or event.key == pg.K_d:
                    loc = [highlight.loc[0]+1, highlight.loc[1]]
                    while loc[0]<world.size:
                        c = world.get_creature_at(loc)
                        if c:
                            set_highlight(c)
                            break
                        loc[0]+=1
                elif event.key == pg.K_LEFT or event.key == pg.K_a:
                    loc = [highlight.loc[0]-1, highlight.loc[1]]
                    while loc[0]>0:
                        c = world.get_creature_at(loc)
                        if c:
                            set_highlight(c)
                            break
                        loc[0]-=1
                elif event.key == pg.K_UP or event.key == pg.K_w:
                    loc = [highlight.loc[0], highlight.loc[1]-1]
                    while loc[1]>0:
                        c = world.get_creature_at(loc)
                        if c:
                            set_highlight(c)
                            break
                        loc[1]-=1
                elif event.key == pg.K_DOWN or event.key == pg.K_s:
                    loc = [highlight.loc[0], highlight.loc[1]+1]
                    while loc[1]<world.size:
                        c = world.get_creature_at(loc)
                        if c:
                            set_highlight(c)
                            break
                        loc[1]+=1
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                if pg.Rect(*play_button_rect).collidepoint(event.pos):
                    playing=not playing
                if pg.Rect(*showres_button_rect).collidepoint(event.pos):
                    show_res
                elif pg.Rect(*map_offset,map_size,map_size).collidepoint(event.pos):
                    for c in world.creatures.values():
                        if -2<c.loc[0]-(event.pos[0]-map_offset[0])//scale<2 and -2<c.loc[1]-(event.pos[1]-map_offset[1])//scale<2:
                            set_highlight(c)
                            break
            elif event.button == 3:
                highlight=None

    mouse = pg.mouse.get_pos()
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # RENDER YOUR GAME HERE

    # ui
    pg.draw.rect(screen, "red", play_button_rect)
    play_text = font.render("Pause" if playing else "Play", True, "white")
    screen.blit(play_text, play_button_rect[:2])

    pg.draw.rect(screen, "purple", showres_button_rect)
    showres_text = font.render("Show Res", True, "white")
    screen.blit(showres_text, showres_button_rect[:2])

    # step_text = font.render(f'Step: {world.step_cnt}', True, "green", "blue")
    # screen.blit(step_text, step_text_loc)

    if highlight is not None:
        infos = [
            f'id = {world.get_cid(highlight)}, loc = {(highlight.loc[0],highlight.loc[1])}, age={highlight.age}/{highlight.life_expectency}',
            f'generation={highlight.generation}, # gene={len(highlight.genome)}, spawn range={highlight.spawn_range}, # children={highlight.n_children}',
            f'rp={round(highlight.rp,2)}/{highlight.max_resource}, hp={round(highlight.attack,2)}/{highlight.max_health}',
            f'mass={highlight.mass}, atk={highlight.attack}, def={highlight.defense}',
            f'mem = {", ".join(["%.2f"%x for x in highlight.reflex.mem])}',
            f'    = [{", ".join(["%.2f"%x for x in highlight.recent_input_values])}]',
            f'inputs: {highlight.reflex.enabled_inputs}',
            f'    = [{", ".join(["%.2f"%x for x in highlight.recent_input_values])}]',
            f'outputs: {highlight.reflex.enabled_outputs}',
            f'    = [{", ".join(["%.2f"%x for x in highlight.recent_output_values])}]',

        ]
        for i, line in enumerate(infos):
            info_text = font.render(line, True, "black")
            screen.blit(info_text, (info_text_loc[0],info_text_loc[1]+i*32))
    
        screen.blit(reflex_png,reflex_png_loc)


    # map area
    pg.draw.rect(screen, "black", 
                     (map_offset[0]-margin,map_offset[1]-margin,map_size+margin*2,map_size+margin*2), width=boundary_thickness)

    if show_res:
        w = cell_size *2
        for x in range(world.size):
            for y in range(world.size):
                rgb = max(0,min(255,255-world.res[x,y]/64*255))
                pg.draw.rect(screen, (rgb,rgb,rgb), 
                        (map_offset[0]+x*scale,map_offset[1]+y*scale,w,w))

    for c in world.creatures.values():
        if highlight is not None and world.get_cid(highlight)==world.get_cid(c):
            pg.draw.circle(screen, "red", 
                           (map_offset[0]+c.loc[0]*scale,map_offset[1]+c.loc[1]*scale), cell_size+2, width=2)
        pg.draw.circle(screen, c.color, 
                           (map_offset[0]+c.loc[0]*scale,map_offset[1]+c.loc[1]*scale), cell_size)

    # logic
    if playing:
        world.step()
    # flip() the display to put your work on screen
    pg.display.flip()

    clock.tick(60)  # limits FPS to 60

pg.quit()