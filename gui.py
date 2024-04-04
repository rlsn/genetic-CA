import pygame as pg
from simulator import GridWorldSimulator
from intepreter import Intepreter, stats
import numpy as np
from utils import *
from run_sim import init_world
import argparse

screen_size = [1800, 960]

step_size = 5
map_offset = (150,100)
map_size = 768
margin = 20
boundary_thickness = 5

reflex_filename = "reflex.tmp.gv.png"
export_genome_filename = "genome.txt"

reflex_png = None

play_button_rect = (600,40,120,32)
showres_button_rect = (200,40,160,32)
export_button_rect = (1400,60,250,32)

step_text_loc = (800,40)
info_text_loc = (1000,60)
map_text_loc = (150,960)

reflex_png_loc = (1000,500)
reflex_png_size = (500,368)

res_color = (250, 200, 120)
###############################################################################

parser = argparse.ArgumentParser(description='simulator gui')
parser.add_argument('--continue_sim','-c', action='store_true')
args = parser.parse_args()

###############################################################################
def draw_reflex(c):
    global reflex_png
    g=visualize_reflex(c.reflex, graphname='reflex.tmp')
    g.graph_attr.update(size=f"{reflex_png_size[0]},{reflex_png_size[1]}", dpi="300")
    g.render(format='png')
    reflex_png = pg.image.load(reflex_filename)
    reflex_png.convert()
    w,h = reflex_png.get_width(),reflex_png.get_height()
    scl = max(w/reflex_png_size[0],h/reflex_png_size[1])
    reflex_png = pg.transform.scale(reflex_png, (int(w/scl),int(h/scl)))

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

Rotation_NumPad = [8,9,6,3,2,1,4,7]

# pygame setup
pg.init()
screen = pg.display.set_mode(screen_size)
clock = pg.time.Clock()
running = True

pg.font.init()
font = pg.font.Font('freesansbold.ttf', 24)

single_step = False
playing = False
highlight = None
cursor = None
show_res = True
def init():
    global step, playing, highlight
    playing = False
    highlight = None
def set_highlight(c):
    global highlight
    highlight = c
    if c is not None:
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
            if event.key == pg.K_RIGHT:
                single_step=True
            if cursor is not None:
                if event.key == pg.K_d:
                    cursor = [(cursor[0]+1)%world.size, cursor[1]]
                    set_highlight(world.get_creature_at(cursor))
                elif event.key == pg.K_a:
                    cursor = [(cursor[0]-1)%world.size, cursor[1]]
                    set_highlight(world.get_creature_at(cursor))
                elif event.key == pg.K_w:
                    cursor = [cursor[0], (cursor[1]-1)%world.size]
                    set_highlight(world.get_creature_at(cursor))
                elif event.key == pg.K_s:
                    cursor = [cursor[0], (cursor[1]+1)%world.size]
                    set_highlight(world.get_creature_at(cursor))
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                if pg.Rect(*play_button_rect).collidepoint(event.pos):
                    playing=not playing
                elif pg.Rect(*showres_button_rect).collidepoint(event.pos):
                    show_res=not show_res
                elif pg.Rect(*map_offset,map_size,map_size).collidepoint(event.pos):
                    cursor = [(event.pos[0]-map_offset[0])//scale, (event.pos[1]-map_offset[1])//scale]
                    set_highlight(world.get_creature_at(cursor))
                elif highlight is not None and pg.Rect(*export_button_rect).collidepoint(event.pos):
                    with open(export_genome_filename, "w") as wf:
                        wf.write(", ".join([str(g) for g in highlight.genome]))
            elif event.button == 3:
                highlight=None
                cursor=None

    mouse = pg.mouse.get_pos()
    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")

    # RENDER YOUR GAME HERE

    # ui
    pg.draw.rect(screen, "red", play_button_rect)
    play_text = font.render("Pause" if playing else "Play", True, "white")
    screen.blit(play_text, play_button_rect[:2])

    pg.draw.rect(screen, "purple", showres_button_rect)
    showres_text = font.render("Hide res" if show_res else "Show res", True, "white")
    screen.blit(showres_text, showres_button_rect[:2])

    # step_text = font.render(f'Step: {world.step_cnt}', True, "green", "blue")
    # screen.blit(step_text, step_text_loc)

    if highlight is not None:
        cursor = highlight.loc
        pg.draw.rect(screen, "blue", export_button_rect)
        export_text = font.render("Export Genome", True, "white")
        screen.blit(export_text, export_button_rect[:2])


        infos = [
            f'id = {world.get_cid(highlight)}, age={highlight.age}/{highlight.life_expectency}, {"alive" if highlight.alive else "dead"}',
            f'loc = {(highlight.loc[0],highlight.loc[1])}, rot = {Rotation_NumPad[highlight.r]}',
            f'generation={highlight.generation}, # gene={len(highlight.genome)}, spawn range={highlight.spawn_range}, # children={highlight.n_children}',
            f'rp={round(highlight.rp,2)}/{highlight.max_resource}, hp={round(highlight.hp,2)}/{highlight.max_health}',
            f'mass={highlight.mass}, atk={highlight.attack}, def={highlight.defense}',
            f'# internal={highlight.nmem}, {highlight.ninternal}',

            f'mem = [{", ".join(["%.2f"%x for x in highlight.reflex.mem])}]',
            f'inputs: {highlight.reflex.enabled_inputs}',
            f'    = [{", ".join(["%.2f"%x for x in highlight.recent_input_values])}]',
            f'outputs: {highlight.reflex.enabled_outputs}',
            f'    = [{", ".join(["%.2f"%x for x in highlight.recent_output_values])}]',

        ]
        for i, line in enumerate(infos):
            info_text = font.render(line, True, "black")
            screen.blit(info_text, (info_text_loc[0],info_text_loc[1]+i*40))
    
        screen.blit(reflex_png,reflex_png_loc)


    # map area
    pg.draw.rect(screen, "black", 
                     (map_offset[0]-margin,map_offset[1]-margin,map_size+margin*2,map_size+margin*2), width=boundary_thickness)

    if show_res:
        ref_res = 16
        w = cell_size *2
        for x in range(world.size):
            for y in range(world.size):
                a = max(0,min(1,world.res[x,y]/ref_res))
                pg.draw.rect(screen, pg.Color(255,255,255).lerp(res_color,a),
                        (map_offset[0]+x*scale-cell_size,map_offset[1]+y*scale-cell_size,w,w))

    for c in world.creatures.values():
        pg.draw.circle(screen, c.color, 
                           (map_offset[0]+c.loc[0]*scale,map_offset[1]+c.loc[1]*scale), cell_size)

    # indcator:
    if highlight is not None:
        pg.draw.circle(screen, "red", 
                    (map_offset[0]+highlight.loc[0]*scale,map_offset[1]+highlight.loc[1]*scale), cell_size+2, width=2)
    elif cursor is not None:
        pg.draw.circle(screen, "blue", 
                    (map_offset[0]+cursor[0]*scale,map_offset[1]+cursor[1]*scale), cell_size+2, width=2)

    if cursor is not None:
        line = f'res = {round(world.res[cursor[0],cursor[1]],2)}'
        map_text = font.render(line, True, "black")
        screen.blit(map_text, map_text_loc)

    # logic
    if playing:
        for i in range(step_size):
            world.step()
    elif single_step:
        # stats.update({"repr_out":0,"repr_in":0, "rest_in":0, "rest_out":0})
        world.step()
        single_step=False
        # print(stats)
    # flip() the display to put your work on screen
    pg.display.flip()

    clock.tick(60)  # limits FPS to 60

pg.quit()