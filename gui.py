import pygame as pg
from simulator import GridWorldSimulator
from intepreter import Intepreter
import numpy as np
from utils import *


screen_size = (1920, 960)
map_offset = (150,100)
map_size = 768
margin = 20
boundary_thickness = 5
file_name = "world.pkl"

def load():
    world = GridWorldSimulator.load(file_name)
    for c in world.creatures.values():
        c.color = color_mapping(c.genome)
    return world


world = load()
scale = map_size//world.size
cell_size = map_size//world.size//2

# pygame setup
pg.init()
screen = pg.display.set_mode(screen_size)
clock = pg.time.Clock()
running = True

pg.font.init()
font = pg.font.Font('freesansbold.ttf', 32)
play_button_rect = (600,40,120,32)
reload_button_rect = (200,40,120,32)

step_text_loc = (800,40)
info_text_loc = (1000,60)

step=0
playing = False
highlight = None
def init():
    global step, playing, highlight
    step=0
    playing = False
    highlight = None

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_SPACE:
                playing=not playing
            if highlight is not None:
                if event.key == pg.K_RIGHT or event.key == pg.K_d:
                    loc = [highlight.loc[0]+1, highlight.loc[1]]
                    while loc[0]<world.size:
                        c = world.get_creature_at(loc)
                        if c:
                            highlight = c
                            break
                        loc[0]+=1
                elif event.key == pg.K_LEFT or event.key == pg.K_a:
                    loc = [highlight.loc[0]-1, highlight.loc[1]]
                    while loc[0]>0:
                        c = world.get_creature_at(loc)
                        if c:
                            highlight = c
                            break
                        loc[0]-=1
                elif event.key == pg.K_UP or event.key == pg.K_w:
                    loc = [highlight.loc[0], highlight.loc[1]-1]
                    while loc[1]>0:
                        c = world.get_creature_at(loc)
                        if c:
                            highlight = c
                            break
                        loc[1]-=1
                elif event.key == pg.K_DOWN or event.key == pg.K_s:
                    loc = [highlight.loc[0], highlight.loc[1]+1]
                    while loc[1]<world.size:
                        c = world.get_creature_at(loc)
                        if c:
                            highlight = c
                            break
                        loc[1]+=1
        if event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1:
                if pg.Rect(*play_button_rect).collidepoint(event.pos):
                    playing=not playing
                if pg.Rect(*reload_button_rect).collidepoint(event.pos):
                    world = load()
                    init()


                elif pg.Rect(*map_offset,map_size,map_size).collidepoint(event.pos):
                    for c in world.creatures.values():
                        if -2<c.loc[0]-(event.pos[0]-map_offset[0])//scale<2 and -2<c.loc[1]-(event.pos[1]-map_offset[1])//scale<2:
                            highlight=c
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

    pg.draw.rect(screen, "purple", reload_button_rect)
    reload_text = font.render("Reload", True, "white")
    screen.blit(reload_text, reload_button_rect[:2])

    step_text = font.render(f'Step: {step}', True, "green", "blue")
    screen.blit(step_text, step_text_loc)

    if highlight is not None:
        infos = [
            f'id: {world.get_cid(highlight)}',
            f'loc: {(highlight.loc[0],highlight.loc[1])}',
            f'inputs: {[Intepreter.InputNodes[i] for i,v in enumerate(highlight.reflex.enabled_inputs) if v]}',
            f'outputs: {[Intepreter.OutputNodes[i] for i,v in enumerate(highlight.reflex.enabled_outputs) if v]}',

        ]
        for i, line in enumerate(infos):
            info_text = font.render(line, True, "black")
            screen.blit(info_text, (info_text_loc[0],info_text_loc[1]+i*32))
    

    # map area
    pg.draw.rect(screen, "black", 
                     (map_offset[0]-margin,map_offset[1]-margin,map_size+margin*2,map_size+margin*2), width=boundary_thickness)

    for c in world.creatures.values():
        if highlight is not None and world.get_cid(highlight)==world.get_cid(c):
            pg.draw.circle(screen, "red", 
                           (map_offset[0]+c.loc[0]*scale,map_offset[1]+c.loc[1]*scale), cell_size+2, width=2)
        pg.draw.circle(screen, c.color, 
                           (map_offset[0]+c.loc[0]*scale,map_offset[1]+c.loc[1]*scale), cell_size)

    # logic
    if playing:
        world.step()
        step+=1
    # flip() the display to put your work on screen
    pg.display.flip()

    clock.tick(60)  # limits FPS to 60

pg.quit()