# -*- coding: utf-8 -*-
# xyworld.py
#
#    pyAnimat - Simulate Animats using Transparent Graphs and genetic algo.
#               as a way to AGI
#    Copyright (C) 2017  Wen Xu, Claes Strannegård
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    draw 2-dimension world
  
#from six.moves.tkinter import *
#from six.moves import tkinter_font
#import tkinter as tk
#from tkinter import font
import inspect
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv

from .genetic_environment import *


def debug(*args):
    print('DEBUG:xy_world:', *args)


def drawNetworkGraph():
    g2 = gv.Digraph(format='svg')
    g2.node('A')
    g2.node('B')
    g2.edge('A', 'B')
    g2.render('img/g2')


def isPaused():
    return canvas.data.pause


def pauseWorld():
    canvas.data.pause = True


def resumeWorld():
    canvas.data.pause = False


def showWorldOverall():
    canvas.data.total_str.set(str(len(canvas.data.env.agents)))
    canvas.data.neuter_str.set(str(len([x for x in canvas.data.env.agents if x.is_neuter()])))
    canvas.data.animat_str.set(str(len([x for x in canvas.data.env.agents if not x.is_neuter()])))
    canvas.data.tick_str.set(str(canvas.data.tick))

    if isPaused():
        canvas.data.status_str.set("paused")
    else:
        canvas.data.status_str.set("running")


def showAgentsInfoWithCurve(col, row):

    # show position anyway
    canvas.data.position_str.set("x: %d, y: %d" % (col, row))

    # check if there will be agent(s)
    agents = canvas.data.env.list_agents_at((col, row), GeneticAgent)
    if agents == []:
        canvas.data.name_str.set("None")
        return

    agentsName = [x.name for x in agents]
    debug(agentsName)
    agentsInfo = [' '.join('{}:{}'.format(k, v) for k, v in x.needs.items()) for x in agents]
    debug(agentsInfo)
    agentsStr = [(x + ': ' + y) for (x, y) in zip(agentsName, agentsInfo)]

    # show status of agents (name, vitality levels)
    canvas.data.name_str.set("\n".join(agentsStr))

    # draw vitality curve(s)
    animals = [x for x in agents if isinstance(x, GeneticAnimalAgent)]
    # keep tracking the current animal
    if animals == []:
        return
#    else:
#        canvas.data.curAgent = animals[0]
    # if the world is running, skip drawing the vitality curve of the animal
    if not isPaused():
        return
    # draw the vitality curve of the animal when the world is paused
    for a in animals:
        if len(a.wellbeeingTrail) > 0:
            drawVitalityCurve(a.wellbeeingTrail, canvas.data.tick)


# TODO: for tracking the current agent at real time.
def showCurAgentInfo():
    # show the current agent
    if canvas.data.curAgent == None:
        return

    agentInfo = ' '.join('{}:{}'.format(k, v) for k, v in canvas.data.curAgent.needs.items())
    agentStr = canvas.data.curAgent.name + ', ' + agentInfo

    col = canvas.data.curAgent.position[0]
    row = canvas.data.curAgent.position[1]
    # show position
    canvas.data.position_str.set("x, %d, y, %d" % (col, row))
    # show status of agents (name, vitality levels)
    canvas.data.name_str.set("%s" % agentStr)


def mousePressed(event):
    #str = 'x_root:%d, y_root:%d, x: %d, y:%d' % (event.x_root, event.y_root, event.x, event.y)
    if isPaused():
        resumeWorld()
        plt.close('all')
        return

    pauseWorld()

    x = event.x
    y = event.y
    col = int((x - canvas.data.margin)/canvas.data.cellSize)
    row = int((y - canvas.data.margin)/canvas.data.cellSize)
    print("Position is clicked: x, %d, y, %d" % (col, row))

    showAgentsInfoWithCurve(col, row)


def keyPressed(event):
    # TODO:
    drawNetworkGraph()
    print("key is pressed!")
#    canvas.data.pause = not canvas.data.pause
    pass


def timerFired():
    print ("timer fired")
    showWorldOverall()

    if not canvas.data.pause:
        if canvas.data.tick < canvas.data.maxIterations:
            canvas.data.env.step()
            canvas.data.tick += 1
            redrawAll()

    canvas.after(canvas.data.delay, timerFired) # delay, then call timerFired again


def drawVitalityCurve(trail, ticks):
#    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
#    C, S = np.cos(X), np.sin(X)

    # if the animal was not born at the beginning
    if len(trail) < ticks:
        start = ticks - len(trail)
    else:
        start = 0
    debug(len(trail))
    debug(ticks)
    debug(start)
    X = np.arange(start, ticks, 1)

    # plot the vitality
    plt.plot(X, trail)
    plt.show()


def redrawAll():
    canvas.delete(ALL)
    drawWorld()
    drawAgents()


def drawWorld():
    world = canvas.data.world
    rows = len(world)
    cols = len(world[0])
    for row in range(rows):
        for col in range(cols):
            if (world[row][col] == 'c'):
                drawCell(row, col, 'rect', 'light sea green')
            elif (world[row][col] == 'b'):
                drawCell(row, col, 'rect', 'goldenrod')
            elif (world[row][col] == 'a'):
                drawCell(row, col, 'rect', 'green')
            else:
                drawCell(row, col, 'rect', 'white')


def drawCell(row, col, shape, color, img=None):
    margin = canvas.data.margin
    cellSize = canvas.data.cellSize

    left = margin + col * cellSize
    right = left + cellSize
    top = margin + row * cellSize
    bottom = top + cellSize
    if shape == 'oval':
        canvas.create_oval(left, top, right, bottom, fill=color)
    elif shape == 'rect':
        canvas.create_rectangle(left, top, right, bottom, fill=color)
    else:
        # TODO: image for agent
        canvas.create_image(left,top, image=img)


# row: position[1] or y, col: position[0] or x
def drawAgent(agent, row, col, kind):
    margin = canvas.data.margin
    cellSize = canvas.data.cellSize

    left = margin + col * cellSize
    right = left + cellSize
    top = margin + row * cellSize
    bottom = top + cellSize
    if kind == 'sheep':
#        canvas.create_text(left + cellSize/3, top + cellSize/3, text='S', fill='red', font=ft)
#        canvas.create_image(left + cellSize / 3, top + cellSize / 3, image=canvas.data.sheepImg)
        if agent.is_pregnant():
            canvas.create_image(left+cellSize/2, top+cellSize/2, image=canvas.data.sheepImgB)
        else:
            canvas.create_image(left+cellSize/2, top+cellSize/2, image=canvas.data.sheepImg)
    elif kind == 'grass':
        if agent.needs['length'] < 0.3:
            canvas.create_rectangle(left, top + cellSize*2/3, right, bottom, fill='green2')
        elif agent.needs['length'] < 0.7:
            canvas.create_rectangle(left, top + cellSize/3, right, bottom, fill='green2')
        else:
            canvas.create_rectangle(left, top, right, bottom, fill='green2')
        canvas.create_text(left + cellSize*2/3, top + cellSize*2/3, text='g', fill='green')
    elif kind == 'wolf':
        canvas.create_text(left + cellSize/2, top + cellSize/2, text='w', fill='magenta')
    else:
        canvas.create_text(left + cellSize/2, top + cellSize/2, text='u', fill='black')


def drawAgents():
    for a in canvas.data.env.agents:
        if isinstance(a, GeneticGrassAgent):
            kind = 'grass'
        elif isinstance(a, GeneticSheepAgent):
            kind = 'sheep'
        else:
            kind = 'wolf'

        drawAgent(a, a.position[1], a.position[0], kind)


def loadWorld():
    canvas.data.world = [ ['b', 'c', 'c', 'b', 'c', 'c', 'b', 'c', 'b', 'c'],
                          ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'b', 'b'],
                          ['c', 'c', 'b', 'c', 'c', 'b', 'b', 'b', 'c', 'b'],
                          ['c', 'b', 'c', 'b', 'c', 'b', 'c', 'b', 'b', 'b'],
                          ['b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'b', 'b'],
                          ['b', 'b', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b'],
                          ['b', 'c', 'b', 'c', 'b', 'c', 'b', 'c', 'b', 'c'],
                          ['b', 'b', 'c', 'c', 'b', 'b', 'c', 'c', 'b', 'b'],
                          ['b', 'c', 'c', 'c', 'b', 'b', 'b', 'c', 'b', 'c'],
                          ['b', 'b', 'c', 'c', 'b', 'b', 'c', 'c', 'b', 'b']
                        ]


def printInstructions():
    print ("2 dimension world!")
    print("Click on the cell in the block world to pause/resume the simulation of the world!")


def init():
    printInstructions()
    loadWorld()
    redrawAll()


def insModule(module):
    funcs = inspect.getmembers(module)
    for n, _ in funcs:
        debug('function: ', n)


def initOverallWindow():
    overall_window = Toplevel()
    total_s       = Label(overall_window, text="Total agents:")
    total_amount  = Label(overall_window, textvariable=canvas.data.total_str)
    neuter_s      = Label(overall_window, text="Neuters:")
    neuter_amount = Label(overall_window, textvariable=canvas.data.neuter_str)
    animat_s      = Label(overall_window, text="Animats:")
    animat_amount = Label(overall_window, textvariable=canvas.data.animat_str)
    status_s      = Label(overall_window, text="Status:")
    status_label  = Label(overall_window, textvariable=canvas.data.status_str)
    tick_s        = Label(overall_window, text="Current tick:")
    tick_label    = Label(overall_window, textvariable=canvas.data.tick_str)

    name_s     = Label(overall_window, text="Agent:")
    name_label = Label(overall_window, textvariable=canvas.data.name_str)
    pos_s = Label(overall_window, text="Position:")
    pos_label  = Label(overall_window, textvariable=canvas.data.position_str)

    total_s.grid(row=0, column=0, sticky=W)
    neuter_s.grid(row=1, column=0, sticky=W)
    animat_s.grid(row=2, column=0, sticky=W)
    status_s.grid(row=3, column=0, sticky=W)
    tick_s.grid(row=4, column=0, sticky=W)

    total_amount.grid(row=0, column=1, sticky=W)
    neuter_amount.grid(row=1, column=1, sticky=W)
    animat_amount.grid(row=2, column=1, sticky=W)
    status_label.grid(row=3, column=1, sticky=W)
    tick_label.grid(row=4, column=1, sticky=W)

    name_s.grid(row=0, column=2, rowspan=3, sticky=W)
    pos_s.grid(row=4, column=2, sticky=W)

    name_label.grid(row=0, column=3, rowspan=3, sticky=W)
    pos_label.grid(row=4, column=3, sticky=W)


########### copy-paste below here ###########

def run2DWorld(env, maxIterations=1000):
    # create the root and the canvas
    global canvas
    global infoCanvas
    global margin
    global cellsize

    margin = 5
    cellsize = 50

    root = Tk()
    canvas = Canvas(root, width=margin*2+cellsize*len(env.world[0]), height=margin*2+cellsize*len(env.world))
    canvas.pack()

    # Store canvas in root and in canvas itself for callbacks
    root.canvas = canvas.canvas = canvas
    # create global variables
    class Struct: pass
    canvas.data = Struct()

    canvas.data.world = env.world
    canvas.data.env = env
    canvas.data.maxIterations = maxIterations
    canvas.data.tick = 0
    canvas.data.margin = margin
    canvas.data.cellSize = cellsize
    canvas.data.fontItalic = 'italic' #tkinter_font.Font(family='Helvetica')
    canvas.data.fontBold = 'bold' #tkinter_font.Font(weight='bold')
    canvas.data.pause = False
    canvas.data.delay = 2000 # milliseconds
    canvas.data.name_str = StringVar(master=root)
    canvas.data.vitality_str = StringVar(master=root)
    canvas.data.position_str = StringVar(master=root)
    canvas.data.total_str = StringVar(master=root)
    canvas.data.neuter_str = StringVar(master=root)
    canvas.data.animat_str = StringVar(master=root)
    canvas.data.status_str = StringVar(master=root)
    canvas.data.tick_str = StringVar(master=root)
    canvas.data.curAgent = None
    canvas.data.sheepImg = PhotoImage(file='./images/sheep-02-30x20.png')
    canvas.data.sheepImgB = PhotoImage(file='./images/sheep-02-45x30.png')
    # create a separate window to show the real time statistic of the world
    initOverallWindow()

    # set up events
    root.bind("<Button-1>", mousePressed)
#    root.bind("<Key>", keyPressed)

    # draw the UI
    redrawAll()
    showAgentsInfoWithCurve(0,0)
    timerFired()
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits until you close the window!)
    plt.close('all')
    print("the end!")

