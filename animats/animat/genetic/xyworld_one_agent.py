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

# from six.moves.tkinter import *
# from six.moves import tkinter_font
# import tkinter as tk
# from tkinter import font
import inspect
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv

from .genetic_environment import *

DEBUG_MODE = True


def debug(*args):
    if DEBUG_MODE: print('DEBUG:xy_world:', *args)


def drawNetworkGraph():
    debug('drawNetworkGraph')
    return
    g = gv.Digraph(format='svg')

    a = canvas.data.curAgent
    debug('agent name:', a.name)
    nodes_names = a.network.nodes.keys()
    nodes = a.network.allNodes()

    sensors = []
    others = []
    for n in nodes_names:
        if n[0] == '$':
            sensors.append(n[1:])
        else:
            others.append(n)

    with g.subgraph(name='cluster_1') as g1:
        g1.attr(style='filled')
        g1.attr(color='lightgrey')
        g1.node_attr.update(style='filled', color='white')
        g1.attr(label='Sensors')
        # add all sensors
        for n in sensors:
            g1.node(n)
        debug(g1)

    if len(others) > 0:
        with g.subgraph(name='cluster_2') as g2:
            g2.attr(style='filled')
            g2.attr(color='lightgrey')
            g2.node_attr.update(style='filled', color='white')
            g2.attr(label='AND / SEQ ')
            # add other nodes
            for n in others:
                g2.node(n)
            debug(g2)

    with g.subgraph(name='cluster_3') as g3:
        g3.attr(style='filled')
        g3.attr(color='lightgrey')
        g3.node_attr.update(style='filled', color='white')
        g3.attr(label='Motors')
        # add all motors
        for m in a.network.motors:
            g3.node(m.name)
        debug(g3)

    debug('edges')

    # add edges between sensors and other nodes
    for x in nodes:
        debug(x)
        # add in edges
        for i in x.inputs:
            ip = i.name
            op = x.name
            debug('input:', ip)
            debug('output', op)
            if ip[0] == '$':
                ip = ip[1:]

            g.edge(ip, op)

    # add all edges to motors
    actions = list(a.network.actions.values())
    debug(actions)
    #actions = a.network.availableActions()

    for n in nodes:
        acts = [x for x in actions if x.node == n]
        qe = 0
        qw = 0
        ep = a.network.motors[0].name
        wp = a.network.motors[0].name

        for act in acts:
            if act.getQ('energy') >= qe:
                ep = act.motor.name
                qe = act.getQ('energy')
            if act.getQ('water') >= qw:
                wp = act.motor.name
                qw = act.getQ('water')

        ip = n.getName()
        if ip[0] == '$':
            ip = ip[1:]

        if qe > 0:
            g.edge(ip, ep, color='red')
        if qw > 0:
            g.edge(ip, wp, color='blue')

#    g.node('Agent: ' + a.name, shape='Mdiamond')
#    g.node('Age: ' + str(a.age), shape='Msquare')
    g.attr(label='Agent: ' + a.name + ' age: ' + str(a.age))
    debug(g)
    dst = 'img/'+a.name
    #g.render(dst)
    g.view(filename=dst)


def isPaused():
    return canvas.data.pause


def pauseWorld():
    canvas.data.pause = True


def resumeWorld():
    canvas.data.pause = False


def showWorldOverall():
    # display overall info.
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
    # else:
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

    agentInfo = ' '.join('{}:{:10.3f}'.format(k, v) for k, v in canvas.data.curAgent.needs.items())
    agentStr = canvas.data.curAgent.name + ', ' + agentInfo

    col = canvas.data.curAgent.position[0]
    row = canvas.data.curAgent.position[1]
    # show position
    canvas.data.position_str.set("col, %d, row, %d" % (col, row))
    # show status of agents (name, vitality levels)
    canvas.data.name_str.set(agentStr)


def mousePressed(event):
    # str = 'x_root:%d, y_root:%d, x: %d, y:%d' % (event.x_root, event.y_root, event.x, event.y)
    if isPaused():
        resumeWorld()
        plt.close('all')
        return

    pauseWorld()

    x = event.x
    y = event.y
    col = int((x - canvas.data.margin) / canvas.data.cellSize)
    row = int((y - canvas.data.margin) / canvas.data.cellSize)
    print("Position is clicked: col, %d, row, %d" % (col, row))

#    showAgentsInfoWithCurve(col, row)
#    showCurAgentInfo()
    drawNetworkGraph()
    # draw vitality curve
    drawVitalityCurve(canvas.data.curAgent.wellbeeingTrail, canvas.data.tick)


def keyPressed(event):
    # TODO:

    print("key is pressed!")
#    drawNetworkGraph()
    #    canvas.data.pause = not canvas.data.pause
    pass


def timerFired():
#    print("timer fired")
    # display the overall of world
    showWorldOverall()

    if not canvas.data.pause:
        if canvas.data.tick < canvas.data.maxIterations:
            canvas.data.env.step()
            canvas.data.tick += 1
            redrawAll()
            # display the info. of current agent
            showCurAgentInfo()
            # display the bar graph of the vitality level of current agent
            drawVitalityBar()

    canvas.after(canvas.data.delay, timerFired)  # delay, then call timerFired again

def drawVitalityBar():
#    print("draw vitality bar!")
    if canvas.data.curAgent == None:
        return

    energy = canvas.data.curAgent.needs['energy']
    water = canvas.data.curAgent.needs['water']

    margin = canvas.data.margin
    cellSize = canvas.data.cellSize
    barWidth = canvas.data.barWidth
    barScale = canvas.data.barScale

    # draw current Action
    left = margin*3 + len(canvas.data.world[0])*cellSize
    top = margin
    if canvas.data.curAgent.trail != []:
        cell, act = canvas.data.curAgent.trail[-1]
        canvas.create_text(left, top, text="Action -> " + act, anchor=NW, fill='red3')

    # draw indicator text
    left = margin*3 + len(canvas.data.world[0])*cellSize
    top = margin + len(canvas.data.world)*cellSize - barScale - 14
    canvas.create_text(left, top, text='Water         Energy', font=('Helvetica', '6'), anchor=NW)

    # draw frame of bars
    left = margin*3 + len(canvas.data.world[0])*cellSize
    right = left + 2*barWidth
    bottom = margin + len(canvas.data.world)*cellSize
    top = margin + len(canvas.data.world)*cellSize-barScale
    canvas.create_rectangle(left, top, right, bottom, fill='white')
    canvas.create_line(left+barWidth,top,left+barWidth,bottom, fill='black')

    left = margin*3 + len(canvas.data.world[0])*cellSize
    right = left + barWidth
    bottom = margin + len(canvas.data.world)*cellSize
    top = bottom - int(water*barScale)
    # draw water level
    canvas.create_rectangle(left, top, right, bottom, fill='lightblue1')
    # draw energy level
    left = left + barWidth
    right = left + barWidth
    top = bottom - int(energy*barScale)
    canvas.create_rectangle(left, top, right, bottom, fill='tomato')

    # draw vitality values
    water_str = "{:10.3f}".format(water)
    energy_str = "{:10.3f}".format(energy)

    left = margin * 3 + len(canvas.data.world[0]) * cellSize - 8
    bottom = margin + len(canvas.data.world) * cellSize
    top = bottom - 16
    canvas.create_text(left, top, text=water_str, anchor=NW)
    canvas.create_text(left+barWidth, top, text=energy_str, anchor=NW)

def drawVitalityCurve(trail, ticks):
    #    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    #    C, S = np.cos(X), np.sin(X)
    print("draw vitality curves!")

    energyTrail = []
    waterTrail = []

    for d in trail:
        energyTrail.append(d['energy'])
        waterTrail.append(d['water'])

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
#    plt.plot(X, trail)
    plt.xlabel('Tick')
    plt.ylabel('Enegy, water')
    plt.title('Vitality curves of agent')
#    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#    plt.grid(True)
    line_energy, = plt.plot(X, energyTrail, 'r-', label='Energy')
    line_water, = plt.plot(X, waterTrail, 'b-', label='Water')
    plt.legend(handles=[line_energy, line_water])

    plt.show()


def redrawAll():
    canvas.delete(ALL)
    drawWorld()
    drawAgents()

# draw the world of cells with colors
def drawWorldGrid():
    world = canvas.data.world
    rows = len(world)
    cols = len(world[0])

    margin = canvas.data.margin
    cellSize = canvas.data.cellSize

    for row in range(rows):
        for col in range(cols):
            left = margin + col * cellSize
            right = left + cellSize
            top = margin + row * cellSize
            bottom = top + cellSize

            if (world[row][col] == 'c'):
                canvas.create_rectangle(left, top, right, bottom, fill='light sea green')
            elif (world[row][col] == 'b'):
                drawCell(row, col, 'rect', 'goldenrod')
                canvas.create_rectangle(left, top, right, bottom, fill='goldenrod')
            elif (world[row][col] == 'a'):
                drawCell(row, col, 'rect', 'green')
                canvas.create_rectangle(left, top, right, bottom, fill='green')
            else:
                canvas.create_rectangle(left, top, right, bottom, fill='white')

# draw the world of cells with images
def drawWorld():
    world = canvas.data.world
    margin = canvas.data.margin
    cellSize = canvas.data.cellSize

    rows = len(world)
    cols = len(world[0])

    for row in range(rows):
        for col in range(cols):
            left = margin + col * cellSize
            right = left + cellSize
            top = margin + row * cellSize
            bottom = top + cellSize

            if (world[row][col] == 'c'):
                drawBlock(row, col, )
                canvas.create_image(left, top, image=canvas.data.riverImg, anchor=NW)
            elif (world[row][col] == 'b'):
                canvas.create_image(left, top, image=canvas.data.dirtImg, anchor=NW)
            elif (world[row][col] == 'a'):
                canvas.create_image(left, top, image=canvas.data.grassImg, anchor=NW)
            else:
                drawCell(row, col, 'rect', 'white')

def drawBlock(row, col, img=None):
    margin = canvas.data.margin
    cellSize = canvas.data.cellSize

    left = margin + col * cellSize
    right = left + cellSize
    top = margin + row * cellSize
    bottom = top + cellSize
    canvas.create_image(left, top, image=img, anchor=NW)


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
        canvas.create_image(left, top, image=img)


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
            canvas.create_image(left + cellSize / 3, top + cellSize / 3, image=canvas.data.sheepImgB)
        else:
            canvas.create_image(left + cellSize / 3, top + cellSize / 3, image=canvas.data.sheepImg)
    elif kind == 'grass':
        if agent.needs['length'] < 0.3:
            canvas.create_rectangle(left, top + cellSize * 2 / 3, right, bottom, fill='green2')
        elif agent.needs['length'] < 0.7:
            canvas.create_rectangle(left, top + cellSize / 3, right, bottom, fill='green2')
        else:
            canvas.create_rectangle(left, top, right, bottom, fill='green2')
        canvas.create_text(left + cellSize * 2 / 3, top + cellSize * 2 / 3, text='g', fill='green')
    elif kind == 'wolf':
        canvas.create_text(left + cellSize / 2, top + cellSize / 2, text='w', fill='magenta')
    else:
        canvas.create_text(left + cellSize / 2, top + cellSize / 2, text='u', fill='black')


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
    canvas.data.world = [['b', 'c', 'c', 'b', 'c', 'c', 'b', 'c', 'b', 'c'],
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
    print("2 dimension world!")
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
    total_s = Label(overall_window, text="Total agents:")
    total_amount = Label(overall_window, textvariable=canvas.data.total_str)
    neuter_s = Label(overall_window, text="Neuters:")
    neuter_amount = Label(overall_window, textvariable=canvas.data.neuter_str)
    animat_s = Label(overall_window, text="Animats:")
    animat_amount = Label(overall_window, textvariable=canvas.data.animat_str)
    status_s = Label(overall_window, text="Status:")
    status_label = Label(overall_window, textvariable=canvas.data.status_str)
    tick_s = Label(overall_window, text="Current tick:")
    tick_label = Label(overall_window, textvariable=canvas.data.tick_str)

    name_s = Label(overall_window, text="Agent:")
    name_label = Label(overall_window, textvariable=canvas.data.name_str)
    pos_s = Label(overall_window, text="Position:")
    pos_label = Label(overall_window, textvariable=canvas.data.position_str)

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

    margin = 4
    cellsize = 64
    barWidth = 40
    barScale = 160

    root = Tk()
    canvas = Canvas(root, width=margin * 4 + cellsize * len(env.world[0]) + barWidth*2,
                    height=margin * 2 + cellsize * len(env.world))
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
    canvas.data.barWidth = barWidth
    canvas.data.barScale = barScale
    canvas.data.fontItalic = 'italic'  # tkinter_font.Font(family='Helvetica')
    canvas.data.fontBold = 'bold'  # tkinter_font.Font(weight='bold')
    canvas.data.pause = True
    canvas.data.delay = 1500  # milliseconds
    canvas.data.name_str = StringVar(master=root)
    canvas.data.vitality_str = StringVar(master=root)
    canvas.data.position_str = StringVar(master=root)
    canvas.data.total_str = StringVar(master=root)
    canvas.data.neuter_str = StringVar(master=root)
    canvas.data.animat_str = StringVar(master=root)
    canvas.data.status_str = StringVar(master=root)
    canvas.data.tick_str = StringVar(master=root)
    canvas.data.curAgent = env.agents[0]
    canvas.data.sheepImg = PhotoImage(file='./images/sheep_32x32.png')
    canvas.data.sheepImgB = PhotoImage(file='./images/sheep-02-45x30.png')
    canvas.data.dirtImg = PhotoImage(file='./images/dirt_64x64.png')
    canvas.data.riverImg = PhotoImage(file='./images/river_64x64.png')
    canvas.data.grassImg = PhotoImage(file='./images/grass_64x64.png')
    # create a separate window to show the real time statistic of the world
    initOverallWindow()

    # set up events
    root.bind("<Button-1>", mousePressed)
    root.bind("<Key>", keyPressed)

    # draw the UI
    redrawAll()
    #showAgentsInfoWithCurve(0, 0)
    showCurAgentInfo()
    drawVitalityBar()
    timerFired()
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits until you close the window!)
    plt.close('all')
    print("the end!")

