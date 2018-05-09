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
from scipy import misc

import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv



def debug(*args):
    print('DEBUG:xy_world_cellular_automata:', *args)



def isPaused():
    return canvas.data.pause


def pauseWorld():
    canvas.data.pause = True


def resumeWorld():
    canvas.data.pause = False


def showWorldOverall():
    # display overall info.
    r = canvas.data.ramount
    g = canvas.data.gamount
    b = canvas.data.bamount
    # update the real time window
    canvas.data.total_str.set('%d' % (r+g+b))
    canvas.data.paper_str.set('%d' % r)
    canvas.data.rock_str.set('%d' % g)
    canvas.data.scissor_str.set('%d' % b)
    canvas.data.tick_str.set('Current tick: %d' % canvas.data.tick)
    if isPaused():
        canvas.data.status_str.set("paused")
    else:
        canvas.data.status_str.set("running")


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
    plotCurve(canvas.data.trail, canvas.data.tick)



def keyPressed(event):
    # TODO:
    print("key is pressed!")
    #    canvas.data.pause = not canvas.data.pause
    pass


def timerFired():
    print("timer fired")

    if not canvas.data.pause:
        if canvas.data.tick < canvas.data.maxIterations:
            # change the index
            # prepare the next world
            redrawAll()
            # update and save the statistics
            r,g,b = cellStatistics()
            canvas.data.ramount = r
            canvas.data.gamount = g
            canvas.data.bamount = b
            canvas.data.trail.append((r,g,b))
            updateWorld()
            switchIndex()
            # update the next world
            canvas.data.tick += 1

    canvas.after(canvas.data.delay, timerFired)  # delay, then call timerFired again


def switchIndex():
    canvas.data.current = (canvas.data.current + 1) % 2

# find adjacent cells
def adjacent(row, col):
    world = canvas.data.world
    rows = canvas.data.rows
    cols = canvas.data.cols
    current = canvas.data.current

    items = []

    # row - 1
    if row > 0:
        items.append(world[current][row-1][col])
    # row - 1
    if row < rows-1:
        items.append(world[current][row+1][col])

    # col - 1
    if col > 0:
        items.append(world[current][row][col-1])
    # col + 1
    if col < cols-1:
        items.append(world[current][row][col+1])

    return items


# update the next world of cells
def updateWorld():
    world = canvas.data.world
    current = canvas.data.current
    next  = (current + 1)%2
    rows = canvas.data.rows
    cols = canvas.data.cols

    for row in range(rows):
        for col in range(cols):
            # find out adjacent cells
            adjacents = set(adjacent(row, col))
            if (world[current][row][col] == 'r'):
                # paper -> scissor
                if 'b' in adjacents:
                    world[next][row][col] = 'b'
                else:
                    world[next][row][col] = 'r'
            elif (world[current][row][col] == 'g'):
                # rock -> paper
                if 'r' in adjacents:
                    world[next][row][col] = 'r'
                else:
                    world[next][row][col] = 'g'
            elif (world[current][row][col] == 'b'):
                # scissor
                if 'g' in adjacents:
                    world[next][row][col] = 'g'
                else:
                    world[next][row][col] = 'b'
            else:
                pass


def redrawAll():
    canvas.delete(ALL)
    showWorldOverall()
    drawWorld()


# draw the world of cells with colors
def drawWorld():
    world = canvas.data.world
    current = canvas.data.current
    margin = canvas.data.margin
    cellSize = canvas.data.cellSize
    rows = canvas.data.rows
    cols = canvas.data.cols

    for row in range(rows):
        for col in range(cols):
            left = margin + col * cellSize
            right = left + cellSize
            top = margin + row * cellSize
            bottom = top + cellSize

            if (world[current][row][col] == 'r'):
                # paper
                canvas.create_rectangle(left, top, right, bottom, fill='red')
            elif (world[current][row][col] == 'g'):
                # rock
                canvas.create_rectangle(left, top, right, bottom, fill='green')
            elif (world[current][row][col] == 'b'):
                # scissor
                canvas.create_rectangle(left, top, right, bottom, fill='blue')
            else:
                canvas.create_rectangle(left, top, right, bottom, fill='white')


def plotCurve(trail, ticks):
    #    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    #    C, S = np.cos(X), np.sin(X)
    print("draw vitality curves!")

    paperTrail = [x for (x,y,z) in trail]
    rockTrail = [y for (x,y,z) in trail]
    scissorTrail = [z for (x,y,z) in trail]

    # getting the average of vitality levels
    paperMean = np.mean(paperTrail)
    rockMean = np.mean(rockTrail)
    scissorMean = np.mean(scissorTrail)


    X = np.arange(0, ticks, 1)


    fig = plt.figure()
#    ax0 = fig.add_subplot(211)
#    plt.xlabel('Tick')
#    plt.ylabel('Total amount of animats')
#    amountTrail = canvas.data.agentsNumberTrail
#    plt.title('Average amount of alive animats: %d' % np.mean(amountTrail))
#    ax0.plot(np.arange(0, ticks, 1), amountTrail, 'g-', label='Amount')


    ax1 = fig.add_subplot(111)
    plt.xlabel('Tick')
    plt.ylabel('Rock, paper, scissor')
    plt.title('Vitality curves of cells')


#    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#    plt.grid(True)
    line_paper, = ax1.plot(X, paperTrail, 'r-', label='Paper')
    line_rock, = ax1.plot(X, rockTrail, 'g-', label='Rock')
    line_scissor, = ax1.plot(X, scissorTrail, 'b-', label='Scissor')
#    ax1.annotate('Mean(Paper): %.3f, Mean(Rock): %.3f, Mean(Scissor): %.3f' % (paperMean, rockMean, scissorMean),
#                xy=(0, 0), xytext=(90, 5),
#                xycoords=('axes fraction', 'figure fraction'),
#                textcoords='offset points',
#                size=8, ha='center', va='bottom')

    plt.legend(handles=[line_paper, line_rock, line_scissor])

    plt.show()


def loadWorld(rows, cols):
#    global world
#    global index
    # 'r': paper - red
    # 'g': rock - green
    # 'b': scissor - blue
    items = ['r', 'g', 'b']

    image = misc.imread('taiji.bmp')



    world = [[], []]

#    world[0] = world[1] = [['b', 'c', 'c', 'b', 'c', 'c', 'b', 'c', 'b', 'c'],
#                         ['b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'b', 'b'],
#                         ['c', 'c', 'b', 'c', 'c', 'b', 'b', 'b', 'c', 'b'],
#                         ['c', 'b', 'c', 'b', 'c', 'b', 'c', 'b', 'b', 'b'],
#                         ['b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'b', 'b'],
#                         ['b', 'b', 'c', 'c', 'c', 'c', 'b', 'b', 'b', 'b'],
#                         ['b', 'c', 'b', 'c', 'b', 'c', 'b', 'c', 'b', 'c'],
#                         ['b', 'b', 'c', 'c', 'b', 'b', 'c', 'c', 'b', 'b'],
#                         ['b', 'c', 'c', 'c', 'b', 'b', 'b', 'c', 'b', 'c'],
#                         ['b', 'b', 'c', 'c', 'b', 'b', 'c', 'c', 'b', 'b']
#                         ]

    index = [0,1,2]


    for r in range(rows):
        world[0].append([])
        world[1].append([])

        for c in range(cols):
            # v = np.random.choice(items)
            #v = np.random.choice(items, p=[0.1, 0.45, 0.45])
#            if r < 77:
#                if c < 50:
#                    v = items[2]
#                else:
#                    v = items[1]
#            else:
#                if c < 50:
#                    v = items[1]
#                else:
#                    v = items[2]

#            if image[r][c] == 0:
#                v = items[0]

            if r < 50:
                v = items[1]
            else:
                v = items[2]

            if image[r][c] == 0:
                v = items[0]

            world[0][r].append(v)
            world[1][r].append(v)

    # randomly generate worlds of cells
#    i = np.random.choice(index, p=[0.3, 0.4, 0.3])
#    for r in range(rows):
#        world[0].append([])
#        world[1].append([])

#        for c in range(cols):
            #v = np.random.choice(items)
            #v = np.random.choice(items, p=[0.1, 0.3, 0.6])
#            v = items[i]
#            if c%(3+(i%3)) == 1:
#                i = (i+1)%3

#            world[0][r].append(v)
#            world[1][r].append(v)

#    for r in range(0, rows, 5):
#        world[0].append([])
#        world[0].append([])
#        world[0].append([])
#        world[0].append([])
#        world[0].append([])
#        world[1].append([])
#        world[1].append([])
#        world[1].append([])
#        world[1].append([])
#        world[1].append([])

#        for c in range(0, cols, 5):
            #v = np.random.choice(items)
            #v = np.random.choice(items, p=[0.1, 0.3, 0.6])
#            i1 = i%3
#            i2 = (i+1)%3
#            i3 = (i+2)%3
#            i = i+1

            # row 0
#            world[0][r].append(items[i1])
#            world[1][r].append(items[i1])
#            world[0][r].append(items[i1])
#            world[1][r].append(items[i1])

#            world[0][r].append(items[i2])
#            world[1][r].append(items[i2])

#            world[0][r].append(items[i3])
#            world[1][r].append(items[i3])
#            world[0][r].append(items[i3])
#            world[1][r].append(items[i3])

            # row 1
#            world[0][r+1].append(items[i1])
#            world[1][r+1].append(items[i1])
#            world[0][r+1].append(items[i2])
#            world[1][r+1].append(items[i2])

#            world[0][r+1].append(items[i3])
#            world[1][r+1].append(items[i3])

#            world[0][r+1].append(items[i2])
#            world[1][r+1].append(items[i2])
#            world[0][r+1].append(items[i3])
#            world[1][r+1].append(items[i3])

            # row 2
#            world[0][r+2].append(items[i2])
#            world[1][r+2].append(items[i2])

#            world[0][r+2].append(items[i2])
#            world[1][r+2].append(items[i2])
#            world[0][r+2].append(items[i3])
#            world[1][r+2].append(items[i3])
#            world[0][r+2].append(items[i2])
#            world[1][r+2].append(items[i2])

#            world[0][r+2].append(items[i2])
#            world[1][r+2].append(items[i2])

            # row 3
#            world[0][r+3].append(items[i3])
#            world[1][r+3].append(items[i3])
#            world[0][r+3].append(items[i2])
#            world[1][r+3].append(items[i2])

#            world[0][r+3].append(items[i3])
#            world[1][r+3].append(items[i3])

#            world[0][r+3].append(items[i2])
#            world[1][r+3].append(items[i2])
#            world[0][r+3].append(items[i1])
#            world[1][r+3].append(items[i1])

            # row 4
#            world[0][r+4].append(items[i3])
#            world[1][r+4].append(items[i3])
#            world[0][r+4].append(items[i3])
#            world[1][r+4].append(items[i3])

#            world[0][r+4].append(items[i2])
#            world[1][r+4].append(items[i2])

#            world[0][r+4].append(items[i1])
#            world[1][r+4].append(items[i1])
#            world[0][r+4].append(items[i1])
#            world[1][r+4].append(items[i1])



    return world

def printInstructions():
    print("2 dimension world!")
    print("Click on the cell in the block world to pause/resume the simulation of the world!")


def cellStatistics():
    world = canvas.data.world
    current = canvas.data.current
    rows = canvas.data.rows
    cols = canvas.data.cols

    # r - paper
    # g - rock
    # b - scissor
    ramount = 0
    gamount = 0
    bamount = 0

    for r in range(rows):
        ramount += len([x for x in world[current][r] if x == 'r'])
        gamount += len([x for x in world[current][r] if x == 'g'])
        bamount += len([x for x in world[current][r] if x == 'b'])

    return ramount, gamount, bamount


def debugInfo():
    world = canvas.data.world
    current = canvas.data.current
    print('Current tick: %d' % canvas.data.tick)
    r, g, b = cellStatistics()
    print('Paper: %d' % r)
    print('Rock: %d' % g)
    print('Scissor: %d' % b)


def initOverallWindow():
    overall_window = Toplevel()
    total_s = Label(overall_window, text="Total cells:")
    total_amount = Label(overall_window, textvariable=canvas.data.total_str)
    paper_s = Label(overall_window, text="Paper:")
    paper_amount = Label(overall_window, textvariable=canvas.data.paper_str)
    rock_s = Label(overall_window, text="Rocks:")
    rock_amount = Label(overall_window, textvariable=canvas.data.rock_str)
    scissor_s = Label(overall_window, text="Scissors:")
    scissor_amount = Label(overall_window, textvariable=canvas.data.scissor_str)
    status_s = Label(overall_window, text="Status:")
    status_label = Label(overall_window, textvariable=canvas.data.status_str)
    tick_s = Label(overall_window, text="Current tick:")
    tick_label = Label(overall_window, textvariable=canvas.data.tick_str)


    total_s.grid(row=0, column=0, sticky=W)
    paper_s.grid(row=1, column=0, sticky=W)
    rock_s.grid(row=2, column=0, sticky=W)
    scissor_s.grid(row=3, column=0, sticky=W)
    status_s.grid(row=4, column=0, sticky=W)
    tick_s.grid(row=5, column=0, sticky=W)

    total_amount.grid(row=0, column=1, sticky=W)
    paper_amount.grid(row=1, column=1, sticky=W)
    rock_amount.grid(row=2, column=1, sticky=W)
    scissor_amount.grid(row=3, column=1, sticky=W)
    status_label.grid(row=4, column=1, sticky=W)
    tick_label.grid(row=5, column=1, sticky=W)


########### copy-paste below here ###########

def run2DWorld(margin, cellsize, rows, cols, maxIterations=1000):
    # create the root and the canvas
    global canvas


    root = Tk()
    canvas = Canvas(root, width=margin * 2 + cellsize * cols,
                    height=margin * 2 + cellsize * rows)
    canvas.pack()

    # Store canvas in root and in canvas itself for callbacks
    root.canvas = canvas.canvas = canvas

    # create global variables
    class Struct: pass

    canvas.data = Struct()
    canvas.data.world = loadWorld(rows, cols)
    canvas.data.rows = rows
    canvas.data.cols = cols
    canvas.data.current = 0
    canvas.data.maxIterations = maxIterations
    canvas.data.tick = 0
    canvas.data.margin = margin
    canvas.data.cellSize = cellsize
    canvas.data.fontItalic = 'italic'  # tkinter_font.Font(family='Helvetica')
    canvas.data.fontBold = 'bold'  # tkinter_font.Font(weight='bold')
    canvas.data.pause = True
    canvas.data.delay = 200  # milliseconds
    canvas.data.total_str = StringVar(master=root)
    canvas.data.status_str = StringVar(master=root)
    canvas.data.tick_str = StringVar(master=root)
    canvas.data.paper_str = StringVar(master=root)
    canvas.data.rock_str = StringVar(master=root)
    canvas.data.scissor_str = StringVar(master=root)
    canvas.data.trail = []
    r, g, b = cellStatistics()
    canvas.data.ramount = r
    canvas.data.gamount = g
    canvas.data.bamount = b


    # create a separate window to show the real time statistic of the world
    initOverallWindow()

    # set up events
    root.bind("<Button-1>", mousePressed)
#    root.bind("<Key>", keyPressed)
#    root.title(canvas.data.tick_str)
    root.title("Cellular automata")
    redrawAll()
    timerFired()
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits until you close the window!)
    plt.close('all')
    debugInfo()
    print("the end!")



# main program
run2DWorld(2, 8, 100, 100, 1000)