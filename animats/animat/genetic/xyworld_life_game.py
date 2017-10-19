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
#    game of lift in 2-dimension world

# from six.moves.tkinter import *
# from six.moves import tkinter_font
# import tkinter as tk
# from tkinter import font
import inspect
from tkinter import *

import matplotlib.pyplot as plt
import numpy as np
import graphviz as gv



def debug(*args):
    print('DEBUG:xy_world_game_of_life:', *args)



def isPaused():
    return canvas.data.pause


def pauseWorld():
    canvas.data.pause = True


def resumeWorld():
    canvas.data.pause = False


def showWorldOverall():
    # display overall info.
    r = canvas.data.ramount
    # update the real time window
    canvas.data.total_str.set('%d' % (canvas.data.cols*canvas.data.rows))
    canvas.data.live_str.set('%d' % r)
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
            r = cellStatistics()
            canvas.data.ramount = r
            canvas.data.trail.append(r)
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
        if col > 0:
            items.append(world[current][row-1][col-1])
        if col < cols-1:
            items.append(world[current][row-1][col+1])
    # row + 1
    if row < rows-1:
        items.append(world[current][row+1][col])
        if col > 0:
            items.append(world[current][row+1][col-1])
        if col < cols-1:
            items.append(world[current][row+1][col+1])

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
            adjacents = adjacent(row, col)
            lives = [x for x in adjacents if x == 'r']
            l = len(lives)
            if (world[current][row][col] == 'r'):
                # calculate number of lives
                if l < 2 or l > 3:
                    # live cell is about to die
                    world[next][row][col] = 'w'
                else:
                    world[next][row][col] = 'r'

            elif (world[current][row][col] == 'w'):
                # dead to be born
                if l == 3:
                    world[next][row][col] = 'r'
                else:
                    world[next][row][col] = 'w'
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
                # life
                canvas.create_rectangle(left, top, right, bottom, fill='red2')
            elif (world[current][row][col] == 'w'):
                # dead
                canvas.create_rectangle(left, top, right, bottom, fill='white')
            else:
                canvas.create_rectangle(left, top, right, bottom, fill='black')


def plotCurve(trail, ticks):
    #    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    #    C, S = np.cos(X), np.sin(X)
    print("draw vitality curves!")

    liveTrail = trail
    # getting the average of vitality levels
    liveMean = np.mean(liveTrail)

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
    plt.ylabel('Live cells')
    plt.title('Amount curve of cells')


#    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
#    plt.grid(True)
    line_live, = ax1.plot(X, liveTrail, 'r-', label='Live cells')
    ax1.annotate('Mean(Live cells): %.3f' % liveMean,
                xy=(0, 0), xytext=(90, 5),
                xycoords=('axes fraction', 'figure fraction'),
                textcoords='offset points',
                size=8, ha='center', va='bottom')

    plt.legend(handles=[line_live])

    plt.show()


def loadWorld(rows, cols):
#    global world
#    global index
    # 'r': paper - red
    # 'g': rock - green
    # 'b': scissor - blue
    items = ['r', 'w']

    world = [[], []]

#    world[0] = [['r', 'w', 'w', 'w', 'w', 'w', 'r', 'w', 'r', 'w'],
#               ['r', 'w', 'w', 'r', 'w', 'w', 'w', 'r', 'w', 'r'],
#               ['r', 'w', 'w', 'w', 'r', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'w', 'w', 'w', 'w', 'w', 'r', 'w', 'w', 'w'],
#               ['r', 'w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'w', 'r', 'w', 'w', 'r', 'w', 'r', 'w', 'w'],
#               ['w', 'w', 'r', 'w', 'r', 'r', 'w', 'w', 'r', 'w'],
#               ['w', 'r', 'r', 'w', 'w', 'w', 'r', 'r', 'w', 'r'],
#               ['w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']]
#    world[1] = [['r', 'w', 'w', 'w', 'w', 'w', 'r', 'w', 'r', 'w'],
#               ['r', 'w', 'w', 'r', 'w', 'w', 'w', 'r', 'w', 'r'],
#               ['r', 'w', 'w', 'w', 'r', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'w', 'w', 'w', 'w', 'w', 'r', 'w', 'w', 'w'],
#               ['r', 'w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'w', 'r', 'w', 'w', 'r', 'w', 'r', 'w', 'w'],
#               ['w', 'w', 'r', 'w', 'r', 'r', 'w', 'w', 'r', 'w'],
#               ['w', 'r', 'r', 'w', 'w', 'w', 'r', 'r', 'w', 'r'],
#               ['w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w'],
#               ['w', 'r', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']]

    # randomly generate worlds of cells
    for r in range(rows):
        world[0].append([])
        world[1].append([])
        for c in range(cols):
            #v = np.random.choice(items)
            v = np.random.choice(items, p=[0.5, 0.5])
            world[0][r].append(v)
            world[1][r].append(v)

    return world

def printInstructions():
    print("2 dimension world!")
    print("Click on the cell in the block world to pause/resume the simulation of the world!")


def cellStatistics():
    world = canvas.data.world
    current = canvas.data.current
    rows = canvas.data.rows
    cols = canvas.data.cols

    # r - life
    ramount = 0

    for r in range(rows):
        ramount += len([x for x in world[current][r] if x == 'r'])

    return ramount


def debugInfo():
    world = canvas.data.world
    current = canvas.data.current
    print('Current tick: %d' % canvas.data.tick)
    r = cellStatistics()
    print('Live cells: %d' % r)


def initOverallWindow():
    overall_window = Toplevel()
    total_s = Label(overall_window, text="Total cells:")
    total_amount = Label(overall_window, textvariable=canvas.data.total_str)
    live_s = Label(overall_window, text="Live cells:")
    live_amount = Label(overall_window, textvariable=canvas.data.live_str)
    status_s = Label(overall_window, text="Status:")
    status_label = Label(overall_window, textvariable=canvas.data.status_str)
    tick_s = Label(overall_window, text="Current tick:")
    tick_label = Label(overall_window, textvariable=canvas.data.tick_str)


    total_s.grid(row=0, column=0, sticky=W)
    live_s.grid(row=1, column=0, sticky=W)
    status_s.grid(row=4, column=0, sticky=W)
    tick_s.grid(row=5, column=0, sticky=W)

    total_amount.grid(row=0, column=1, sticky=W)
    live_amount.grid(row=1, column=1, sticky=W)
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
    canvas.data.live_str = StringVar(master=root)
    canvas.data.trail = []
    r = cellStatistics()
    canvas.data.ramount = r


    # create a separate window to show the real time statistic of the world
    initOverallWindow()

    # set up events
    root.bind("<Button-1>", mousePressed)
#    root.bind("<Key>", keyPressed)
#    root.title(canvas.data.tick_str)
    root.title("Game of life")
    redrawAll()
    timerFired()
    # and launch the app
    root.mainloop()  # This call BLOCKS (so your program waits until you close the window!)
    plt.close('all')
    debugInfo()
    print("the end!")



# main program
run2DWorld(2, 8, 100, 100, 1000)