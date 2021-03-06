# -*- coding: utf-8 -*-
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


import os
import json

from pprint import pprint

from ..nodes import *
from ..agent import *
from ..environment import *
from .genetic_environment import GeneticEnvironment
from .genetic_agent import *
from .sheep_agent import *
from .grass_agent import *
from .xyworld_mul_agents import *

from datetime import date

# Setup logging
# =============

DEBUG_MODE = False
GUI_MODE = True

def debug(*args):
    if DEBUG_MODE: print('DEBUG:genetic_main:', *args)

def error(*args):
    print('ERROR:genetic_main:', *args)

def warn(*args):
    print('WARNING:genetic_main:', *args)


# The code
# ========

def getOutputPath():
    try:
        currentDir = os.path.dirname(os.path.abspath(__file__))
#        outputDir = currentDir + os.path.join('\output', datetime.datetime.now().isoformat())
        # change the format of logging folder name so that it can work on both Linux and Windows
        outputDir = currentDir + os.path.join('\output', datetime.datetime.now().strftime('log_smh_dmy_%S_%M_%H__%d_%m_%Y'))
        os.makedirs(outputDir)
        return (os.path.join(outputDir, "wellbeeing.csv"), outputDir)

    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def run(inputPath, outputPath, outputDir=None, wss=None):
    wellbeeings1 = []
    wellbeeings2 = []

    with open(inputPath, encoding='utf-8') as data_file:
        conf = json.loads(data_file.read())

    envConfig = EnvironmentConfig(conf)
    envConfig.outputPath = outputDir

    sagentConfig0 = GeneticAgentConfig(conf.get("agent1"))
    sagentConfig1 = GeneticAgentConfig(conf.get("agent2"))
    nagentConfig0 = GeneticAgentConfig(conf.get("nagent"))

    sagnt0 = createGeneticSheepAgent(sagentConfig0, sagentConfig0.objectivesWithValues.copy(), 'male')
    sagnt0.name = 'S0'

    sagnt1 = createGeneticSheepAgent(sagentConfig1, sagentConfig1.objectivesWithValues.copy(), 'male')
    sagnt1.name = 'S1'

    nagnt0 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt0.name = 'G0'
    nagnt1 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt1.name = 'G1'
    nagnt2 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt2.name = 'G2'
    nagnt3 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt3.name = 'G3'
    nagnt4 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt4.name = 'G4'
    nagnt5 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt5.name = 'G5'
    nagnt6 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt6.name = 'G6'
    nagnt7 = createGeneticGrassAgent(nagentConfig0, nagentConfig0.objectivesWithValues.copy(), 'neuter')
    nagnt7.name = 'G7'


    fieldConfig = {
        'numTilesPerSquare': (1, 1),
        'drawGrid': True,
        'randomTerrain': 0,
        'terrain': envConfig.worldmap,
        'agents': {
           'A': {
               'name': 'S0',
               'pos': (0, 0),
               'hidden': False
           },
            'B': {
                'name': 'S1',
                'pos': (0, 1),
                'hidden': False
            },
            'C': {
                'name': 'G0',
                'pos': (2, 2),
                'hidden': False
            }
        }
    }

    env = GeneticEnvironment(envConfig, None, wss, fieldConfig)
    sagnt0.position = (0, 0)
    env.add_thing(sagnt0, 1)
    sagnt1.position = (0, 1)
#    env.add_thing(sagnt1, 2)
    nagnt0.position = (0, 0)
    env.add_thing(nagnt0, 3)
    nagnt1.position = (0, 1)
    env.add_thing(nagnt1, 4)
    nagnt2.position = (0, 2)
    env.add_thing(nagnt2, 5)
    nagnt3.position = (0, 3)
    env.add_thing(nagnt3, 6)
    nagnt4.position = (0, 4)
    env.add_thing(nagnt4, 7)
    nagnt5.position = (0, 5)
    env.add_thing(nagnt5, 8)
    nagnt6.position = (0, 6)
    env.add_thing(nagnt6, 9)
    nagnt7.position = (0, 7)
    env.add_thing(nagnt7, 10)


    if wss is not None:
        wss.send_init(fieldConfig)

    if GUI_MODE:
        run2DWorld(env, envConfig.maxIterations)
    else:
        env.run(envConfig.maxIterations)


    if DEBUG_MODE:
        sagnt1.network.printNetwork()
        env.printWorld()
        for i,x in enumerate(sagnt1.trail):
            print((i, x[0], x[1]))
        print("SURPRISE MATRIX")
        pprint(sagnt1.surpriseMatrix)
        print("SEQ SURPRISE MATRIX")
        pprint(sagnt1.surpriseMatrix_SEQ)

#    wellbeeings1.append(sagnt0.wellbeeingTrail)
#    wellbeeings2.append(sagnt1.wellbeeingTrail)

    # Save the wellbeeing trails to file
#    fp = open(outputPath+'.1', "w")
#    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
#    fp.close()

    fp = open(outputPath+'.2', "w")
    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
    fp.close()
