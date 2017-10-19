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

    sagentConfig1 = GeneticAgentConfig(conf.get("agent1"))
    sagentConfig2 = GeneticAgentConfig(conf.get("agent2"))
#    nagentConfig1 = GeneticAgentConfig(conf.get("nagent"))

    sagnt1 = createGeneticSheepAgent(sagentConfig1, sagentConfig1.objectivesWithValues.copy(), 'male')
    sagnt1.name = 'S1'

    sagnt2 = createGeneticSheepAgent(sagentConfig2, sagentConfig2.objectivesWithValues.copy(), 'male')
    sagnt2.name = 'S2'

#    nagnt1 = createGeneticGrassAgent(nagentConfig1, nagentConfig1.objectivesWithValues.copy(), 'neuter')
#    nagnt1.name = 'G'


    fieldConfig = {
        'numTilesPerSquare': (1, 1),
        'drawGrid': True,
        'randomTerrain': 0,
        'terrain': envConfig.worldmap,
        'agents': {
           'A': {
               'name': 'S1',
               'pos': (0, 0),
               'hidden': False
           },
            'B': {
                'name': 'S2',
                'pos': (0, 1),
                'hidden': False
            },
            'C': {
                'name': 'G',
                'pos': (2, 2),
                'hidden': False
            }
        }
    }

    env = GeneticEnvironment(envConfig, None, wss, fieldConfig)
    sagnt1.position = (0, 0)
    env.add_thing(sagnt1, 1)
    sagnt2.position = (0, 1)
    env.add_thing(sagnt2, 2)
#    nagnt1.position = (2, 2)
#    env.add_thing(nagnt1, 3)

    if wss is not None:
        wss.send_init(fieldConfig)

    if GUI_MODE:
        run2DWorld(env, envConfig.maxIterations)
    else:
        env.run(envConfig.maxIterations)


    if DEBUG_MODE:
        sagnt2.network.printNetwork()
        env.printWorld()
        for i,x in enumerate(sagnt2.trail):
            print((i, x[0], x[1]))
        print("SURPRISE MATRIX")
        pprint(sagnt2.surpriseMatrix)
        print("SEQ SURPRISE MATRIX")
        pprint(sagnt2.surpriseMatrix_SEQ)

#    wellbeeings1.append(agnt1.wellbeeingTrail)
#    wellbeeings2.append(agnt2.wellbeeingTrail)

    # Save the wellbeeing trails to file
#    fp = open(outputPath+'.1', "w")
#    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
#    fp.close()

    fp = open(outputPath+'.2', "w")
    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
    fp.close()
