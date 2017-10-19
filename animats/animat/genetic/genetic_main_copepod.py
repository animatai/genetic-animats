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
#from ..agent import *
from ..environment import *
from .genetic_environment import *
from .genetic_agent import *

from datetime import date

# Setup logging
# =============

DEBUG_MODE = False

def debug(*args):
    if DEBUG_MODE: print('DEBUG:genetic_main_copepod:', *args)

def error(*args):
    print('ERROR:genetic_main_copepod:', *args)

def warn(*args):
    print('WARNING:genetic_main_copepod:', *args)


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

    agentConfig1 = GeneticAgentConfig(conf.get("agent"))

    agnt1 = createGeneticAgent(agentConfig1, {k:1 for k in conf.get("objectives")}, 'neuter')
    agnt1.name = 'A'


    fieldConfig = {
        'numTilesPerSquare': (1, 1),
        'drawGrid': True,
        'randomTerrain': 0,
        'terrain': envConfig.worldmap,
        'agents': {
           'A': {
               'name': 'A',
               'pos': (0, 0),
               'hidden': False
           }
        }
    }

    env = GeneticEnvironment(envConfig, None, wss, fieldConfig)
    env.add_thing(agnt1, 1)

    if wss is not None:
        wss.send_init(fieldConfig)

    env.run(envConfig.maxIterations)

    if DEBUG_MODE:
        agnt1.network.printNetwork()
        env.printWorld()
        for i,x in enumerate(agnt1.trail):
            print((i, x[0], x[1]))
        print("SURPRISE MATRIX")
        pprint(agnt1.surpriseMatrix)
        print("SEQ SURPRISE MATRIX")
        pprint(agnt1.surpriseMatrix_SEQ)

#    wellbeeings1.append(agnt1.wellbeeingTrail)
#    wellbeeings2.append(agnt2.wellbeeingTrail)

    # Save the wellbeeing trails to file
#    fp = open(outputPath+'.1', "w")
#    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
#    fp.close()

    fp = open(outputPath+'.2', "w")
    print("\n".join([";".join([str(x).replace(".",",") for x in line]) for line in zip(*wellbeeings1)]), file=fp)
    fp.close()
