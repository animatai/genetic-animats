# -*- coding: utf-8 -*-
#
#    pyAnimat - Simulate Animats using Transparent Graphs as a way to AGI
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


import math
import itertools
import copy

from ..motor import *
from ..sensor import *
from ..network import *
from ..agent import *
from .. import environment
from .. import nodes
from .genetic_agent import *

#from agents import Agent as AgentClass

# Setup logging
# =============
GDEBUG_MODE = True

DEBUG_MODE = False

def gdebug(*args):
    if GDEBUG_MODE: print('GDEBUG:genetic_agent:', *args)

def debug(*args):
    if DEBUG_MODE: print('DEBUG:genetic_agent:', *args)

def error(*args):
    print('ERROR:genetic_agent:', *args)

def warn(*args):
    print('WARNING:geentic_agent:', *args)




# The code
# ========

# Create a grass agent
def createGeneticGrassAgent(conf, objectives, gender='neuter', growthRate=0.01):
    agent = GeneticGrassAgent(conf, createNetwork(conf.network, objectives, conf.seed), objectives, (0,0), gender, growthRate)
    return agent



class GeneticGrassAgent(GeneticPlantAgent):
    def grow_by_length(self):
        self.needs['length'] = min((self.needs['length'] + self.growthRate), 1)

    def is_mature(self):
        return self.needs['length'] >= self.config.geno.pubertyAge

    def is_to_die(self):
        return self.needs['length'] < self.config.objectivesWithValues['length']

    def is_grazed(self):
        # if the grass is grazed, reduce the length by 0.2
        self.needs['length'] = max((self.needs['length'] - 0.2), 0)
