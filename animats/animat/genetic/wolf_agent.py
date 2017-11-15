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

import os
import sys
import math
import itertools
import copy


#from ..network import *
from .genetic_agent import createNetwork
from .sheep_agent import GeneticAnimalAgent


# Setup logging
# =============
GDEBUG_MODE = True

DEBUG_MODE = False

def gdebug(*args):
    if GDEBUG_MODE: print('GDEBUG:wolf_agent:', *args)

def debug(*args):
    if DEBUG_MODE: print('DEBUG:wolf_agent:', *args)

def error(*args):
    print('ERROR:wolf_agent:', *args)

def warn(*args):
    print('WARNING:wolf_agent:', *args)


# The code
# ========

# helper function

# Create a sheep agent
def createGeneticWolfAgent(conf, objectives, gender='female'):
    agent = GeneticWolfAgent(conf, createNetwork(conf.network, objectives, conf.seed), objectives, (0,0), gender)
    return agent



class GeneticWolfAgent(GeneticAnimalAgent):
    def lockPrey(self, p):
        self.prey = p

    def getLockedPrey(self):
        return self.prey

    def resetLockPrey(self):
        self.prey = None

    def escape(self):
        pass
