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
from .genetic_agent import *


# Setup logging
# =============
GDEBUG_MODE = True

DEBUG_MODE = False

def gdebug(*args):
    if GDEBUG_MODE: print('GDEBUG:sheep_agent:', *args)

def debug(*args):
    if DEBUG_MODE: print('DEBUG:sheep_agent:', *args)

def error(*args):
    print('ERROR:sheep_agent:', *args)

def warn(*args):
    print('WARNING:sheep_agent:', *args)


# The code
# ========

# helper function

# Create a sheep agent
def createGeneticSheepAgent(conf, objectives, gender='female'):
    agent = GeneticSheepAgent(conf, createNetwork(conf.network, objectives, conf.seed), objectives, (0,0), gender)
    return agent


class GeneticSheepAgent(GeneticAnimalAgent):
    def is_bitten(self):
        # if the sheep is bitten, reduce the vitalness by 0.3
        self.needs['energy'] = max((self.needs['energy'] - 0.3), 0)
        self.needs['water'] = max((self.needs['water'] - 0.3), 0)

    def escape(self):
        pass
