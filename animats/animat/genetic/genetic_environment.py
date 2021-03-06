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
import datetime
import itertools
import random

from .. import agent as agentModule
from ..network import *
from ..sensor import *
from ..motor import *
from ..environment import Environment as EnvClass
from ..environment import EnvironmentConfig
from .grass_agent import *
from .sheep_agent import *
from .wolf_agent import *

from agents import Thing
from agents import Agent


# Setup logging
# =============
GENETIC_ON = True
DEBUG_MODE = True
GDEBUG_MODE = False
ADEBUG_MODE = False

def adebug(*args):
    if ADEBUG_MODE: print('ADEBUG:genetic_environments:', *args)

def gdebug(*args):
    if GDEBUG_MODE: print('GDEBUG:genetic_environments:', *args)

def debug(*args):
    if DEBUG_MODE: print('DEBUG:genetic_environments:', *args)

def error(*args):
    print('ERROR:genetic_environments:', *args)

def warn(*args):
    print('WARNING:genetic_environments:', *args)

# The code
# =========

# check whether two locations are adjacent or not
def locations_are_adjacent(position_a, position_b):

    if (position_a[0] == position_b[0]) and abs(position_a[1] - position_b[1]) == 1:
        return True
    elif (position_a[1] == position_b[1]) and abs(position_a[0] - position_b[0]) == 1:
        return True
    elif (abs(position_a[0] - position_b[0]) == 1) and (abs(position_a[1] - position_b[1]) == 1):
        return True
    else:
        return False


class GeneticEnvironment(EnvClass):
    def __init__(self, config=None, objectives=None, wss=None, fieldConfig=None):
        super().__init__(config, objectives, wss, fieldConfig)
        self.genetic = True
        self.counter = 0

    # Updates the sensors, does not return a percept
    def percept(self, agent, delta=(0,0)):
        '''prints & return a list of things that are in our agent's location'''
        # y: row
        # x: col
        y = (agent.position[1]+delta[1]) % len(self.world)
        x = (agent.position[0]+delta[0]) % len(self.world[y])
        cell = self.world[y][x]

        observation = self.config.blocks.get(cell,{}).copy()

        if len(observation) > 1:
            debug("Something is wrong!")

        if isinstance(agent, GeneticSheepAgent):
            # find out whether there is grass
            if self.some_agents_at(agent.position, GeneticGrassAgent) and ('r' in observation):
                # if there is grass on top of sand, set 'g' to be 1, 'r' to be 0
                observation.clear()
                observation['g'] = 1
                debug("Grass is here, observation: ", str(observation))

        if isinstance(agent, GeneticWolfAgent):
            sheep = self.list_agents_at(agent.position, GeneticSheepAgent)
            if len(sheep) > 0:
                # lock this sheep
                agent.lockPrey(sheep[0])
                adebug('Sheep ' + sheep[0].name + ' is locked by wolf ' + agent.name)

                # if current block is sand, disable it (adaption to the existing code)
                if 'r' in observation:
                    observation.clear()
                # trigger sensors
                observation['g'] = 1
                adebug("Sheep is nearby, observation: ", str(observation))

        debug('--------------\npercept - cell:' + cell + ", observation:" + str(observation))
        agent.network.tick(observation)

        # for the moment no hormone for plant agent
        if isinstance(agent, type(GeneticPlantAgent)):
            return None

        # check surrounding agents for secreting or stopping hormon
        adjacents = self.adjacent_agents_at(agent, type(agent))
        n = len(adjacents)
        if agent.is_hormoneOn():
            if n <= agent.getHormoneThresholdLower() or n >= agent.getHormoneThresholdUpper() :
                agent.hormoneOff()
        else:
            if n == agent.getHormoneThresholdTrigger():
                agent.hormoneOn()

        return None

    def step(self):
        """Run the environment for one time step. If the
        actions and exogenous changes are independent, this method will
        do. If there are interactions between them, you'll need to
        override this method."""
        if not self.is_done():
            actions = []
            # the flag newlyBorn is to mark the newly born grass so that
            # it will not cause problem between percept and execute_action
            # (see comments in execute_action)
            for agent in self.agents:
                agent.unset_newlyBorn()
            for agent in self.agents:
                if agent.is_to_die():
                    if isinstance(agent, GeneticGrassAgent):
                        tmp = 'grass ' + agent.name
                    elif isinstance(agent, GeneticSheepAgent):
                        tmp = 'sheep ' + agent.name
                    elif isinstance(agent, GeneticWolfAgent):
                        tmp = 'wolf ' + agent.name
                    else:
                        tmp = agent.name
                    adebug('Remove dead agent ' + tmp)
                    self.delete_thing(agent)
            for agent in self.agents:
                if agent.alive:
                    self.agent_reproduce(agent)
                    if isinstance(agent, GeneticGrassAgent):
                        actions.append(('grow', None, 0))
                    else:
                        actions.append(agent.program(self.percept(agent)))
                else:
                    actions.append("")
            # reset the flag "chosen' of male agents after reproduction activity
            for agent in self.agents:
                agent.grow()
                agent.reset_chosen()
            adebug('Environment: STEP tick: ', self.counter)
            self.counter = self.counter + 1
            adebug('Environment:STEP actions', actions)
            for (agent, action) in zip(self.agents, actions):
                if isinstance(agent, GeneticAnimalAgent):
                    self.execute_action(agent, action)
                elif isinstance(agent, GeneticGrassAgent):
                    agent.grow_by_length()
                    #adebug('Grass ' + agent.name + ' grows')
                    agent.wellbeeingTrail.append(agent.wellbeeing())

            self.exogenous_change()

    def execute_action(self, agent, action):
        '''changes the state of the environment based on what the agent does.'''
        adebug("execute_action - position:", agent.position, ", action:", action)

        act, _, _ = action
        reward = self.takeAction(agent, act)
        if act == 'eat':
            if isinstance(agent, GeneticSheepAgent):
                # a possible sequence is:
                # 1, agent choose to 'eat' at a sand block with no grass, perception outputs 'r' = 1, reward is negative.
                # 2, during step, a grass is given to birth at this sand block
                # 3, agent now exectute the action: 'eat' with the negative reward. If we do not exclude the new grass which
                #    did not exist before, we will eat the grass with the negative reward. So we ignore newly born grass to
                #    keep consistency with the previous action/rewards ('eat' in a sand block with negative rewards)
                all_grass = self.list_agents_at(agent.position, GeneticGrassAgent)
                old_grass = [x for x in all_grass if not x.is_newlyborn()]
                if len(old_grass) > 0:
                    old_grass[0].is_grazed()
                    adebug('sheep ' + agent.name + ' graze grass ' + old_grass[0].name)
                    # for debugging
                    if reward['energy'] < 0:
                        debug("something is wrong!")
            elif isinstance(agent, GeneticWolfAgent):
                sheep = agent.getLockedPrey()
                if sheep != None:
                    sheep.is_bitten()
                    agent.resetLockPrey()
                    adebug('Sheep ' + sheep.name + ' is bitten by wolf ' + agent.name)

        agent.takeAction(action, reward)

    def _getReward(self, action, sensor, status):
        rm = self.config.rewardMatrix
        am = rm.get(action, rm.get('*',{}))
        r = am.get(sensor, am.get('*',0.0))
        reward = environment.makeRewardDict(r, status)
        # TODO: truncate reward if need is satisfied
        return reward

    def takeAction(self, agent, action):
        cell = self.currentCell(agent)
        # TODO: reward for multiple active sensors?
        sensor = agent.network.activeSensors()
        if sensor[0].name[1:] == 'g' and action == 'eat':
            adebug("interesting for debugging")
        reward = self._getReward(action, sensor[0].name[1:], agent.needs)
        adebug("takeAction - reward:", reward, ", position:", agent.position, ", action:", action, ", wss:",
              self.wss)

        def move_agent(agent, dx, dy):
            nx = agent.position[0] + dx
            ny = agent.position[1] + dy
            if self.config.is_torus:
                nx = nx % self.getWidth()
                ny = ny % self.getHeight()
            self._playback(agent, action, nx, ny)
            if nx >= 0 and nx < self.getWidth() and ny >= 0 and ny < self.getHeight():
                # if the cell to go is not occupied by same kind of animal as itself
                if not self.some_agents_at((nx, ny), type(agent)):
                    agent.position = (nx, ny)
                if self.wss is not None:
                    self.fieldConfig['agents'][agent.name]['pos'] = agent.position
                    self.wss.send_update_agent(agent.name, self.fieldConfig['agents'][agent.name])

        if action == 'up':
            dx, dy = agentModule.ORIENTATION_MATRIX[agent.orientation % 8]
            move_agent(agent, dx, dy)
        elif action == 'down':
            dx, dy = agentModule.ORIENTATION_MATRIX[(agent.orientation + 4) % 8]
            move_agent(agent, dx, dy)
        elif action == 'left':
            dx, dy = agentModule.ORIENTATION_MATRIX[(agent.orientation - 2) % 8]
            move_agent(agent, dx, dy)
        elif action == 'right':
            dx, dy = agentModule.ORIENTATION_MATRIX[(agent.orientation + 2) % 8]
            move_agent(agent, dx, dy)
        elif action == 'turn_right':
            agent.orientation = (agent.orientation + 1) % 8
            self._playback(agent, action)
        elif action == 'turn_left':
            agent.orientation = (agent.orientation - 1) % 8
            self._playback(agent, action)
        elif action == 'eat':
            self._playback(agent, action)
            if self.wss is not None:
                self.wss.send_print_message('Agent ' + agent.name + ' ate')
        elif action == 'drink':
            self._playback(agent, action)
            if self.wss is not None:
                self.wss.send_print_message('Agent ' + agent.name + ' drank')

        trans = self.config.transform.get(action, {}).get(cell, None)
        if trans:
            debug("takeAction - *** transform action:", action, ", cell:", cell, ", trans", trans)
            self.setCurrentCell(agent, trans)

        return reward

    def list_agents_at(self, location, tclass=Thing):
        """Return all things exactly at a given location."""
        return [agent for agent in self.agents
                if agent.position == location and isinstance(agent, tclass)]

    def some_agents_at(self, location, tclass=Thing):
        """Return true if at least one of the things at location
        is an instance of class tclass (or a subclass)."""
        return self.list_agents_at(location, tclass) != []

    def adjacent_agents_at(self, agent, tclass=Thing):
        return [x for x in self.agents
                if agent.name != x.name and locations_are_adjacent(x.position, agent.position) and isinstance(x, tclass)]

    def adjacent_available_positions_at(self, agent, tclass=Thing):
        occupied = [x.position for x in self.agents
                if agent.name != x.name and locations_are_adjacent(x.position, agent.position) and isinstance(x, tclass)]
        available = self.adjacent_locations(agent.position)
        return list(set(available) - set(occupied))

    # check whether two locations are adjacent or not
    def adjacent_locations(self, position):
        locations = []
        if position[0] > 0:
            locations.append((position[0] - 1, position[1]))
            if position[1] > 0:
                locations.append((position[0] - 1, position[1] - 1))
            if position[1] < (self.getHeight() - 1):
                locations.append((position[0] - 1, position[1] + 1))

        if position[0] < (self.getWidth() - 1):
            locations.append((position[0] + 1, position[1]))
            if position[1] > 0:
                locations.append((position[0] + 1, position[1] - 1))
            if position[1] < (self.getHeight() - 1):
                locations.append((position[0] + 1, position[1] + 1))

        if position[1] > 0:
            locations.append((position[0], position[1] - 1))

        if position[1] < (self.getHeight() - 1):
            locations.append((position[0], position[1] + 1))

        return locations

    # agent tries to reproduce when possible
    def agent_reproduce(self, agent):
        if GENETIC_ON == False:
            return
        # exclude agents that
        #  1, are already chosen by other agents in previous iterations of current tick (not previous tick),
        #  2, are pregnant but not ready for delivery
        #  3, are not old enough ( mature )
        if not agent.is_mature():
            return

        if agent.is_pregnant() and (not agent.ready_for_delivery()):
            return

        if agent.is_chosen():
            return

        # The agent (neuter or female) is pregnant and is ready for delivery
        if agent.is_pregnant() and agent.ready_for_delivery():
            # agent delivery offsprings
            gdebug('Deliver offsprings')
            offsprings = agent.deliver_offsprings()
            if len(offsprings) == 0:
                error("agent_reproduce: error, no children to be delivered!")
                agent.reset_pregnancy()
                return

            if isinstance(agent, GeneticPlantAgent):
                locations = self.adjacent_available_positions_at(agent,GeneticPlantAgent)
                #gdebug('agent_reproduce - plant: ego location ', agent.position)
                #gdebug('agent_reproduce - plant: available positions ', locations)
                # filter out water blocks for plant agents
                locations = [x for x in locations if self.world[x[1]][x[0]] != 'c']
                #debug('agent_reproduce: available positions after filtering water blocks', locations)
            else:
                locations = self.adjacent_available_positions_at(agent,GeneticAnimalAgent)

            if len(locations) == 0:
                #gdebug("agent_reproduce: no available cells for offsprings delivery!")
                agent.reset_pregnancy()
                return

            if len(locations) <= len(offsprings):
                size = len(locations)
                # randomly select a number of offsprings
                offsprings = random.sample(set(offsprings), size)
            else:
                size = len(offsprings)
                # randomly select a number of locations
                locations = random.sample(set(locations), size)

            loop_i = 0
            for x, p in zip(offsprings, locations):
                x.name = agent.name + ' ' + str(agent.age) + ' ' + str(loop_i)
                x.position = p
                self.add_thing(x)
                loop_i = loop_i + 1
            agent.reset_pregnancy()
            debug(str(len(self.agents)) + ' agents! neuter: ' + str(len([x for x in self.agents if x.gender == 'neuter'])) + ', male: ' + str(len([x for x in self.agents if x.gender == 'male'])) + ', female: ' + str(len([x for x in self.agents if x.gender == 'female'])))
            return

        # mate at the chance of 0.5
        if random.random() < 0.5:
            return

        # if the agent is neuter
        if agent.is_neuter():
            # the agent reproduce and becomes pregnant at the given probability
            gdebug('neuter reproduce')
            agent.reproduce()
        else:  # agent can be male or female
            # find out the list of mature agents adjacent to the agent
            adjacent = self.adjacent_agents_at(agent, type(agent))
            if len(adjacent) > 0:  # if the list is not empty
                #debug('agent (male/female) has adjacent agents around!')
                if agent.is_female():  # current agent is female
                    # female agent become pregnant by finding a male neighbour agent
                    males = [a for a in adjacent if a.is_male() and a.ready_for_reproduction()]
                    if len(males) > 0:
                        # choose a male agent randomly, take the first
                        debug('female -> male')
                        agent.mate(males[0])

                else:  # current agent is male
                    # male help a female neighbour agent become pregnant
                    females = [a for a in adjacent if a.is_female() and a.ready_for_reproduction()]
                    if len(females) > 0:
                        debug('male -> female')
                        agent.mate(females[0])
