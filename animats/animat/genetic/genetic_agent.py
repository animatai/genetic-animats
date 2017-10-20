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

from ..motor import *
from ..sensor import *
from ..network import *
from ..agent import *
#from .. import environment
from .. import nodes

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

# helper function

def createName(kind, nodes, sort=True):
    if sort:
        return "%s(%s)" % (kind, ", ".join(sorted(nodes)))
    else:
        return "%s(%s)" % (kind, ", ".join(nodes))


def mean_of_two_values(a, b):
    return (a + b)/2


# choose one of given values at the probability as p, (0,1] (default value: 0.5)
def random_choose(a, b, p=0.5):
    if random.random() <= p:
        return a
    else:
        return b


# copy a node
def copy_node(node):
    node_new = Node(node.getName(), node.inputs, node.outputs)
    pass


# reproduce geno network configuration
def reproduce_netconf(agent):
    conf = {}
    conf['epsilon'] = agent.config.network.epsilon
    conf['utility_function'] = agent.config.network.utility_function
    conf['max_reward_history'] = agent.config.network.max_reward_history
    conf['q_learning_factor'] = agent.config.network.q_learning_factor
    conf['q_discount_factor'] = agent.config.network.q_discount_factor
    conf['reward_learning_factor'] = agent.config.network.reward_learning_factor
    conf['sensors'] = agent.config.geno.sensors
    conf['motors'] = agent.config.geno.motors
#    nodes_list = [v for k, v in agent.network.nodes.items() if v not in set(agent.network.sensors)]
#    nodes_other = [(v.name, [x.name for x in v.inputs], [x.name for x in v.outputs]) for v in nodes_list]

    return conf, agent.config.geno.others


def reproduce_genotype(agent):
    conf = {}
    conf['fertility'] = agent.config.geno.fertility
    conf['pubertyAge'] = agent.config.geno.pubertyAge
    conf['mortalityAge'] = agent.config.geno.mortalityAge
    conf['mortalityProb'] = agent.config.geno.mortalityProb
    conf['pregnancyLen'] = agent.config.geno.pregnancyLen
    conf['offspringNum'] = agent.config.geno.offspringNum
    conf['crossoverRate'] = agent.config.geno.crossoverRate
    conf['mutationRate'] = agent.config.geno.mutationRate
    conf['memorySizeMax'] = agent.config.geno.memorySizeMax
    conf['PLOTTER_ENABLED'] = False
    conf['PLOTTER_EVERY_FRAME'] = False
    conf['surprise_const'] = agent.config.surprise_const
    conf['wellbeeing_function'] = agent.config.wellbeeing_function
    conf['wellbeeing_const'] = agent.config.wellbeeing_const.copy()
    conf['objectives'] = agent.config.objectivesWithValues.copy()
    return conf


def reproduce(agent):
    conf = reproduce_genotype(agent)
    netconf, other_nodes = reproduce_netconf(agent)
    conf['network'] = netconf
    return conf, other_nodes

# compare whether agent_a is more vital than agent_b
# arguments: agent_a, agent_b
# return: True, agent_a > agent_b
#         False, agent_a < agent_b
def is_more_vital_than(agent_a, agent_b):
    # todo
    return True


# remove a node from dynamic graph
# arguments:
# return:

# check whether it is feasible for two agents to mate
# arguments: agent_a, agent_b
# return: True
#         False
def mate_feasible(agent_a, agent_b):
    # check whether two agents can mate or not
    debug('Function: mate_feasible')
    dm = difference_ratio_lists(agent_a.config.geno.motors, agent_b.config.geno.motors)
    dn = difference_ratio_lists(agent_a.config.geno.sensors, agent_b.config.geno.sensors)
    return (dm + dn)/2 <= 0.4


# compare the similarity between two list of nodes (sensors, actions and motors)
# arguments: list a, list b
# return value: [0,1] - 0% - 100%
def difference_ratio_lists(list_a, list_b):
    x = list(set(list_a) - set(list_b))
    y = list(set(list_b) - set(list_a))
    if len(list_a) > len(list_b):
        N = len(list_a)
    else:
        N = len(list_b)
    return (len(x)/N + len(y)/N)/2


# cross_over of in genotypes
def cross_over_genotype(agent_a, agent_b):
    conf = {}
    conf['fertility'] = agent_a.config.geno.fertility
#    conf['pubertyAge'] = int(round(mean_of_two_values(agent_a.config.geno.pubertyAge, agent_b.config.geno.pubertyAge)))
    conf['pubertyAge'] = random_choose(agent_a.config.geno.pubertyAge, agent_b, agent_b.config.geno.pubertyAge)

#    conf['mortalityAge'] = int(round(mean_of_two_values(agent_a.config.geno.mortalityAge, agent_b.config.geno.mortalityAge)))
    conf['mortalityAge'] = random_choose(agent_a.config.geno.mortalityAge, agent_b.config.geno.mortalityAge)

#    conf['mortalityProb'] = mean_of_two_values(agent_a.config.geno.mortalityProb, agent_b.config.geno.mortalityProb)
    conf['mortalityProb'] = random_choose(agent_a.config.geno.mortalityProb, agent_b.config.geno.mortalityProb)

#    conf['pregnancyLen'] = int(round(mean_of_two_values(agent_a.config.geno.pregnancyLen, agent_b.config.geno.pregnancyLen)))
    conf['pregnancyLen'] = random_choose(agent_a.config.geno.pregnancyLen, agent_b.config.geno.pregnancyLen)

#    conf['offspringNum'] = int(round(mean_of_two_values(agent_a.config.geno.offspringNum, agent_b.config.geno.offspringNum)))
    conf['offspringNum'] = random_choose(agent_a.config.geno.offspringNum, agent_b.config.geno.offspringNum)

#    conf['mutationRate'] = mean_of_two_values(agent_a.config.geno.mutationRate, agent_b.config.geno.mutationRate)
    conf['mutationRate'] = random_choose(agent_a.config.geno.mutationRate, agent_b.config.geno.mutationRate)

#    conf['memorySizeMax'] = int(round(mean_of_two_values(agent_a.config.geno.memorySizeMax, agent_b.config.geno.memorySizeMax)))
    conf['memorySizeMax'] = random_choose(agent_a.config.geno.memorySizeMax, agent_b.config.geno.memorySizeMax)

    conf['PLOTTER_ENABLED'] = False
    conf['PLOTTER_EVERY_FRAME'] = False

#    conf['surprise_const'] = mean_of_two_values(agent_a.config.surprise_const, agent_b.config.surprise_const)
    conf['surprise_const'] = random_choose(agent_a.config.surprise_const, agent_b.config.surprise_const)

    if random.random() <= 0.5:
        conf['wellbeeing_function'] = agent_a.config.wellbeeing_function
        conf['wellbeeing_const'] = agent_a.config.wellbeeing_const.copy()
    else:
        conf['wellbeeing_function'] = agent_b.config.wellbeeing_function
        conf['wellbeeing_const'] = agent_b.config.wellbeeing_const.copy()

    conf['objectives'] = agent_a.config.objectivesWithValues.copy()
    return conf

# remove a node from the graph
# argument: nodes - list of node tuples (name, inputs, outputs)
#                   where inputs and outputs are list of node names
#           node_tuple - the node to be removed (name, inputs, outputs)
#                   where inputs and outputs are list of node names
def remove_node(allnodes, node):
    # iterate nodes in its output and remove them
    for x in node[2]:
        # find the corresponding node
        tmp = [y for y in allnodes if y.name == x]
        # remove all nodes in outputs
        for y in tmp:
            allnodes = remove_node(allnodes, y)

    # iterate nodes in its input nodes and remove connections to itself
    for n in node[1]:
        loop_i = 0
        while loop_i < len(allnodes):
            if allnodes[loop_i][0] == n:
                # remove node from its outputs
                allnodes[loop_i][2] = [z for z in allnodes[loop_i][2] if z != node[0]]
            loop_i = loop_i + 1

    # remove current node
    return [y for y in allnodes if y[0] != node[0]]


def merge_sensor_nodes_by_name(agent_a, agent_b):
    s = set(agent_a.config.geno.sensors).union(agent_b.config.geno.sensors)
    return list(s)


def merge_motor_nodes_by_name(agent_a, agent_b):
    s = set(agent_a.config.geno.motors).union(agent_b.config.geno.motors)
    return list(s)


def merge_other_nodes_by_name(agent_a, agent_b):
    # extract name
    nodes_a_names = [n for n, i, o in agent_a.config.geno.others]
    nodes_b_names = [n for n, i, o in agent_b.config.geno.others]
    # b subtract a
    nodes_b_subtract_a_names = set(nodes_b_names) - set(nodes_a_names)

    # take all that are in b but not a (b - a)
    nodes_b_subtract_a = [(n, i, o) for n, i, o in agent_b.config.geno.others if n in nodes_b_subtract_a_names]
    # merge = a + ( b - a )
    nodes_other_all = agent_a.config.geno.others + nodes_b_subtract_a

    return nodes_other_all


# merge network configurations
# arguments: agent_a, agent_b
# return value: network configuration after cross over operation
#        conf = {
#            'epsilon': epsilon,
#            'utility_function': utility_function,
#            'q_function': q_function,
#            'max_reward_history': max_reward_history,
#            'q_learning_factor': q_learning_factor,
#            'q_discount_factor': q_discount_factor,
#            'reward_learning_factor': reward_learning_factor,
#            'sensors': sensors,
#            'motors': motors
#        }
def cross_over_netconf(agent_a, agent_b):
    conf = {}

    conf['epsilon'] = random_choose(agent_a.config.network.epsilon, agent_b.config.network.epsilon)
    conf['utility_function'] = random_choose(agent_a.config.network.utility_function, agent_b.config.network.utility_function)
    conf['max_reward_history'] = random_choose(agent_a.config.network.max_reward_history, agent_b.config.network.max_reward_history)
    conf['q_learning_factor'] = random_choose(agent_a.config.network.q_learning_factor, agent_b.config.network.q_learning_factor)
    conf['q_discount_factor'] = random_choose(agent_a.config.network.q_discount_factor, agent_b.config.network.q_discount_factor)
    conf['reward_learning_factor'] = random_choose(agent_a.config.network.reward_learning_factor, agent_b.config.network.reward_learning_factor)

    # take disjoint sensor nodes
    sensors = merge_sensor_nodes_by_name(agent_a, agent_b)
    # take disjoint motor nodes
    motors = merge_motor_nodes_by_name(agent_a, agent_b)
    # merge other nodes
    nodes_other = merge_other_nodes_by_name(agent_a, agent_b)

    conf['sensors'] = sensors
    conf['motors'] = motors

    return conf, nodes_other


# merge genotype and network configuration of two agents
# arguments: agent_a (female), agent_b (male)
# return value: agent configuration after cross over operation
def cross_over(agent_a, agent_b):
    # two agents mate
    debug('Function: cross_over')
    # crossover of genotypes
    conf = cross_over_genotype(agent_a, agent_b)
    # crossover of the network configurations
    netconf, other_nodes = cross_over_netconf(agent_a, agent_b)
    conf['network'] = netconf

    return conf, other_nodes


# mutate the list of other nodes (AND, SEQ etc)
# today we do not remove sensors or motors?
# arguments: agent configuration, list of other nodes (AND, SEQ etc)
# return value: agent configuration and list of other nodes after mutation
def mutate(conf, other_nodes):
    # agent's genome mutate
    debug('Function: mutate')
    # remove other nodes (AND, SEQ etc)
    loop_i = 0

    # randomly select elements to remove according to mutation rate, here 20%.
#    l = random.randint(1, len(other_nodes), int(len(other_nodes)/5))
#    for x in l:
#        other_nodes = remove_node(other_nodes, other_nodes[x])
    # for debugging purpose
    if len(other_nodes) > 1:
        i = random.randint(0, len(other_nodes)-1)
        other_nodes = remove_node(other_nodes, other_nodes[i])

    # add other nodes using existing sensors
    sensors = conf.get("network").get("sensors")
    for a, b in itertools.combinations(sensors, 2):
        if random.random() < 0.3:
            inputs = ['$' + x for x in sorted([a, b])]
            if random.random() <= 0.5:
                name = createName('AND', inputs)
            else:
                # SEQ inputs has its own order therefore cannot be sorted
                name = createName('SEQ', inputs, sort=False)
            # check whether the node already exists or not
            if name not in set([x[0] for x in other_nodes]):
                other_nodes.append((name, inputs, []))

    # todo: add random values to actions
    return conf, other_nodes


# Create an genetic agent
def createGeneticAgent(conf, objectives, gender='neuter'):
    agent = GeneticAgent(conf, createNetwork(conf.network, objectives, conf.seed), objectives, (0,0), gender)
    return agent


class GenoType():
    def __init__(self, conf, others):
        # genetic extension
        self.fertility = conf.get("fertility", "asexual")  # asexual, sexual
        self.pubertyAge = conf.get("pubertyAge", 15)
        self.mortalityAge = conf.get("mortalityAge", 300)
        self.mortalityProb = conf.get("mortalityProb", 0.1)
        self.pregnancyLen = conf.get("pregnancyLen", 3)
        self.offspringNum = conf.get("offspringNum", 2)
        self.crossoverRate = conf.get("crossoverRate", 0.8)
        self.mutationRate = conf.get("mutationRate", 0.1)
        self.memorySizeMax = conf.get("memorySizeMax", 200)
        # network of genotype
        self.sensors = conf.get("network", {}).get("sensors", "rgb0")
        self.motors = conf.get("network", {}).get("motors", ["left", "right", "up", "down", "eat", "drink"])
        self.others = others


class GeneticAgentConfig(AgentConfig):
    def __init__(self, conf, others=[]):
        super().__init__(conf)
        self.learning = conf.get("learning", True)
        # move objectives and initial values into AgentConfig
        self.objectivesWithValues = conf.get("objectives", {})
        self.objectives = list(self.objectivesWithValues.keys())
        # genetic extension
        self.geno = GenoType(conf, others)

# y=0 up
# x=0 left
ORIENTATION_MATRIX = [(0,-1), (1,-1), (1,0), (1,1), (0,1), (-1,1), (-1,0), (-1,-1)]

class GeneticAgent(Agent):
    def __init__(self, config, network, needs=None, position=(0,0), gender='neuter', growthRate=0.01):
        super().__init__(config, network, needs, position, growthRate)

        # genetic extension
        self.age = 0          # age of the agent
        self.newlyBorn = True # flag for newly born agent, True by default. To be set to False once going through step()
        self.gender = gender  # neuter/male/female
        self.pregnant = False # a female or neuter agent is pregnant
        self.chosen = False   # a male is chosen by a neighbour female
        self.pregnantAt = -1  # when the female or neuter became pregnant, -1: invalid
        self.offspringsConf = [] # configuration of its offsprings

    # genetic extension
    def is_mature(self):
        return self.age >= self.config.geno.pubertyAge

    def is_newlyborn(self):
        return self.newlyBorn

    def unset_newlyBorn(self):
        self.newlyBorn = False

    def is_to_die(self):
        return self.age >= self.config.geno.mortalityAge

    def is_female(self):
        return self.gender == 'female'

    def is_male(self):
        return self.gender == 'male'

    def is_neuter(self):
        return self.gender == 'neuter'

    def is_chosen(self):
        return self.chosen

    def become_chosen(self):
        self.chosen = True

    def reset_chosen(self):
        self.chosen = False

    def is_pregnant(self):
        return self.pregnant

    def become_pregnant(self):
        self.pregnant = True
        self.pregnantAt = self.age

    def reset_pregnancy(self):
        self.offspringsConf[:] = []
        self.pregnant = False
        self.pregnantAt = -1

    def wellbeeing(self):
        return self.needs.copy()

    def is_alive(self):
        return super().wellbeeing() > 0.0

    def grow(self):
        self.age = self.age + 1

    def ready_for_reproduction(self):

        if not self.is_mature():
            return False
        if self.is_male() and (not self.is_chosen()):
            return True
        if self.is_female() and (not self.is_pregnant()):
            return True
        return False

    def ready_for_delivery(self):
        return self.pregnant and (self.age - self.pregnantAt >= self.config.geno.pregnancyLen)

    def reproduce(self):
        # neuter agent reproduce alone
        debug('Function: reproduce')
        # neuter agent copy its network to its offspring
        loop_i = 0
        while loop_i < self.config.geno.offspringNum:
            conf, other_nodes = reproduce(self)
            # mutate()
            conf, other_nodes = mutate(conf, other_nodes)
            agent_conf = GeneticAgentConfig(conf, other_nodes)
            self.offspringsConf.append(agent_conf)
            loop_i = loop_i + 1
        # set the agent to be pregnant
        self.become_pregnant()
        return

    def mate(self, agent):
        debug('Function: mate')
        if self.is_neuter() or \
                not agent or \
                self.gender == agent.gender:
            error('mate: Cannot mate!')
            return

        # check the feasibility of mating
        if not mate_feasible(self, agent):
            return
        else:
            loop_i = 0
#            if random.random() < 0.5:
#                offspringNumber = self.config.geno.offspringNum
#            else:
#                offspringNumber = self.config.geno.offspringNum - 1
            offspringNumber = self.config.geno.offspringNum

            if self.is_female(): # self: female, agent: male
                gdebug('female agent mates with male agent!')
                while loop_i < offspringNumber:
                    conf, other_nodes = cross_over(self, agent)
                    # mutate()
                    conf, other_nodes = mutate(conf, other_nodes)
                    agent_conf = GeneticAgentConfig(conf, other_nodes)
                    self.offspringsConf.append(agent_conf)
                    loop_i = loop_i + 1
                self.become_pregnant()
                agent.become_chosen()
            else: # self: male, agent: female
                gdebug('male agent mates with female agent!')
                while loop_i < offspringNumber:
                    conf, other_nodes = cross_over(agent, self)
                    # mutate()
                    conf, other_nodes = mutate(conf, other_nodes)
                    agent_conf = GeneticAgentConfig(conf, other_nodes)
                    agent.offspringsConf.append(agent_conf)
                    loop_i = loop_i + 1
                self.become_chosen()
                agent.become_pregnant()

        return

    def deliver_offsprings(self):
        from .grass_agent import GeneticGrassAgent
        from .grass_agent import createGeneticGrassAgent
        from .sheep_agent import GeneticSheepAgent
        from .sheep_agent import createGeneticSheepAgent
        debug('Function: deliver_offspring')

        if self.is_male() or not self.is_pregnant() or not self.ready_for_delivery():
            error('Cannot deliver offsprings!')
        else:
            loop_i = 0
            offsprings = []
            # create offsprings of the agent
            while loop_i < len(self.offspringsConf):
                # randomly select the gender of the offspring
                if self.is_neuter():
                    gender = 'neuter'
                else:
                    if random.random() < 0.40:
                        gender = 'male'
                    else:
                        gender = 'female'
                # give birth to offspring
                if isinstance(self, GeneticSheepAgent):
                    child = createGeneticSheepAgent(self.offspringsConf[loop_i], self.config.objectivesWithValues.copy(), gender)
                elif isinstance(self, GeneticGrassAgent):
                    child = createGeneticGrassAgent(self.offspringsConf[loop_i], self.config.objectivesWithValues.copy(), gender)
                else:
                    child = createGeneticAgent(self.offspringsConf[loop_i], self.config.objectivesWithValues.copy(), gender)
                # add other nodes (AND, SEQ, OR etc)
                other_nodes = self.offspringsConf[loop_i].geno.others
                for n, i, o in other_nodes:
                    inp = [v for k, v in child.network.nodes.items() if k in set(i)]
                    outp = [v for k, v in child.network.nodes.items() if k in set(o)]
                    if n.startswith('AND'):
                        node = nodes.AndNode(inputs=inp, outputs=outp, virtual=False)
                        child.network.addNode(node)
                    elif n.startswith('SEQ'):
                        node = nodes.SEQNode(inputs=inp, outputs=outp, virtual=False)
                        child.network.addNode(node)
                    else: # todo: for other kinds of nodes
                        pass
                offsprings.append(child)
                loop_i = loop_i + 1

            if self.is_neuter():
                gender = 'neuter'
                gdebug(gender + ' agent deliver ' + str(loop_i) + ' offsprings!')
            else:
                gender = 'female'
                gdebug(gender + ' agent deliver ' + str(loop_i) + ' offsprings!')


        return offsprings


class GeneticAnimalAgent(GeneticAgent):
    def __init__(self, config, network, needs=None, position=(0, 0), gender='neuter', growthRate=0.01):
        super().__init__(config, network, needs, position, gender, growthRate)
        # genetic extension
        self.caught = False # whether this animal is caught or not

    def isCaught(self):
        return self.caught

    def setCaught(self, v):
        self.caught = v


