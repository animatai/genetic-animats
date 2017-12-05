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
import numpy as np

from ..motor import *
from ..sensor import *
from ..network import *
from ..agent import *
#from .. import environment
from .. import nodes

#from agents import Agent as AgentClass

# Setup logging
# =============
GDEBUG_MODE = False

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


def merge_two_dicts(a, b):
    c = a.copy()
    c.update(b)
    return c


def random_choices(a, s):
    r = []

    if s > 0:
        index = np.random.choice(len(a), s, replace=False)
        for i in index:
            r.append(a[i])

    return r


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
    conf['nodeCost'] = agent.config.geno.nodeCost
    conf['sensors'] = agent.config.geno.sensors
    conf['motors'] = agent.config.geno.motors
    conf['others'] = agent.config.geno.others
#    conf['qtables'] = agent.config.geno.qtables.copy()

    q_tables = agent.config.geno.qtables.copy()

    for s in agent.config.geno.sensors:
        s = '$'+s
        q = q_tables.get(s, None)
        if q == None:
            v = {k: 0 for k in agent.network.objectives}
            q = {k:v.copy() for k in agent.config.geno.motors}
            q_tables[s] = q

    conf['qtables'] = q_tables

#    nodes_list = [v for k, v in agent.network.nodes.items() if v not in set(agent.network.sensors)]
#    nodes_other = [(v.name, [x.name for x in v.inputs], [x.name for x in v.outputs]) for v in nodes_list]

    return conf


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
    netconf = reproduce_netconf(agent)
    conf['network'] = netconf
    return conf

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
#    dm = difference_ratio_lists(agent_a.config.geno.motors, agent_b.config.geno.motors)
#    dn = difference_ratio_lists(agent_a.config.geno.sensors, agent_b.config.geno.sensors)
#    return (dm + dn)/2 <= 0.4
    return (set(agent_a.config.geno.motors) == set(agent_b.config.geno.motors)
            and set(agent_a.config.geno.sensors) == set(agent_b.config.geno.sensors))



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
    # remove current node
    gdebug('remove_node nodes: ', allnodes)
    gdebug('node to be removed: ', node)
    allnodes = [y for y in allnodes if y[0] != node[0]]
    #gdebug('after removal: ', allnodes)

    name, inp, outp = node
    # iterate nodes in its output and remove them
    for x in outp:
        gdebug('remove_node: remove output nodes. ')
        # find the corresponding node
        tmp = [y for y in allnodes if y.name == x]
        # remove all nodes in outputs
        for y in tmp:
            allnodes = remove_node(allnodes, y)

    # iterate nodes in its input nodes and remove connections to itself
    for n in inp:
        gdebug('remove_node: remove connections to input nodes!')
        loop_i = 0
        while loop_i < len(allnodes):
            if allnodes[loop_i][0] == n:
                # remove node from its outputs
                gdebug('remove_node: find the input node: ', allnodes[loop_i])
                x,y,z = allnodes[loop_i]
                z = [x for x in z if x != node[0]]
                allnodes[loop_i] = (x,y,z)
                gdebug('remove_node: update the input node: ', allnodes[loop_i])
            loop_i = loop_i + 1

    gdebug('remove_node: final result ', allnodes)
    return allnodes

# add a node to the graph
# argument: allnodes - list of node tuples (name, inputs, outputs)
#                   where inputs and outputs are list of node names
#           node     - the node to be added (name, inputs, outputs)
#                   where inputs and outputs are list of node names
def add_node(allnodes, node):
    # remove current node
    gdebug('add_node nodes: ', allnodes)
    gdebug('node to be added: ', node)

    name, inp, outp = node

    # iterate nodes in its input nodes and remove connections to itself
    for n in inp:
        loop_i = 0
        while loop_i < len(allnodes):
            if allnodes[loop_i][0] == n:
                # remove node from its outputs
                gdebug('add_node: find the input node: ', allnodes[loop_i])
                x,y,z = allnodes[loop_i]
                z.append(name)
                allnodes[loop_i] = (x,y,z)
                gdebug('add_node: update the input node: ', allnodes[loop_i])
            loop_i = loop_i + 1

    allnodes.append(node)
    gdebug('add_node: final result ', allnodes)
    return allnodes


def merge_sensor_nodes_by_name(agent_a, agent_b):
    s = set(agent_a.config.geno.sensors).union(agent_b.config.geno.sensors)

    # get the Q tables for actions of nodes (sensors + others)
    q_tables = {}

    for x in s:
        # get Q table of every action of sensor x
        x = '$'+x
        q1 = agent_a.config.geno.qtables.get(x, None)
        q2 = agent_b.config.geno.qtables.get(x, None)
        if q1 != None and q2 == None:
            # get Q table of agent a
            q_tables[x] = q1
        elif q1 == None and q2 != None:
            # get Q table of agent b
            q_tables[x] = q2
        elif q1 != None and q2 != None:
            q_tables[x] = random_choose(q1, q2)
        else:
            # if none of agents have initial q tables, we use None
            v = {k:0 for k in agent_a.network.objectives}
            q_tables[x] = {k:v.copy() for k in agent_a.config.geno.motors}

    return list(s), q_tables


def merge_motor_nodes_by_name(agent_a, agent_b):
    s = set(agent_a.config.geno.motors).union(agent_b.config.geno.motors)
    return list(s)


def compl_dependency(node, all_nodes, sensors_name, existing_name):
    _, i, _ = node

    dependent_nodes = []

    gdebug('compl_dependency: node ', node)

    # add the nodes that are dependent to current node while neither in sensors nor in others
    for n in i:
        if n not in set(sensors_name) and n not in set(existing_name):
            gdebug('compl_dependency: n ', n)
            gdebug('compl_dependency: existing names ', existing_name)
            gdebug('compl_dependency: all_nodes ', all_nodes)

            tmp = [x for x in all_nodes if x[0] == n]
            gdebug('compl_dependency: tmp ', tmp)
            nodes = compl_dependency(tmp[0], all_nodes, sensors_name, existing_name)
            gdebug('compl_dependency: results ', nodes)
            dependent_nodes.append(tmp[0])
            dependent_nodes.extend(nodes)

    return dependent_nodes


def merge_other_nodes_by_name(agent_a, agent_b):
    # extract name
    nodes_a_names = [n for n, i, o in agent_a.config.geno.others]
    nodes_b_names = [n for n, i, o in agent_b.config.geno.others]

    gdebug('merge_other_nodes: node a ', nodes_a_names)
    gdebug('merge_other_nodes: node b ', nodes_b_names)

    # preserve the nodes in common
    nodes_a_intersect_b_names = set(nodes_a_names).intersection(set(nodes_b_names))
    nodes_a_intersect_b = [(n, i, o) for n, i, o in agent_a.config.geno.others if n in nodes_a_intersect_b_names]
    gdebug('merge_other_nodes: a intersect b ', nodes_a_intersect_b)


    # a subtract b
    nodes_a_subtract_b_names = set(nodes_a_names) - set(nodes_b_names)
    # take all that are in a but not b (a - b)
    nodes_a_subtract_b = [(n, i, o) for n, i, o in agent_a.config.geno.others if n in nodes_a_subtract_b_names]
    # b subtract a
    nodes_b_subtract_a_names = set(nodes_b_names) - set(nodes_a_names)
    # take all that are in b but not a (b - a)
    nodes_b_subtract_a = [(n, i, o) for n, i, o in agent_b.config.geno.others if n in nodes_b_subtract_a_names]

    gdebug('merge_other_nodes: a-b ', nodes_a_subtract_b_names)
    gdebug('merge_other_nodes: b-a ', nodes_b_subtract_a_names)

    if nodes_a_names != [] or nodes_b_names != []:
        gdebug('merge other nodes: for debugging!')

    # according to the cross-over percentage drawn from the uniform dist.
    percentage = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    p = random.choice(percentage)

    # draw the random crossover percentage p from uniform distribution
    if len(nodes_a_subtract_b) != 0:
        amount_from_a = max(round(len(nodes_a_subtract_b)*p), 1)
    else:
        amount_from_a = 0
    if len(nodes_b_subtract_a) != 0:
        amount_from_b = max(round(len(nodes_b_subtract_a)*(1-p)), 1)
    else:
        amount_from_b = 0

    # take the nodes randomly (p% from a, rest from b)
    nodes_from_a = random_choices(nodes_a_subtract_b, amount_from_a)
    nodes_from_b = random_choices(nodes_b_subtract_a, amount_from_b)

    sensors_a = set(['$'+x for x in agent_a.config.geno.sensors])
    sensors_b = set(['$'+x for x in agent_b.config.geno.sensors])

    dependency_a = []
    dependency_b = []

    # b subtract a
    gdebug('merge_other_nodes: nodes from a ', nodes_from_a)
    # complement the dependency of selected nodes
    existing_names = set([n for n, i, o in nodes_from_a]).union(nodes_a_intersect_b_names)
    for x in nodes_from_a:
        tmp = compl_dependency(x, nodes_a_subtract_b, sensors_a, existing_names)
        existing_names.union(set([n[0] for n in tmp]))
        dependency_a.extend(tmp)

    gdebug('merge_other_nodes: nodes from b ', nodes_from_b)
    existing_names = set([n for n, i, o in nodes_from_b]).union(nodes_a_intersect_b_names)
    for y in nodes_from_b:
        tmp = compl_dependency(y, nodes_b_subtract_a, sensors_b, existing_names)
        existing_names.union(set([n[0] for n in tmp]))
        dependency_b.extend(tmp)

    nodes_from_a.extend(dependency_a)
    nodes_from_b.extend(dependency_b)

    # put them together
    nodes_all = []
    nodes_all.extend(nodes_a_intersect_b)
    nodes_all.extend(nodes_from_a)
    nodes_all.extend(nodes_from_b)

    # get the Q tables for actions of other nodes
    q_tables = {}
    v = {k: 0 for k in agent_a.network.objectives}

    for n, i, o in nodes_a_intersect_b:
        # get Q table of every action of common node n
        q1 = agent_a.config.geno.qtables.get(n, None)
        q2 = agent_b.config.geno.qtables.get(n, None)
        if q1 != None and q2 == None:
            # get Q table of agent a
            q_tables[n] = q1
        elif q1 == None and q2 != None:
            # get Q table of agent b
            q_tables[n] = q2
        elif q1 != None and q2 != None:
            q_tables[n] = random_choose(q1, q2)
        else:
            # if none of agents have initial q tables, we use None
            q_tables[n] = {k:v.copy() for k in agent_a.config.geno.motors}

    for n, i, o in nodes_from_a:
        # get Q table of every action of other node n in agent a
        q = agent_a.config.geno.qtables.get(n, None)
        if q != None:
            # get Q table of agent a
            q_tables[n] = q
        else:
            # if none of agents have initial q tables, we use None
            q_tables[n] = {k:v.copy() for k in agent_a.config.geno.motors}

    for n, i, o in nodes_from_b:
        # get Q table of every action of other node n in agent a
        q = agent_b.config.geno.qtables.get(n, None)
        if q != None:
            # get Q table of agent a
            q_tables[n] = q
        else:
            # if none of agents have initial q tables, we use None
            q_tables[n] = {k:v.copy() for k in agent_a.config.geno.motors}

    return nodes_all, q_tables


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
#            'nodeCost': node cost factor
#            'sensors': sensors,
#            'motors': motors,
#            'others': other nodes
#        }
def cross_over_netconf(agent_a, agent_b):
    conf = {}

    conf['epsilon'] = random_choose(agent_a.config.network.epsilon, agent_b.config.network.epsilon)
    conf['utility_function'] = random_choose(agent_a.config.network.utility_function, agent_b.config.network.utility_function)
    conf['max_reward_history'] = random_choose(agent_a.config.network.max_reward_history, agent_b.config.network.max_reward_history)
    conf['q_learning_factor'] = random_choose(agent_a.config.network.q_learning_factor, agent_b.config.network.q_learning_factor)
    conf['q_discount_factor'] = random_choose(agent_a.config.network.q_discount_factor, agent_b.config.network.q_discount_factor)
    conf['reward_learning_factor'] = random_choose(agent_a.config.network.reward_learning_factor, agent_b.config.network.reward_learning_factor)
    conf['hormoneThresholdTrigger'] = random_choose(agent_a.config.geno.hormoneThresholdTrigger, agent_b.config.geno.hormoneThresholdTrigger)
    conf['hormoneThresholdSecreteLower'] = random_choose(agent_a.config.geno.hormoneThresholdSecreteLower, agent_b.config.geno.hormoneThresholdSecreteLower)
    conf['hormoneThresholdSecreteUpper'] = random_choose(agent_a.config.geno.hormoneThresholdSecreteUpper, agent_b.config.geno.hormoneThresholdSecreteUpper)
    conf['nodeCost'] = random_choose(agent_a.config.geno.nodeCost, agent_b.config.geno.nodeCost)
    # take disjoint sensor nodes
    sensors, q_sensors = merge_sensor_nodes_by_name(agent_a, agent_b)
    # take disjoint motor nodes
    motors = merge_motor_nodes_by_name(agent_a, agent_b)
    # merge other nodes
    other_nodes, q_others = merge_other_nodes_by_name(agent_a, agent_b)

    # get the Q tables for actions of nodes (sensors + others)
    q_all = merge_two_dicts(q_sensors, q_others)

    conf['sensors'] = sensors
    conf['motors'] = motors
    conf['others'] = other_nodes
    conf['qtables'] = q_all

    return conf


# merge genotype and network configuration of two agents
# arguments: agent_a (female), agent_b (male)
# return value: agent configuration after cross over operation
def cross_over(agent_a, agent_b):
    # two agents mate
    debug('Function: cross_over')
    # crossover of genotypes
    conf = cross_over_genotype(agent_a, agent_b)
    # crossover of the network configurations
    netconf = cross_over_netconf(agent_a, agent_b)
    conf['network'] = netconf

    return conf


# mutate the list of other nodes (AND, SEQ etc)
# today we do not remove sensors or motors?
# arguments: agent configuration, list of other nodes (AND, SEQ etc)
# return value: agent configuration and list of other nodes after mutation
def mutate(conf):
    # agent's genome mutate
    debug('Function: mutate')
    # remove other nodes (AND, SEQ etc)
    loop_i = 0
    netconf = conf.get("network", {})
    objectives = list(conf.get("objectives", {}).keys())
    q_all = netconf.get("qtables", {})
    other_nodes_original = netconf.get("others", [])
    other_nodes = other_nodes_original.copy()

    # find top nodes
    top_nodes = [n for n in other_nodes if n[2]==[]]
    # randomly select elements to remove according to mutation rate, here 50%.
    for n in top_nodes:
        if random.random() < 0.2:
            continue
        if random.random() < 0.5:
            other_nodes = remove_node(other_nodes, n)
        else:
            # change the gate at 50%
            name, i, o = n
            if name.startswith("AND"):
                new_name = "SEQ"+name[3:]
            elif name.startswith("SEQ"):
                new_name = "AND"+name[3:]
            else:
                continue
            # update the entris in other nodes and qtable
            v1 = q_all[name]
            v2 = q_all.get(new_name, None)
            if v2 == None:
                # update the other_nodes
                other_nodes = remove_node(other_nodes, n)
                top_nodes.remove(n)
                other_nodes = add_node(other_nodes, (new_name, i, o))
                # update the qtable: add new node, remove old node
                q_all[new_name] = v1
                #q_all.pop(name, None)
            else:
                # no need to update other_nodes
                # update the qtables only: switch values of two nodes
                q_all[name] = v2
                q_all[new_name] = v1

    # remove q table of the nodes removed during mutation
    nodes_removed = list(set([n for n,i,o in other_nodes_original]) - set([n for n,i,o in other_nodes]))
    gdebug('mutate: nodes_removed ', nodes_removed)
    #gdebug('mutate: q_all ', q_all.keys())
    for n in nodes_removed:
        q_all.pop(n)
    #gdebug('muate: q_all updated ', q_all.keys())

    # add other nodes using existing sensors
    sensors = ['$' +x for x in netconf.get("sensors")]
    motors = netconf.get('motors', [])

    other_names = [x[0] for x in other_nodes]
    for a, b in itertools.combinations(sensors+other_names, 2):
        if random.random() < 0.1:
            inputs = sorted([a, b])
            if random.random() <= 0.5:
                name = createName('AND', inputs)
            else:
                # SEQ inputs has its own order therefore cannot be sorted
                name = createName('SEQ', inputs, sort=False)
            # check whether the node already exists or not
            if name not in set(other_names):
                # TODO: eliminate redundant gates i.e., g AND (b AND g)
                # need NOT to update 'outputs' of the input nodes of the new node
                # the update of 'outputs' will be done during initialization of agent
                new_node = (name, inputs, [])
                #gdebug('mutate: add new node ', new_node)
                other_nodes = add_node(other_nodes, new_node)
                #gdebug('muate: other_nodes after adding new node ', other_nodes)
                # update the names list
                other_names = [x[0] for x in other_nodes]
                q = {}
                for m in motors:
                    q[m] = {k:0 for k in objectives}
                q_all[name] = q
                #gdebug('mutate: q_all after adding new node ', q_all.keys())

    allnames = [n for n, i, o in other_nodes]+sensors
    # mutate Q tables of actions of the resulted other nodes
    for n in allnames:
        # take the entry in the qtable
        val = q_all.get(n, {})
        if val == {}:
            gdebug('something is not right!', q_all)
        for m in motors:
            q = val.get(m, {k:0 for k in objectives})
            for k in objectives:
                d = random.uniform(-0.2, 0.2)
                #gdebug('random value: ', d)
                t = q.get(k, 0)
                q[k] = max(t+d, 0)

            val[m] = q
            #gdebug('new q value: ', val[m])
        # assign it back to q_all
        q_all[n] = val

    # update the network configuration
    netconf["others"] = other_nodes
    netconf["qtables"] = q_all
    conf["network"] = netconf
    return conf


# Create an genetic agent
def createGeneticAgent(conf, objectives, gender='neuter'):
    agent = GeneticAgent(conf, createNetwork(conf.network, objectives, conf.seed), objectives, (0,0), gender)
    return agent


class GenoType():
    def __init__(self, conf):
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
        self.others = conf.get("network", {}).get("others", [])
        self.qtables = conf.get("network", {}).get("qtables", {})
        self.nodeCost = conf.get("nodeCost", 0)
        # for hormone
        self.hormoneThresholdTrigger = conf.get("hormoneTriggerThreshold", 3)
        self.hormoneThresholdSecreteLower = conf.get("hormoneThresholdSecreteLower", 1)
        self.hormoneThresholdSecreteUpper = conf.get("hormoneThresholdSecreteUpper", 4)


class GeneticAgentConfig(AgentConfig):
    def __init__(self, conf):
        super().__init__(conf)
        self.learning = conf.get("learning", True)
        # move objectives and initial values into AgentConfig
        self.objectivesWithValues = conf.get("objectives", {})
        self.objectives = list(self.objectivesWithValues.keys())
        # genetic extension
        self.geno = GenoType(conf)

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
        self.hormone = False  # secrete hormone
        self.chosen = False   # a male is chosen by a neighbour female
        self.pregnantAt = -1  # when the female or neuter became pregnant, -1: invalid
        self.offspringsConf = [] # configuration of its offsprings
        self.brainTrail = []  # trail of its brain (dynamic graph)
        # add other nodes
        gdebug('GenericAgent_init: other nodes: ', self.config.geno.others)
        for x in self.config.geno.others:
            self.addNode(x)
        gdebug('GenericAgent_init: other nodes are generated: ', self.network.nodes.keys())
        # assign initial Q values
        for name in self.network.nodes:
            node = self.network.nodes[name]
            q = self.config.geno.qtables.get(name, None)
            if q != None:
                #q = self.config.geno.qtables.get(name, {k:{} for k in self.config.geno.motors})
                #assign Q to each action of the node
                for a in node.actions:
                    t = q.get(a.getName(), {k:0 for k in self.network.objectives})
                    a.Q = t
                    a.minQ = t
                    a.maxQ = t

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

    def is_hormoneOn(self):
        return self.hormone

    def hormoneOn(self):
        self.hormone = True

    def hormoneOff(self):
        self.hormone = False

    def getHormoneThresholdTrigger(self):
        return self.config.geno.hormoneThresholdTrigger

    def getHormoneThresholdLower(self):
        return self.config.geno.hormoneThresholdSecreteLower

    def getHormoneThresholdUpper(self):
        return self.config.geno.hormoneThresholdSecreteUpper

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

    def addNode(self, node):
        n, i, o = node
        inp = []
        outp = []

        gdebug('addNode: node ', node)
        # inputs
        for x in set(i):
            gdebug('addNode: loop starts!')
            p = self.network.nodes.get(x, None)
            if p == None:
                gdebug('addNode: node does not exist ', x)
                # create the node for inputs
                for y in self.config.geno.others:
                    if y[0] == x:
                        gdebug('addNode: find the node ', y)
                        p = self.addNode(y)
                        break
            # add p to the input list
            if p != None:
                gdebug('addNode: add node to inp', p)
                inp.append(p)
            else:
                gdebug('addNode: somthing is wrong!', self.network.nodes.keys())

        # only add existing nodes to output lists
        for y in set(o):
            q = self.network.nodes.get(y, None)
            if q != None:
                outp.append(q)

        gdebug('addNode: inp: ', [x.name for x in inp])
        gdebug('addNode: outp: ', [x.name for x in outp])
        # create current node
        newnode = None
        if n.startswith('AND'):
            gdebug('addNode: AND node')
            newnode = nodes.AndNode(inputs=inp, outputs=outp, virtual=False)
            self.network.addNode(newnode)
        elif n.startswith('SEQ'):
            gdebug('addNode: SEQ node')
            newnode = nodes.SEQNode(inputs=inp, outputs=outp, virtual=False)
            self.network.addNode(newnode)
        else:  # todo: for other kinds of nodes
            gdebug('addNode: unknown node type, wrong!')
            pass

        gdebug('addNode: end ', self.network.nodes.keys())
        return newnode

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
            conf = reproduce(self)
            # mutate()
            conf = mutate(conf)
            agent_conf = GeneticAgentConfig(conf)
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
                    conf = cross_over(self, agent)
                    # mutate()
                    conf = mutate(conf)
                    agent_conf = GeneticAgentConfig(conf)
                    self.offspringsConf.append(agent_conf)
                    loop_i = loop_i + 1
                self.become_pregnant()
                agent.become_chosen()
            else: # self: male, agent: female
                gdebug('male agent mates with female agent!')
                while loop_i < offspringNumber:
                    conf = cross_over(agent, self)
                    # mutate()
                    conf = mutate(conf)
                    agent_conf = GeneticAgentConfig(conf)
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

                offsprings.append(child)
                loop_i = loop_i + 1

            if self.is_neuter():
                gender = 'neuter'
                gdebug(gender + ' agent deliver ' + str(loop_i) + ' offsprings!')
            else:
                gender = 'female'
                gdebug(gender + ' agent deliver ' + str(loop_i) + ' offsprings!')

        return offsprings

    def takeAction(self, arg, reward):

        action, prediction, numPredictions = arg

        # should check sensor!
        cell = [x.name for x in self.network.activeSensors()] # self.environment.currentCell(self)

        #debug("KNOWN ACTIONS:", self.network.knownActions(need))

        if not action: return

        surprise = relative_surprise(prediction, reward)
        debug("takeAction >>> - best action:", action, ", surprise:", surprise, ", numPredictions:", numPredictions, ", prediction:", prediction, ", reward:", reward, ", cell:", cell, ", action:", action,)

        self.trail.append( (cell, action) )
        self.wellbeeingTrail.append( self.wellbeeing() )
        self._beginLearning(surprise, reward, action, prediction, numPredictions)

        # update status vector
        # record current brain
        brain = self.network.nodes.keys()
        self.brainTrail.append(brain)
        # cost according to the size of its brain
        nodeAmout = self.network.node_count
        delta = nodeAmout*self.config.geno.nodeCost
        if delta != 0:
            for o in self.config.objectives:
                reward[o] = reward[o] - delta
                debug('Function: takeAction - deduce nodes cost!')

        self._updateNeeds(reward)


class GeneticAnimalAgent(GeneticAgent):
    def __init__(self, config, network, needs=None, position=(0, 0), gender='sexual', growthRate=0.01):
        super().__init__(config, network, needs, position, gender, growthRate)
        self.prey = None # the prey being locked by this agent


class GeneticPlantAgent(GeneticAgent):
    def is_grazed(self):
        pass
