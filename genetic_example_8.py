# Copyright (C) 2017  Wen Xu, Claes Strannegård
#
# [Using Google Style Guide](https://google.github.io/styleguide/pyguide.html)
#
#  a wolf animal learns to drink when level of water is low and to bite sheep when level of energy is low
#  configuration file: ./test/genetic-example-7.json
#

import os
import datetime
import argparse

import animats.main
import animats.animat.genetic.genetic_example_8

parser = argparse.ArgumentParser()
parser.add_argument("config", help="animat configuration")
args = parser.parse_args()

(outputPath, outputDir) = animats.animat.genetic.genetic_example_8.getOutputPath()
animats.animat.genetic.genetic_example_8.run(args.config, outputPath, outputDir)
