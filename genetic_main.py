# Copyright (C) 2017  Wen Xu, Claes Strannegård
#
# [Using Google Style Guide](https://google.github.io/styleguide/pyguide.html)

import os
import datetime
import argparse

import animats.main
import animats.animat.genetic.genetic_main

parser = argparse.ArgumentParser()
parser.add_argument("config", help="animat configuration")
args = parser.parse_args()

(outputPath, outputDir) = animats.animat.genetic.genetic_main.getOutputPath()
animats.animat.genetic.genetic_main.run(args.config, outputPath, outputDir)
