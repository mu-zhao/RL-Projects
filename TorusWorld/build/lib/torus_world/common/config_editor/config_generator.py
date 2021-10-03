import json
import os
import logging
import argparse
from collections import defaultdict

import numpy as np


logger = logging.getLogger(__name__)

_BASE_DIR = "TorusWorld/data/configs"

def _random_generate_config(args):
    config = {}
    