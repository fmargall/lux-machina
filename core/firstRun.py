import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from _structures         import (_Ray, _LightSource, _Sensor, _Intersection)
from _generateRays       import _generateRays
from _accumulateOnSensor import _accumulateOnSensor


wp.init()

