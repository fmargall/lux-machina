import time

import cv2
import matplotlib.pyplot as plt
import numpy             as np
import warp              as wp

from generatePrimaryRays import generatePrimary3DRays
from intersections       import intersect3DRays
from propagations        import propagate3DRays

from structures import LightSource3D, Sensor3D

if __name__ == "__main__":
    # GPU support using NVIDIA Warp
    wp.init()

    # 0. Light tracer parameters
    nbParallelRays  = 10_000
    maximumRayDepth = 10
    timeSleep       = 0.

    # 0.(i) Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 2 # Parallelogram collimated light source
    lightSource.v0 = wp.vec3(0., 0., 0.) # Center
    lightSource.v1 = wp.vec3(1., 0., 0.) # Tangent
    lightSource.v2 = wp.vec3(0., 1., 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    # 0.(ii) Primitives initialization

    # 0.(iii) Sensors initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0, 0.0, 1.) # Center
    sensor.v1 = wp.vec3(0.0, 1.5, 0.) # Tangent
    sensor.v2 = wp.vec3(1.5, 0.0, 0.) # Bitangent
    sensor.i0 = wp.int32(512) # Number of pixels on the tangential axis

    # Initializing sensor buffer
    sensorPixelsBitangent = sensor.i0 * wp.norm_l2(sensor.v2) / wp.norm_l2(sensor.v1)
    sensorBuffer = wp.zeros((sensor.i0, sensorPixelsBitangent), dtype=wp.float32)