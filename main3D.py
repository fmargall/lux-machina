import time

import cv2
import matplotlib.pyplot as plt
import numpy             as np
import warp              as wp

from accumulateSensor    import accumulateIdeal3DSensor
from generatePrimaryRays import generatePrimary3DRays
from intersections       import intersect3DRays
from propagations        import propagate3DRays

from structures import Intersection3D, LightSource3D, Primitive3D, Ray3D, Sensor3D

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

    lightSourcesBuffer = wp.array([lightSource], dtype=LightSource3D, ndim=1)

    # 0.(ii) Primitives initialization
    primitivesList   = []
    primitivesBuffer = wp.array(primitivesList, dtype=Primitive3D, ndim=1)
    nbPrimitives     = wp.int32(len(primitivesList))

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
    sensorData   = sensorBuffer.numpy()

    # Initializing sensor figure
    fig, ax = plt.subplots()
    imgPlot = ax.imshow(
        sensorData, 
        origin="lower",
        aspect="equal",
        interpolation="nearest",
        cmap='gray', 
        vmin=0., 
        vmax=1.
    )

    ax.set_title("Sensor")

    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    plt.show(block=False)

    # The first infinite while loop concerns each 
    # initial ray generation, directly from light
    # sources
    nbIterations = 0
    while True:
        nbIterations += nbParallelRays
        frameID = nbIterations // nbParallelRays

        # I. Generating primary rays from light sources
        raysBuffer = wp.zeros(nbParallelRays, dtype=Ray3D)

        wp.launch(
            kernel  = generatePrimary3DRays,
            dim     = nbParallelRays,
            inputs  = [frameID, lightSourcesBuffer],
            outputs = [raysBuffer]
        )

        # II. Compute intersections of rays with scene
        raysStatusBuffer    = wp.zeros((nbParallelRays,), dtype=wp.bool)
        intersectionsBuffer = wp.zeros((nbParallelRays,), dtype=Intersection3D)

        wp.launch(
            kernel  = intersect3DRays,
            dim     = nbParallelRays,
            inputs  = [raysBuffer, primitivesBuffer, nbPrimitives],
            outputs = [intersectionsBuffer, raysStatusBuffer]
        )

        # III. Accumulate data on sensor and display it
        wp.launch(
            kernel  = accumulateIdeal3DSensor,
            dim     = sensorBuffer.shape,
            inputs  = [intersectionsBuffer, nbParallelRays, sensor],
            outputs = [sensorBuffer]
        )

        sensorData     = (sensorData * (nbIterations - nbParallelRays) + sensorBuffer.numpy()) / nbIterations
        sensorDataNorm =  sensorData / np.max(sensorData) if np.max(sensorData) > 0 else sensorData

        fig.canvas.restore_region(background)

        imgPlot.set_data(sensorDataNorm.T)

        # redraw minimal
        ax.draw_artist(imgPlot)
        fig.canvas.blit(ax.bbox)
        fig.canvas.flush_events()