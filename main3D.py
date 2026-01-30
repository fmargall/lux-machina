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
    lightSource.type = 1 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.003, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.003, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesBuffer = wp.array([lightSource], dtype=LightSource3D, ndim=1)

    # 0.(ii) Primitives initialization
    primitivesList = []
    
    # Aspheric lens
    nI = 1.
    nO = 1.51
    asphericLens00 = Primitive3D()
    asphericLens00.type = 2 # First surface of aspheric lens is a spherical cap
    asphericLens00.v0 = wp.vec3(0., 0.,  0.0718379) # Center of sphere
    asphericLens00.v1 = wp.vec3(0., 0., -0.06999948) # Direction of spherical cap pole, with radius as length
    asphericLens00.f0 = wp.float32(0.182440) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    asphericLens00.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    asphericLens00.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.

    asphericLens01 = Primitive3D()
    asphericLens01.type = 1 # Second surface of aspheric lens is a cylinder
    asphericLens01.v0 = wp.vec3(0., 0., 0.0030) # Center of basis
    asphericLens01.v1 = wp.vec3(0., 0., 0.0012) # Normal of basis with height as length
    asphericLens01.f0 = wp.float32(0.01270)  # Radius
    asphericLens01.f1 = wp.float32(nI)     # Refractive index in the +normal direction, ie. outside
    asphericLens01.f2 = wp.float32(nO)     # Refractive index in the -normal direction, ie. inside.

    """
    asphericLens11 = Primitive3D()
    asphericLens11.type = 3 # Aspheric lens
    asphericLens11.v0 = wp.vec3(0., 0., 0.) # Origin of   axis
    asphericLens11.v1 = wp.vec3(0., 0., 0.) # Normal of z-axis
    asphericLens11.f0  = wp.float32(0.) # Radius
    asphericLens11.f1  = wp.float32(0.) # Conic constant
    asphericLens11.f2  = wp.float32(nI) # Refractive index in the +normal direction
    asphericLens11.f3  = wp.float32(nO) # Refractive index in the -normal direction
    asphericLens11.f4  = wp.float32(0.) # y-intercept
    asphericLens11.f5  = wp.float32(0.) # normalization factor
    asphericLens11.f6  = wp.float32(0.) # 1st coefficient, associated to  2nd power
    asphericLens11.f7  = wp.float32(0.) # 2nd coefficient, associated to  4th power
    asphericLens11.f8  = wp.float32(0.) # 3rd coefficient, associated to  6th power
    asphericLens11.f9  = wp.float32(0.) # 4th coefficient, associated to  8th power
    asphericLens11.f10 = wp.float32(0.) # 5th coefficient, associated to 10th power
    """

    # Biconvex lens
    biconvexLens00 = Primitive3D()
    biconvexLens00.type = 2 # First surface of biconvex lens is a spherical cap
    biconvexLens00.v0 = wp.vec3(0., 0.,  0.076) # Center of sphere
    biconvexLens00.v1 = wp.vec3(0., 0., -0.05) # Direction of spherical cap pole, with radius as length
    biconvexLens00.f0 = wp.float32(0.1) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    biconvexLens00.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    biconvexLens00.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.

    biconvexLens01 = Primitive3D()
    biconvexLens01.type = 1 # Second surface of biconvex lens is a cylinder
    biconvexLens01.v0 = wp.vec3(0., 0., 0.0030) # Center of basis
    biconvexLens01.v1 = wp.vec3(0., 0., 0.0012) # Normal of basis with height as length
    biconvexLens01.f0 = wp.float32(0.02540)  # Radius
    biconvexLens01.f1 = wp.float32(nI)     # Refractive index in the +normal direction, ie. outside
    biconvexLens01.f2 = wp.float32(nO)     # Refractive index in the -normal direction, ie. inside.

    biconvexLens11 = Primitive3D()
    biconvexLens11.type = 2 # third surface of biconvex lens is a spherical cap
    biconvexLens11.v0 = wp.vec3(0., 0.,  0.076) # Center of sphere
    biconvexLens11.v1 = wp.vec3(0., 0., -0.05) # Direction of spherical cap pole, with radius as length
    biconvexLens11.f0 = wp.float32(0.1) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    biconvexLens11.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    biconvexLens11.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.


    primitivesList.append(asphericLens00)
    primitivesList.append(asphericLens01)

    primitivesBuffer = wp.array(primitivesList, dtype=Primitive3D, ndim=1)
    nbPrimitives     = wp.int32(len(primitivesList))

    # 0.(iii) Sensors initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0   , 0.0   , 0.005) # Center (Normally: 0.06530)
    sensor.v1 = wp.vec3(0.0   , 0.0508, 0.0  ) # Tangent
    sensor.v2 = wp.vec3(0.0508, 0.0   , 0.0  ) # Bitangent
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

        time.sleep(timeSleep)

        # The second while loop will run starting from initial
        # light source rays and everytime checks intersection,
        # then rasterize and accumulate, checks dead rays, and
        # continue or close the loop
        longestRayDepth = 0
        while True:

            # II. Compute intersections of rays with scene
            raysStatusBuffer    = wp.zeros((nbParallelRays,), dtype=wp.bool)
            intersectionsBuffer = wp.zeros((nbParallelRays,), dtype=Intersection3D)

            wp.launch(
                kernel  = intersect3DRays,
                dim     = nbParallelRays,
                inputs  = [raysBuffer, primitivesBuffer, nbPrimitives],
                outputs = [intersectionsBuffer, raysStatusBuffer]
            )

            onlyDeadRays = np.sum(raysStatusBuffer.numpy()) == 0

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

            # IV. If all rays are dead, we can break the loop
            # We can also break if we have reached max depth.
            if onlyDeadRays or (longestRayDepth >= maximumRayDepth):
                break

            # V. Otherwise, we continue propagating the rays
            wp.launch(
                kernel  = propagate3DRays,
                dim     = nbParallelRays,
                inputs  = [intersectionsBuffer, frameID],
                outputs = [raysBuffer]
            )

            longestRayDepth += 1

            time.sleep(timeSleep)