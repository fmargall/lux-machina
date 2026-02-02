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
    nbParallelRays  = 100_000
    maximumRayDepth = 2
    timeSleep       = 0.

    # 0.(i) Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 2 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.01796, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.01796, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesBuffer = wp.array([lightSource], dtype=LightSource3D, ndim=1)

    # 0.(ii) Primitives initialization
    primitivesList = []
    
    # Aspheric lens
    nI = 1.0
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

    asphericLens11 = Primitive3D()
    asphericLens11.type = 3 # Aspheric lens
    asphericLens11.v0 = wp.vec3(0., 0., 0.0042) # Origin of   axis
    asphericLens11.v1 = wp.vec3(0., 0., 0.01270) # Normal of z-axis
    asphericLens11.f0  = wp.float32(8.818197) # Radius
    asphericLens11.f1  = wp.float32(-0.9991715) # Conic constant
    asphericLens11.f2  = wp.float32(nI) # Refractive index in the +normal direction
    asphericLens11.f3  = wp.float32(nO) # Refractive index in the -normal direction
    asphericLens11.f4  = wp.float32(11.6383) # y-intercept
    asphericLens11.f5  = wp.float32(12.8155) # normalization factor
    asphericLens11.f6  = wp.float32(0.) # 1st coefficient, associated to  2nd power
    asphericLens11.f7  = wp.float32(8.682167e-5) # 2nd coefficient, associated to  4th power
    asphericLens11.f8  = wp.float32(6.3760123e-8) # 3rd coefficient, associated to  6th power
    asphericLens11.f9  = wp.float32(2.4073084e-9) # 4th coefficient, associated to  8th power
    asphericLens11.f10 = wp.float32(-1.7189021e-11) # 5th coefficient, associated to 10th power

    # Biconvex lens
    biconvexLens00 = Primitive3D()
    biconvexLens00.type = 2 # First surface of biconvex lens is a spherical cap
    biconvexLens00.v0 = wp.vec3(0., 0.,  0.0984) # Center of sphere
    biconvexLens00.v1 = wp.vec3(0., 0., -0.0592) # Direction of spherical cap pole, with radius as length
    biconvexLens00.f0 = wp.float32(0.447189) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    biconvexLens00.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    biconvexLens00.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.

    biconvexLens01 = Primitive3D()
    biconvexLens01.type = 1 # Second surface of biconvex lens is a cylinder
    biconvexLens01.v0 = wp.vec3(0., 0., 0.04490) # Center of basis
    biconvexLens01.v1 = wp.vec3(0., 0., 0.003) # Normal of basis with height as length
    biconvexLens01.f0 = wp.float32(0.02540)  # Radius
    biconvexLens01.f1 = wp.float32(nI)     # Refractive index in the +normal direction, ie. outside
    biconvexLens01.f2 = wp.float32(nO)     # Refractive index in the -normal direction, ie. inside.

    biconvexLens11 = Primitive3D()
    biconvexLens11.type = 2 # third surface of biconvex lens is a spherical cap
    biconvexLens11.v0 = wp.vec3(0., 0.,  0.0984) # Center of sphere
    biconvexLens11.v1 = wp.vec3(0., 0., +0.0592) # Direction of spherical cap pole, with radius as length
    biconvexLens11.f0 = wp.float32(0.447189) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    biconvexLens11.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    biconvexLens11.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.

    # Retaining rings
    retainingRing0 = Primitive3D()
    retainingRing0.type = 0 # Annulus blocker
    retainingRing0.v0 = wp.vec3(0., 0., 0.04380) # Origin
    retainingRing0.v1 = wp.vec3(0., 0., 1.) # Normal
    retainingRing0.f0 = wp.float32(0.02290) # Inner radius
    retainingRing0.f1 = wp.float32(0.02540) # Outer radius

    retainingRing1 = Primitive3D()
    retainingRing1.type = 0 # Annulus blocker
    retainingRing1.v0 = wp.vec3(0., 0., 0.04900) # Origin
    retainingRing1.v1 = wp.vec3(0., 0., 1.) # Normal
    retainingRing1.f0 = wp.float32(0.02290) # Inner radius
    retainingRing1.f1 = wp.float32(0.02540) # Outer radius

    primitivesList.append(asphericLens00)
    primitivesList.append(asphericLens01)
    primitivesList.append(asphericLens11)

    #primitivesList.append(biconvexLens00)
    #primitivesList.append(biconvexLens01)
    #primitivesList.append(biconvexLens11)

    #primitivesList.append(retainingRing0)
    #primitivesList.append(retainingRing1)

    primitivesBuffer = wp.array(primitivesList, dtype=Primitive3D, ndim=1)
    nbPrimitives     = wp.int32(len(primitivesList))

    # 0.(iii) Sensors initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0 , 0.0, 0.019) # Center (0.0536 for direct output after last primitive / 0.0653 for output after last mechanical support)
    sensor.v1 = wp.vec3(0.0 , 0.026, 0.0   ) # Tangent (0.0508)
    sensor.v2 = wp.vec3(0.026, 0.0 , 0.0  ) # Bitangent
    sensor.i0 = wp.int32(256) # Number of pixels on the tangential axis

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

            # III.(i) Accumulate data on sensor and display it
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

            # Redraw minimal
            ax.draw_artist(imgPlot)
            fig.canvas.blit(ax.bbox)
            fig.canvas.flush_events()

            # II.(ii) Rasterizing in 2D is too useful for debugging
            wp.launch(
                kernel  = rasterize3D,
                dim     = rasterizerBuffer.shape,
                inputs  = [intersectionsBuffer, nbParallelRays, rasterizer],
                outputs = [rasterizerBuffer]
            )

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