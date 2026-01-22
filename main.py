import time

import cv2
import matplotlib.pyplot as plt
import numpy             as np
import warp              as wp

from generatePrimaryRays import generatePrimaryRays
from intersections       import intersectRays
from propagations        import propagateRays

from accumulateSensor import accumulateIdealSensor, accumulateShackHartmannIdealSensor

from rasterizer import rasterize

from structures import Intersection, LightSource, Primitive, Ray, Sensor

"""
   Circular mean with ponderation
"""
def circmean(oldWeight, oldAngle, newWeight, newAngle):
    return np.angle(oldWeight * np.exp(1j * oldAngle) + newWeight * np.exp(1j * newAngle))

"""
   Converts world coordinates to screen coordinates.
"""
def worldCoordinatesToScreenCoordinates(
    lightSourcesList,#: list[LightSource], # Type hint only valid in Python 3.9+
    primitivesList  ,#: list[Primitive]  , # Type hint only valid in Python 3.9+,
    sensorsList     ,#: list[Sensor]     , # Type hint only valid in Python 3.9+,

    pixelSizeInWorldUnits: wp.float32       ,
    bottomLeftCornerWorldCoordinates: wp.vec2

): # -> tuple[wp.array(dtype=LightSource, ndim=1),  # Type hint only valid in Python 3.9+
   #          wp.array(dtype=Primitive  , ndim=1)]: # Type hint only valid in Python 3.9+

    # Transforming light sources
    for lightSource in lightSourcesList:
        if   lightSource.type == 0: # Point light source
            lightSource.v0 = (lightSource.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        elif lightSource.type == 1: # Lambertian source
            lightSource.v0 = (lightSource.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            lightSource.v1 = (lightSource.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        else:
            raise ValueError(f"Unknown lightsource type: {lightSource.type} in scene coordinates conversion.")

    # Transforming primitives
    for primitive in primitivesList:
        if   primitive.type == -1: # Bounding box
            pass # No modification required
        elif primitive.type == 0:  # Ideal lens
            primitive.v0 = (primitive.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.v1 = (primitive.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.f0 =  primitive.f0 / pixelSizeInWorldUnits # Focal length
        elif primitive.type == 1:  # Segment
            primitive.v0 = (primitive.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.v1 = (primitive.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        elif primitive.type == 2:  # Circular arc
            primitive.v0 = (primitive.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.f0 =  primitive.f0 / pixelSizeInWorldUnits # Radius
        elif primitive.type == 3:  # Aspheric lens
            primitive.v0 = (primitive.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.v1 = (primitive.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        elif primitive.type == 4:  # Blocker
            primitive.v0 = (primitive.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
            primitive.v1 = (primitive.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        else:
            raise ValueError(f"Unknown primitive type: {primitive.type} in scene coordinates conversion.")

    # Transforming sensors
    for sensorID, sensor in enumerate(sensorsList):
        sensor.v0 = (sensor.v0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        sensor.v1 = (sensor.v1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        sensorsList[sensorID] = sensor

    return (wp.array(lightSourcesList, dtype=LightSource, ndim=1),
            wp.array(primitivesList  , dtype=Primitive  , ndim=1),
            sensorsList)

if __name__ == "__main__":
    # GPU support using NVIDIA Warp
    wp.init()

    # 0. Light tracer parameters
    width, height   = 512, 512
    nbParallelRays  = 10_000
    maximumRayDepth = 10
    timeSleep       = 0.
    
    # 0.(i) Light sources initialisation
    lightSourceLED = LightSource()
    lightSourceLED.type = 1
    lightSourceLED.f0   = 1.0
    lightSourceLED.v0   = wp.vec2( 0.0,   0.0015)
    lightSourceLED.v1   = wp.vec2( 0.0,  -0.0015)
    lightSourcesList = [lightSourceLED]
    
    # 0.(ii) Scene primitives initialisation
    primitivesList = []

    # Bounding box (to reveal all rays)
    bbox = Primitive()
    bbox.type = -1
    bbox.f0   = width / height

    #primitivesList.append(bbox)

    # Aspheric lens
    nIOR = 1.51
    asphericLens00 = Primitive()
    asphericLens00.type = 2
    asphericLens00.v0   = wp.vec2(0.0718379, 0.)
    asphericLens00.f0   = wp.float32(0.06999948)
    asphericLens00.f1   = wp.float32(2.959153)
    asphericLens00.f2   = wp.float32(3.324033)
    asphericLens00.f3   = wp.float32(nIOR)
    asphericLens00.f4   = wp.float32(1.0)

    asphericLens01 = Primitive()
    asphericLens01.type = 1
    asphericLens01.v0   = wp.vec2(0.0030, 0.01270)
    asphericLens01.v1   = wp.vec2(0.0042, 0.01270)
    asphericLens01.f0   = wp.float32(nIOR)
    asphericLens01.f1   = wp.float32(1.)

    asphericLens10 = Primitive()
    asphericLens10.type = 1
    asphericLens10.v0   = wp.vec2(0.0030, -0.01270)
    asphericLens10.v1   = wp.vec2(0.0042, -0.01270)
    asphericLens10.f0   = wp.float32(1.)
    asphericLens10.f1   = wp.float32(nIOR)

    asphericLens11 = Primitive()
    asphericLens11.type = 3
    asphericLens11.v0   = wp.vec2(0.0042,  0.01270)
    asphericLens11.v1   = wp.vec2(0.0042, -0.01270)
    asphericLens11.f0   = wp.float32( 8.818197)
    asphericLens11.f1   = wp.float32(-0.9991715)
    asphericLens11.f2   = wp.float32(nIOR)
    asphericLens11.f3   = wp.float32(1.0)
    asphericLens11.f4   = wp.float32(11.6383)
    asphericLens11.f5   = wp.float32(12.8155)
    asphericLens11.f6   = wp.float32(0.)
    asphericLens11.f7   = wp.float32(8.682167e-5)
    asphericLens11.f8   = wp.float32(6.3760123e-8)
    asphericLens11.f9   = wp.float32(2.4073084e-9)
    asphericLens11.f10  = wp.float32(-1.7189021e-11)

    primitivesList.append(asphericLens00)
    primitivesList.append(asphericLens01)
    primitivesList.append(asphericLens10)
    primitivesList.append(asphericLens11)

    # Biconvex lens
    biconvex00 = Primitive()
    biconvex00.type = 2
    biconvex00.v0   = wp.vec2(0.0983741, 0.0)
    biconvex00.f0   = wp.float32(0.0592)
    biconvex00.f1   = wp.float32(2.698148)
    biconvex00.f2   = wp.float32(3.585038)
    biconvex00.f3   = wp.float32(1.51)
    biconvex00.f4   = wp.float32(1.0)
    
    biconvex01 = Primitive()
    biconvex01.type = 1
    biconvex01.v0   = wp.vec2(0.04490, 0.02540)
    biconvex01.v1   = wp.vec2(0.04790, 0.02540)
    biconvex01.f0   = wp.float32(1.51)
    biconvex01.f1   = wp.float32(1.)

    biconvex10 = Primitive()
    biconvex10.type = 1
    biconvex10.v0   = wp.vec2(0.04490, -0.02540)
    biconvex10.v1   = wp.vec2(0.04790, -0.02540)
    biconvex10.f0   = wp.float32(1.)
    biconvex10.f1   = wp.float32(1.51)
    
    biconvex11 = Primitive()
    biconvex11.type = 2
    biconvex11.v0   = wp.vec2(-0.0055741, 0.0)
    biconvex11.f0   = wp.float32(0.0592)
    biconvex11.f1   = wp.float32(5.839740)
    biconvex11.f2   = wp.float32(0.443445)
    biconvex11.f3   = wp.float32(1.51)
    biconvex11.f4   = wp.float32(1.0)

    primitivesList.append(biconvex00)
    primitivesList.append(biconvex01)
    primitivesList.append(biconvex10)
    primitivesList.append(biconvex11)

    # Blockers
    blocker0 = Primitive()
    blocker0.type = 4
    blocker0.v0   = wp.vec2(0.04380, -0.02540)
    blocker0.v1   = wp.vec2(0.04380, -0.02290)

    blocker1 = Primitive()
    blocker1.type = 4
    blocker1.v0   = wp.vec2(0.04380,  0.02540)
    blocker1.v1   = wp.vec2(0.04380,  0.02290)

    blocker2 = Primitive()
    blocker2.type = 4
    blocker2.v0   = wp.vec2(0.04900, -0.02540)
    blocker2.v1   = wp.vec2(0.04900, -0.02290)

    blocker3 = Primitive()
    blocker3.type = 4
    blocker3.v0   = wp.vec2(0.04900,  0.02540)
    blocker3.v1   = wp.vec2(0.04900,  0.02290)

    primitivesList.append(blocker0)
    primitivesList.append(blocker1)
    primitivesList.append(blocker2)
    primitivesList.append(blocker3)

    nbPrimitives = len(primitivesList)

    # 0.(iii) Sensors initialization
    sensorsList = []

    sensor0 = Sensor()
    sensor0.type = 0
    sensor0.v0   = wp.vec2(1.75,  0.465)
    sensor0.v1   = wp.vec2(1.75, -0.465)
    sensor0.i0   = 512

    sensorsList.append(sensor0)

    sensor0Buffer = wp.zeros((sensor0.i0,), dtype=wp.float32)
    sensor0Data  = sensor0Buffer.numpy()

    # Initializing sensors buffer
    sensorsBufferList = []
    sensorsBufferList.append(sensor0Buffer)
    sensorsDataList = []
    sensorsDataList.append(sensor0Data)

    # Initializing sensors figure
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(0, sensor0.i0)
    ax.set_ylim(0 ,1)
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)
    plt.show(block=False)

    nbSensors = len(sensorsList)

    # 0.(iv) Converting to screen coordinates
    bottomLeftCornerWorldCoordinates = wp.vec2(0.005, -.0275)
    pixelSizeInWorldUnits = wp.float32(0.055)
    lightSourcesBuffer, primitivesBuffer, sensorsList = worldCoordinatesToScreenCoordinates(
        lightSourcesList, primitivesList, sensorsList,
        pixelSizeInWorldUnits, bottomLeftCornerWorldCoordinates
    )

    # Initialising image buffer
    imageBuffer = wp.zeros((height, width), dtype=wp.float32)
    img = imageBuffer.numpy()

    # The first infinite while loop concerns each 
    # initial ray generation, directly from light
    # sources
    previousTime = time.time()
    nbIterations = 0
    while True:
        nbIterations += nbParallelRays
        frameID = nbIterations // nbParallelRays

        # I. Generating primary rays from light sources
        raysBuffer = wp.zeros(nbParallelRays, dtype=Ray)
        
        wp.launch(
            kernel  = generatePrimaryRays,
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
        breakRun = False # To break outer loop if is required.
        while True:
            # II. Checking ray-scene intersections
            raysStatusBuffer    = wp.zeros((nbParallelRays,), dtype=wp.bool)
            intersectionsBuffer = wp.zeros((nbParallelRays,), dtype=Intersection)
            
            wp.launch(
                kernel  = intersectRays,
                dim     = nbParallelRays,
                inputs  = [raysBuffer, primitivesBuffer, nbPrimitives],
                outputs = [intersectionsBuffer, raysStatusBuffer]
            )
            
            onlyDeadRays = np.sum(raysStatusBuffer.numpy()) == 0

            # III. Rasterizing and accumulating to image buffer
            wp.launch(
                kernel  = rasterize,
                dim     = (height, width),
                inputs  = [intersectionsBuffer, nbParallelRays],
                outputs = [imageBuffer]
            )
            
            # III.(ii) Accumulating and normalizing the image buffer
            img = (img * (nbIterations - nbParallelRays) + imageBuffer.numpy()) / nbIterations

            # III.(iii) Once accumulated, we can display the current image

            # Computing current RPS
            currentTime = time.time()
            rps = nbParallelRays / (currentTime - previousTime)
            previousTime = currentTime

            # Show everything on screen
            imgNorm = img / img.max()
            cv2.putText(imgNorm, f"RPS: {rps:.2f}"      , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(imgNorm, f"Rays: {nbIterations}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("FiatLux!", imgNorm)
            
            # Check for quitting the application
            # CAUTION: This block NEEDS to be be
            # called BEFORE any break statement!
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                breakRun = True
                break

            # III.(iv) Plot results for each sensor
            for sensorID, sensor in enumerate(sensorsList):

                if (sensor.type == 0): # Ideal sensor
                    sensorBuffer = sensorsBufferList[sensorID]
                    sensorData = sensorsDataList[sensorID]

                    wp.launch(
                        kernel  = accumulateIdealSensor,
                        dim     = sensor.i0,
                        inputs  = [intersectionsBuffer, nbParallelRays, sensor],
                        outputs = [sensorBuffer]
                    )

                    sensorData = (sensorData * (nbIterations - nbParallelRays) + sensorBuffer.numpy()) / nbIterations

                    sensorsDataList[sensorID] = sensorData

                    y = sensorData / np.max(sensorData)
                    x = np.arange(len(y))

                    fig.canvas.restore_region(background)

                    # update des données uniquement
                    line.set_data(x, y)

                    # redraw minimal
                    ax.draw_artist(line)
                    fig.canvas.blit(ax.bbox)
                    fig.canvas.flush_events()
                
                elif (sensor.type == 1): # Shack-Hartmann sensor
                    sensorCountBuffer = sensorsBufferList[sensorID][0]
                    sensorAngleBuffer = sensorsBufferList[sensorID][1]

                    sensorCountData = sensorsDataList[sensorID][0]
                    sensorAngleData = sensorsDataList[sensorID][1]

                    wp.launch(
                        kernel  = accumulateShackHartmannIdealSensor,
                        dim     = sensor.i0,
                        inputs  = [intersectionsBuffer, nbParallelRays, sensor],
                        outputs = [sensorCountBuffer, sensorAngleBuffer]
                    )

                    sensorCountData = (sensorCountData * (nbIterations - nbParallelRays) + sensorCountBuffer.numpy()) / nbIterations
                    sensorAngleData = circmean((nbIterations - nbParallelRays) / nbIterations, sensorAngleData,
                                                                            1. / nbIterations, sensorAngleBuffer.numpy())

                    sensorsDataList[sensorID][0] = sensorCountData
                    sensorsDataList[sensorID][1] = sensorAngleData

            # IV. If all rays are dead, we can break the loop
            # We can also break if we have reached max depth.
            if onlyDeadRays or (longestRayDepth >= maximumRayDepth):
                break
            
            # V. Otherwise, we continue propagating the rays
            wp.launch(
                kernel  = propagateRays,
                dim     = nbParallelRays,
                inputs  = [intersectionsBuffer, frameID],
                outputs = [raysBuffer]
            )
            
            longestRayDepth += 1

            time.sleep(timeSleep)

        if breakRun: 
            break