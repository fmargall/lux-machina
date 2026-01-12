import time

import cv2
import numpy as np
import warp  as wp

from generatePrimaryRays import generatePrimaryRays
from intersections       import intersectRays

from rasterizer import rasterize

from structures import Intersection, LightSource, Primitive, Ray

"""
   Converts world coordinates to screen coordinates.
"""
def worldCoordinatesToScreenCoordinates(
    lightSourcesList,#: list[LightSource], # Type hint only valid in Python 3.9+
    primitivesList  ,#: list[Primitive]  , # Type hint only valid in Python 3.9+

    pixelSizeInWorldUnits: wp.float32       ,
    bottomLeftCornerWorldCoordinates: wp.vec2

): # -> tuple[wp.array(dtype=LightSource, ndim=1),  # Type hint only valid in Python 3.9+
   #          wp.array(dtype=Primitive  , ndim=1)]: # Type hint only valid in Python 3.9+

    # Transforming light sources
    for lightSource in lightSourcesList:
        lightSource.p0 = (lightSource.p0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        lightSource.p1 = (lightSource.p1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits

    # Transforming primitives
    for primitive in primitivesList:
        primitive.p0 = (primitive.p0 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits
        primitive.p1 = (primitive.p1 - bottomLeftCornerWorldCoordinates) / pixelSizeInWorldUnits

    return (wp.array(lightSourcesList, dtype=LightSource, ndim=1),
            wp.array(primitivesList  , dtype=Primitive  , ndim=1))

if __name__ == "__main__":
    # GPU support using NVIDIA Warp
    wp.init()

    # 0. Light tracer parameters
    width, height   = 720, 360
    nbParallelRays  = 1
    maximumRayDepth = 1
    
    # 0.(i) Light sources initialisation
    lightSourceLED = LightSource()
    lightSourceLED.type      = 1
    lightSourceLED.intensity = 1.0
    lightSourceLED.p0        = wp.vec2( 0.0,   0.0015)
    lightSourceLED.p1        = wp.vec2( 0.0,  -0.0015)
    lightSourcesList = [lightSourceLED]
    
    # 0.(ii) Scene primitives initialisation
    primitivesList = []

    thinFilmIOR = 1.6
    thinFilmSegment00 = Primitive()
    thinFilmSegment00.type = 0
    thinFilmSegment00.p0   = wp.vec2(0.003, -0.003)
    thinFilmSegment00.p1   = wp.vec2(0.003,  0.003)
    thinFilmSegment00.ni   = 1.0
    thinFilmSegment00.no   = thinFilmIOR

    thinFilmSegment01 = Primitive()
    thinFilmSegment01.type = 0
    thinFilmSegment01.p0   = wp.vec2(0.003 , 0.003)
    thinFilmSegment01.p1   = wp.vec2(0.0035, 0.003)
    thinFilmSegment01.ni   = 1.0
    thinFilmSegment01.no   = thinFilmIOR

    thinFilmSegment10 = Primitive()
    thinFilmSegment10.type = 0
    thinFilmSegment10.p0   = wp.vec2(0.003 , -0.003)
    thinFilmSegment10.p1   = wp.vec2(0.0035, -0.003)
    thinFilmSegment10.ni   = thinFilmIOR
    thinFilmSegment10.no   = 1.0

    thinFilmSegment11 = Primitive()
    thinFilmSegment11.type = 0
    thinFilmSegment11.p0   = wp.vec2(0.0035, -0.003)
    thinFilmSegment11.p1   = wp.vec2(0.0035,  0.003)
    thinFilmSegment11.ni   = thinFilmIOR
    thinFilmSegment11.no   = 1.0

    primitivesList.append(thinFilmSegment00)
    primitivesList.append(thinFilmSegment01)
    primitivesList.append(thinFilmSegment10)
    primitivesList.append(thinFilmSegment11)

    nbPrimitives = len(primitivesList)

    # 0.(iii) Converting to screen coordinates
    bottomLeftCornerWorldCoordinates = wp.vec2(0.0, -.005)
    pixelSizeInWorldUnits = wp.float32(0.01)
    lightSourcesBuffer, primitivesBuffer = worldCoordinatesToScreenCoordinates(
        lightSourcesList, primitivesList, 
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
        
        # The second while loop will run starting from initial
        # light source rays and everytime checks intersection,
        # then rasterize and accumulate, checks dead rays, and
        # continue or close the loop
        longestRayDepth = 1
        breakRun = False # To break outer loop if is required.
        while True:
            # II. Checking ray-scene intersections
            raysStatusBuffer    = wp.zeros(1, dtype=wp.bool)
            intersectionsBuffer = wp.zeros((nbParallelRays,), dtype=Intersection)
            
            wp.launch(
                kernel  = intersectRays,
                dim     = nbParallelRays,
                inputs  = [raysBuffer, primitivesBuffer, nbPrimitives],
                outputs = [intersectionsBuffer, raysStatusBuffer]
            )
            
            onlyDeadRays = not raysStatusBuffer.numpy()[0]

            # III. Rasterizing and accumulating to image buffer
            
            wp.launch(
                kernel  = rasterize,
                dim     = (height, width),
                inputs  = [raysBuffer, intersectionsBuffer, nbParallelRays, 
                           height, width],
                outputs = [imageBuffer]
            )
            
            # III.(ii) Accumulating and normalizing the image buffer
            img = (img * (nbIterations - nbParallelRays) + imageBuffer.numpy()) / nbIterations

            # III.(iii) Once accumulated, we can display the current imageq

            # Computing current RPS
            currentTime = time.time()
            rps = nbParallelRays / (currentTime - previousTime)
            previousTime = currentTime

            # Show everything on screen
            imgNorm = img / img.max()
            #cv2.putText(imgNorm, f"RPS: {rps:.2f}"      , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            #cv2.putText(imgNorm, f"Rays: {nbIterations}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("FiatLux!", imgNorm)
            
            # Check for quitting the application
            # CAUTION: This block NEEDS to be be
            # called BEFORE any break statement!
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                breakRun = True
                break

            # IV. If all rays are dead, we can break the loop
            # We can also break if we have reached max depth.
            if onlyDeadRays or (longestRayDepth >= maximumRayDepth):
                break

            # V. Otherwise, we continue propagating the rays
            """
            wp.launch(
                kernel  = propagateRays,
                dim     = nbParallelRays,
                inputs  = [raysBuffer, intersectionsBuffer],
                outputs = [raysBuffer]
            )
            """

            longestRayDepth += 1

        if breakRun: 
            break