import time

import cv2
import numpy as np
import warp  as wp

from generatePrimaryRays import generatePrimaryRays
from intersections       import intersectRays
from propagations        import propagateRays

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
        else:
            raise ValueError(f"Unknown primitive type: {primitive.type} in scene coordinates conversion.")


    return (wp.array(lightSourcesList, dtype=LightSource, ndim=1),
            wp.array(primitivesList  , dtype=Primitive  , ndim=1))

if __name__ == "__main__":
    # GPU support using NVIDIA Warp
    wp.init()

    # 0. Light tracer parameters
    width, height   = 720, 720
    nbParallelRays  = 10_000
    maximumRayDepth = 10
    
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

    primitivesList.append(bbox)

    # Biconvex lens
    sphere = Primitive()
    sphere.type = 2
    sphere.v0   = wp.vec2(0.0015, 0.0)
    sphere.f0   = wp.float32(0.0005)
    sphere.f1   = wp.float32(0.)
    sphere.f2   = wp.float32(7.)
    sphere.f3   = wp.float32(1.5)
    sphere.f4   = wp.float32(1.0)

    primitivesList.append(sphere)

    # Thin film
    thinFilm00 = Primitive()
    thinFilm00.type = 1
    thinFilm00.v0 = wp.vec2( 0.003,  0.003)
    thinFilm00.v1 = wp.vec2( 0.003, -0.003)
    thinFilm00.f0 = wp.float32(1.)
    thinFilm00.f1 = wp.float32(1.6)

    thinFilm01 = Primitive()
    thinFilm01.type = 1
    thinFilm01.v0 = wp.vec2( 0.003,  0.003)
    thinFilm01.v1 = wp.vec2( 0.003606,  0.003)
    thinFilm01.f0 = wp.float32(1.6)
    thinFilm01.f1 = wp.float32(1.)

    thinFilm10 = Primitive()
    thinFilm10.type = 1
    thinFilm10.v0 = wp.vec2( 0.003, -0.003)
    thinFilm10.v1 = wp.vec2( 0.003606, -0.003)
    thinFilm10.f0 = wp.float32(1.)
    thinFilm10.f1 = wp.float32(1.6)

    thinFilm11 = Primitive()
    thinFilm11.type = 1
    thinFilm11.v0 = wp.vec2( 0.003606,  0.003)
    thinFilm11.v1 = wp.vec2( 0.003606, -0.003)
    thinFilm11.f0 = wp.float32(1.6)
    thinFilm11.f1 = wp.float32(1.)

    primitivesList.append(thinFilm00)
    primitivesList.append(thinFilm01)
    primitivesList.append(thinFilm10)
    primitivesList.append(thinFilm11)

    idealLens = Primitive()
    idealLens.type = 0
    idealLens.v0 = wp.vec2( 0.005,  0.003)
    idealLens.v1 = wp.vec2( 0.005, -0.003)
    idealLens.f0 = wp.float32(0.003)
    
    primitivesList.append(idealLens)

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

            # III.(iii) Once accumulated, we can display the current imageq

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

        if breakRun: 
            break