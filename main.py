import time

import cv2
import numpy as np
import warp  as wp

from generatePrimaryRays import generatePrimaryRays
from intersections       import intersectRays, propagateRays

from rasterizer import rasterize

from structures import Intersection, LightSource, Primitive, Ray

if __name__ == "__main__":
    # GPU support using NVIDIA Warp
    wp.init()

    # 0. Light tracer parameters
    height, width   = 512, 512
    nbParallelRays  = 100_000
    maximumRayDepth = 1

    # 0.(i) Light sources initialisation
    lightSource = LightSource()
    lightSource.type      = 1
    lightSource.intensity = 1.0
    lightSource.p0        = wp.vec2(-1.0,  0.25)
    lightSource.p1        = wp.vec2(-1.0, -0.25)
    lightSourcesBuffer = wp.array([lightSource], dtype=LightSource)

    # 0.(ii) Scene primitives initialisation
    segment0 = Primitive()
    segment0.type = 0
    segment0.p0 = wp.vec2(-0.75, -0.25)
    segment0.p1 = wp.vec2(-0.75,  0.25)
    segment0.ni = 1.0
    segment0.no = 1.6

    segment1 = Primitive()
    segment1.type = 0
    segment1.p0 = wp.vec2(-0.75,  0.25)
    segment1.p1 = wp.vec2(-0.50,  0.25)
    segment1.ni = 1.6
    segment1.no = 1.0

    segment2 = Primitive()
    segment2.type = 0
    segment2.p0 = wp.vec2(-0.50, -0.25)
    segment2.p1 = wp.vec2(-0.50,  0.25)
    segment2.ni = 1.0
    segment2.no = 1.6

    segment3 = Primitive()
    segment3.type = 0
    segment3.p0 = wp.vec2(-0.50, -0.25)
    segment3.p1 = wp.vec2(-0.75, -0.25)
    segment3.ni = 1.6
    segment3.no = 1.0

    primitivesBuffer = wp.array([segment0, segment1, segment2, segment3], dtype=Primitive)
    nbPrimitives = primitivesBuffer.shape[0]

    # Initialising image buffer
    imageBuffer = wp.zeros((width, height), dtype=wp.float32)
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
        longestRayDepth = 5
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
                dim     = (width, height),
                inputs  = [raysBuffer, intersectionsBuffer, nbParallelRays, 
                           width, height],
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
            )"""

            longestRayDepth += 1

        if breakRun: 
            break