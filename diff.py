import warp as wp

from generatePrimaryRays import generatePrimary3DRays

from structures import LightSource3D, Primitive3D, Ray3D, Sensor3D


@wp.func
def rotateVec(v: wp.vec3f, theta: wp.float32, phi: wp.float32, psi: wp.float32):
    cth = wp.cos(theta)
    sth = wp.sin(theta)

    cph = wp.cos(phi)
    sph = wp.sin(phi)

    cps = wp.cos(psi)
    sps = wp.sin(psi)

    # Rz(phi)
    x1 =  cph * v.x - sph * v.y
    y1 =  sph * v.x + cph * v.y
    z1 =  v.z

    # Ry(theta)
    x2 =  cth * x1 + sth * z1
    y2 =  y1
    z2 = -sth * x1 + cth * z1

    # Rz(psi)
    x3 =  cps * x2 - sps * y2
    y3 =  sps * x2 + cps * y2
    z3 =  z2

    return wp.vec3(x3, y3, z3)

@wp.kernel
def transformEmitterSystem(
    poseParams: wp.array(dtype=wp.float32, ndim=1),

    nbLightSources: wp.int32,
    nbPrimitives  : wp.int32,
    
    emitterLightSourcesLocalBuffer: wp.array(dtype=LightSource3D, ndim=1),
    emitterPrimitivesLocalBuffer  : wp.array(dtype=Primitive3D, ndim=1),

    emitterLightSourcesWorldBuffer: wp.array(dtype=LightSource3D, ndim=1),
    emitterPrimitivesWorldBuffer  : wp.array(dtype=Primitive3D, ndim=1)
):
    tx    = poseParams[0]
    ty    = poseParams[1]
    tz    = poseParams[2]
    theta = poseParams[3]
    phi   = poseParams[4]
    psi   = poseParams[5]

    t = wp.vec3(tx, ty, tz)

    # Transforming light sources
    for lightSourceID in range(nbLightSources):
        lsLocal = emitterLightSourcesLocalBuffer[lightSourceID]

        if lsLocal.type == 1: # Parallelogram lambertian
            lsWorld = LightSource3D(
                type = lsLocal.type,
                v0   = rotateVec(lsLocal.v0, theta, phi, psi) + t,
                v1   = rotateVec(lsLocal.v1, theta, phi, psi),
                v2   = rotateVec(lsLocal.v2, theta, phi, psi),
                f0   = lsLocal.f0
            )

        emitterLightSourcesWorldBuffer[lightSourceID] = lsWorld

    # Transforming primitives
    for primitiveID in range(nbPrimitives):
        pLocal = emitterPrimitivesLocalBuffer[primitiveID]
        
        if    ((pLocal.type == 0)   # Annulus blocker
            or (pLocal.type == 1)   # Cylinder
            or (pLocal.type == 2)   # Spherical cap
            or (pLocal.type == 3)): # Aspheric profile
            
            pWorld = Primitive3D(
                type = pLocal.type,
                v0   = rotateVec(pLocal.v0, theta, phi, psi) + t,
                v1   = rotateVec(pLocal.v1, theta, phi, psi),
                v2   = pLocal.v2,   # Unused for these types
                v3   = pLocal.v3,   # Unused for these types
                f0   = pLocal.f0,
                f1   = pLocal.f1,
                f2   = pLocal.f2,
                f3   = pLocal.f3,
                f4   = pLocal.f4,
                f5   = pLocal.f5,
                f6   = pLocal.f6,
                f7   = pLocal.f7,
                f8   = pLocal.f8,
                f9   = pLocal.f9,
                f10  = pLocal.f10
            )

        emitterPrimitivesWorldBuffer[primitiveID] = pWorld


if __name__ == "__main__":
    wp.init()
    wp.config.verbose = True # Better to check potentiel problems

    # Initialisation

    tx    = wp.float32(0.)
    ty    = wp.float32(0.)
    tz    = wp.float32(0.)
    theta = wp.float32(0.)
    phi   = wp.float32(0.)
    psi   = wp.float32(0.)

    # Since poseParams is the array that will be optimised
    # at the end, it is thus needed to set 'requires_grad' 
    # as True
    poseParams = wp.array(
        [tx, ty, tz, theta, phi, psi],
        dtype=wp.float32,
        requires_grad=True
    )

    # 0.(i) Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 1 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.003, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.003, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesList = [lightSource]

    # Aspheric lens
    nI = 1.0
    nO = 1.52
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
    biconvexLens11.v0 = wp.vec3(0., 0., -0.0056) # Center of sphere
    biconvexLens11.v1 = wp.vec3(0., 0., +0.0592) # Direction of spherical cap pole, with radius as length
    biconvexLens11.f0 = wp.float32(0.447189) # Angle of the spherical cap: pi(/2) for an (hemi)sphere
    biconvexLens11.f1 = wp.float32(nI) # Refractive index in the +normal direction, ie. outside
    biconvexLens11.f2 = wp.float32(nO) # Refractive index in the -normal direction, ie. inside.

    # Retaining rings
    retainingRing0 = Primitive3D()
    retainingRing0.type = 0 # Annulus blocker
    retainingRing0.v0 = wp.vec3(0., 0., 0.04380) # Origin (0.04380)
    retainingRing0.v1 = wp.vec3(0., 0., 1.) # Normal
    retainingRing0.f0 = wp.float32(0.02290) # Inner radius (0.02290)
    retainingRing0.f1 = wp.float32(0.02540) # Outer radius

    retainingRing1 = Primitive3D()
    retainingRing1.type = 0 # Annulus blocker
    retainingRing1.v0 = wp.vec3(0., 0., 0.04900) # Origin
    retainingRing1.v1 = wp.vec3(0., 0., 1.) # Normal
    retainingRing1.f0 = wp.float32(0.02290) # Inner radius
    retainingRing1.f1 = wp.float32(0.02540) # Outer radius

    primitivesList = []
    primitivesList.append(asphericLens00)
    primitivesList.append(asphericLens01)
    primitivesList.append(asphericLens11)

    primitivesList.append(biconvexLens00)
    primitivesList.append(biconvexLens01)
    primitivesList.append(biconvexLens11)

    primitivesList.append(retainingRing0)
    primitivesList.append(retainingRing1)

    # Initializing buffers
    emitterLightSourcesLocalBuffer = wp.array(lightSourcesList, dtype=LightSource3D)
    emitterPrimitivesLocalBuffer   = wp.array(primitivesList  , dtype=Primitive3D)
    emitterLightSourcesWorldBuffer = wp.zeros(1, dtype=LightSource3D)
    emitterPrimitivesWorldBuffer   = wp.zeros(len(primitivesList), dtype=Primitive3D)

    # Sensor initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0 , 0.0, 0.3653) # Center (0.0536 for direct output after last primitive / 0.0653 for output after last mechanical support)
    sensor.v1 = wp.vec3(0.0 , 0.1208, 0.0   ) # Tangent (0.0508) (at 1 meter : 0.4208)
    sensor.v2 = wp.vec3(0.1208, 0.0 , 0.0  ) # Bitangent
    sensor.i0 = wp.int32(256)

    maxDepth = 10
    nbParallelRays = 100_000

    nbOptimIterations = 10
    learningRate = 1.e-4

    for iterationID in range(nbOptimIterations):

        # The Tape class will be used to record kernel launches, and replay them to
        # compute the gradient of a scalar loss function with respect to our inputs
        tape = wp.Tape()

        # Initializing loss and sensor buffers
        sensorPixelsBitangent = sensor.i0 * wp.norm_l2(sensor.v2) / wp.norm_l2(sensor.v1)
        sensorBuffer = wp.zeros((sensor.i0, sensorPixelsBitangent), dtype=wp.float32)
        lossBuffer = wp.zeros(1, dtype=wp.float32)

        with tape:

            # Let's transform the complete emitter system
            wp.launch(
                kernel  = transformEmitterSystem,
                dim     = 1,
                inputs  = [poseParams, len(lightSourcesList), len(primitivesList), 
                           emitterLightSourcesLocalBuffer, emitterPrimitivesLocalBuffer],
                outputs = [emitterLightSourcesWorldBuffer, emitterPrimitivesWorldBuffer]
            )

            # Generate primary rays from light sources buffer
            raysBuffer = wp.zeros(nbParallelRays, dtype=Ray3D)
            wp.launch(
                kernel  = generatePrimary3DRays,
                dim     = nbParallelRays,
                inputs  = [iterationID, emitterLightSourcesWorldBuffer],
                outputs = [raysBuffer]
            )
            
            """
            # Ray tracing
            for depth in range(maxDepth):
                # Intersect rays
                wp.launch(
                    kernel  =
                    dim     =
                    inputs  =
                    outputs =
                )

                # Accumulate
                wp.launch(
                    kernel  =
                    dim     =
                    inputs  =
                    outputs =
                )

                # Propagate rays
                wp.launch(
                    kernel  =
                    dim     =
                    inputs  =
                    outputs =
                )

                # Compute loss
                wp.launch(
                    kernel  =
                    dim     =
                    inputs  =
                    outputs =
                )
        
        tape.backward(lossBuffer)
        
        # Gradient descent
        poseParamsNp = poseParams.numpy()
        gradientNp   = poseParams.grad.numpy()
        
        poseParamsNp -= learningRate * gradientNp
        poseParams.assign(poseParamsNp)

        poseParams.grad.zero_()

        print("Iteration", iteration, "loss", lossBuffer.numpy())
        print("Optimized pose:", poseParams.numpy())
        """