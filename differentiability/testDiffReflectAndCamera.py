import math
import numpy as np
import warp  as wp

# Better for differentiable rendering, especially with 
# dichotomy method for aspheric lens intersection. But
# this will slow down gradient computation strongly.
wp.config.max_unroll = 64 

from accumulateSensor    import accumulateIdeal3DSensor
from generatePrimaryRays import generatePrimary3DRays
from intersections       import noIntersection3DRays, intersect3DRays
from propagations        import propagate3DRays

from structures import Intersection3D, LightSource3D, Primitive3D, Ray3D, Sensor3D

@wp.func
def rodrigues(r: wp.vec3f) -> wp.mat33f:
    theta = wp.length(r)

    I = wp.mat33f(
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0
    )

    if theta < 1e-8:
        return I

    k = r / theta

    K = wp.mat33f(
         0.0, -k.z,  k.y,
         k.z,  0.0, -k.x,
        -k.y,  k.x,  0.0
    )

    R = I + wp.sin(theta) * K + (1.0 - wp.cos(theta)) * (K @ K)

    return R

@wp.func
def rotateVec(v: wp.vec3f, theta: wp.float32, phi: wp.float32, psi: wp.float32) -> wp.vec3f:
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

    return wp.vec3f(x3, y3, z3)

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
    tx = poseParams[0]
    ty = poseParams[1]
    tz = poseParams[2]

    rx = poseParams[3]
    ry = poseParams[4]
    rz = poseParams[5]

    t = wp.vec3f(tx, ty, tz)
    r = wp.vec3f(rx, ry, rz)

    R = rodrigues(r)

    # Transforming light sources
    for lightSourceID in range(nbLightSources):
        lsLocal = emitterLightSourcesLocalBuffer[lightSourceID]

        if lsLocal.type == 1: # Parallelogram lambertian
            lsWorld = LightSource3D(
                type = lsLocal.type,
                v0 = R @ lsLocal.v0 + t,
                v1 = R @ lsLocal.v1,
                v2 = R @ lsLocal.v2,
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
                v0   = R @ pLocal.v0 + t,
                v1   = R @ pLocal.v1,
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

@wp.kernel
def computeLoss(
    sensorBuffer   : wp.array(dtype=wp.float32, ndim=2),
    referenceBuffer: wp.array(dtype=wp.float32, ndim=2),
    loss: wp.array(dtype=wp.float32, ndim=1)
):
    i, j = wp.tid()

    diff = sensorBuffer[i, j] - referenceBuffer[i, j]

    wp.atomic_add(loss, 0, diff * diff)

@wp.kernel
def computeSum(
    sensorBuffer: wp.array(dtype=wp.float32, ndim=2),
    totalEnergy : wp.array(dtype=wp.float32, ndim=1)
):
    i, j = wp.tid()

    wp.atomic_add(totalEnergy, 0, sensorBuffer[i, j])

@wp.kernel
def normalizeSensor(
    sensorBuffer: wp.array(dtype=wp.float32, ndim=2),
    totalEnergy : wp.array(dtype=wp.float32, ndim=1),

    normalizedBuffer: wp.array(dtype=wp.float32, ndim=2)
):
    i, j = wp.tid()

    eps = wp.float32(1e-8)

    normalizedBuffer[i, j] = sensorBuffer[i, j] / (totalEnergy[0] + eps)


if __name__ == "__main__":
    wp.init()
    #wp.config.verbose=True                      # Better to check potentiel problems
    #wp.config.verify_autograd_array_access=True # Better to check potential problems

    # Initialisation
    tx, ty, tz = wp.float32(0.), wp.float32(0.), wp.float32(0.)
    rx, ry, rz = wp.float32(1e-5), wp.float32(1e-5), wp.float32(1e-5)


    # Since optParams is the array that will be optimised
    # at the end, it is thus needed to set 'requires_grad' 
    # as True
    optParams = wp.array(
        #[tx, ty, tz], dtype=wp.float32, requires_grad=True
        [tx, ty, tz, rx, ry, rz], dtype=wp.float32, requires_grad=True
    )

    # Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 1 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.003, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.003, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesList = [lightSource]

    # 0.(ii) Primitives initialization
    primitivesList = []
    
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

    primitivesList.append(asphericLens00)
    primitivesList.append(asphericLens01)
    primitivesList.append(asphericLens11)

    primitivesList.append(biconvexLens00)
    primitivesList.append(biconvexLens01)
    primitivesList.append(biconvexLens11)

    primitivesList.append(retainingRing0)
    primitivesList.append(retainingRing1)

    # Initializing buffers
    emitterLightSourcesLocalBuffer = wp.array(    lightSourcesList , dtype=LightSource3D)
    emitterLightSourcesWorldBuffer = wp.zeros(len(lightSourcesList), dtype=LightSource3D, requires_grad=True)
    emitterPrimitivesLocalBuffer   = wp.array(    primitivesList   , dtype=Primitive3D)
    emitterPrimitivesWorldBuffer   = wp.zeros(len(primitivesList)  , dtype=Primitive3D, requires_grad=True)

    # Sensor initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0, 0.0, 0.4)
    sensor.v1 = wp.vec3(0.0, 0.1208, 0.0)
    sensor.v2 = wp.vec3(0.1208, 0.0, 0.0)
    sensor.i0 = wp.int32(256)

    sensorBuffer = wp.zeros((256, 256), dtype=wp.float32, requires_grad=True)

    # Reference initialization
    """
    N = 512
    referenceNp = np.zeros((N, N), dtype=np.float32)

    cx = N // 2
    cy = N // 2
    half = N // 4

    referenceNp[cx-half:cx+half, cy-half:cy+half] = 1.0

    """
    from pathlib import Path
    path = Path(__file__).parent / "reference_sensor.npy"
    referenceNp = np.load(path).astype(np.float32)
    #referenceNp = np.load("reference_sensor.npy").astype(np.float32)
    referenceNp /= referenceNp.sum()
    referenceBuffer = wp.array(referenceNp, dtype=wp.float32)

    # Parameters of the renderer
    nbParallelRays = 1_000_000
    maxDepth = 10

    # Parameters of the optimiser
    learningRate = 1e-5
    learningRateVec = np.array([1e-4, 1e-4, 1e-3, 1e-6, 1e-6, 1e-6], dtype=np.float64)
    nIters = 1000

    lossTolerance = 1e-9
    stallCounter = 0
    patience = 20

    # Adam parameters
    beta1 = 0.9
    beta2 = 0.999
    eps = 1.e-8

    nParams = optParams.numpy().shape[0]
    m = np.zeros(nParams)
    v = np.zeros(nParams)

    # Best state tracking
    bestLoss = float("inf")
    bestParams = optParams.numpy().copy()

    # Decay
    initialLearningRate = learningRate
    lrDecay = 1.0
    minLearningRate = 1e-6

    for it in range(nIters):
        sensorBuffer.zero_()

        tape = wp.Tape()
        with tape:
            # ----- Beginning of the optimizer -----

            # Emitter
            wp.launch(
                kernel  = transformEmitterSystem,
                dim     = 1,
                inputs  = [optParams, len(lightSourcesList), len(primitivesList),
                           emitterLightSourcesLocalBuffer, emitterPrimitivesLocalBuffer],
                outputs = [emitterLightSourcesWorldBuffer, emitterPrimitivesWorldBuffer]
            )

            # Generate primary rays from light sources buffer
            raysBuffer = wp.empty(nbParallelRays, dtype=Ray3D, requires_grad=True)
            wp.launch(
                kernel  = generatePrimary3DRays,
                dim     = nbParallelRays,
                # FrameID (0) is stabilised for optimisation:
                inputs  = [0, emitterLightSourcesWorldBuffer],
                outputs = [raysBuffer]
            )

            # Ray tracing
            for depth in range(maxDepth):
                # Generate empty intersection buffer
                raysStatusBuffer    = wp.empty((nbParallelRays,), dtype=wp.bool)
                intersectionsBuffer = wp.empty((nbParallelRays,), dtype=Intersection3D, requires_grad=True)
                wp.launch(
                    kernel  = intersect3DRays, # noIntersection3DRays,
                    dim     = nbParallelRays,
                    inputs  = [raysBuffer, emitterPrimitivesWorldBuffer, len(primitivesList)],
                    outputs = [intersectionsBuffer, raysStatusBuffer]
                )

                # Accumulate
                wp.launch(
                    kernel  = accumulateIdeal3DSensor,
                    dim     = nbParallelRays,
                    inputs  = [intersectionsBuffer, sensor],
                    outputs = [sensorBuffer]
                )

                # Propagate rays
                wp.launch(
                    kernel = propagate3DRays,
                    dim    = nbParallelRays,
                    inputs = [intersectionsBuffer, 0], # IterationID is stabilised for optimisation
                    outputs = [raysBuffer]
                )

            totalEnergy = wp.zeros(1, dtype=wp.float32, requires_grad=True)

            wp.launch(
                kernel  = computeSum,
                dim     = sensorBuffer.shape,
                inputs  = [sensorBuffer],
                outputs = [totalEnergy]
            )

            normalizedBuffer = wp.empty_like(sensorBuffer)

            wp.launch(
                kernel  = normalizeSensor,
                dim     = sensorBuffer.shape,
                inputs  = [sensorBuffer, totalEnergy],
                outputs = [normalizedBuffer]
            )

            # Compute loss
            lossBuffer = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp.launch(
                kernel  = computeLoss,
                dim     = normalizedBuffer.shape,
                inputs  = [normalizedBuffer, referenceBuffer],
                outputs = [lossBuffer]
            )

        # -------- End of the optimizer --------
        tape.backward(loss=lossBuffer, grads=None)

        loss = float(lossBuffer.numpy()[0])

        params = optParams.numpy()        # shape (6,)
        grads  = optParams.grad.numpy()   # shape (6,)

        # ----- Best state tracking -----
        if loss < bestLoss - lossTolerance:
            bestLoss   = loss
            bestParams = params.copy()
            stallCounter = 0
        else:
            stallCounter += 1
        
        # ----- Adam update -----
        g = grads

        m = beta1 * m + (1 - beta1) *  g
        v = beta2 * v + (1 - beta2) * (g * g)

        m_hat = m / (1 - beta1 ** (it + 1))
        v_hat = v / (1 - beta2 ** (it + 1))

        #newParams = params - learningRate * m_hat / (np.sqrt(v_hat) + eps)
        newParams = params - learningRateVec * m_hat / (np.sqrt(v_hat) + eps)

        # ----- Optional rollback if divergence -----
        if loss > bestLoss * 1.1:
            newParams = bestParams.copy()
            m[:] = 0.0
            v[:] = 0.0

            # reduce learning rate
            learningRate *= 0.5
            learningRate = max(learningRate, minLearningRate)

            tape.zero()

        else:
            # Decay
            learningRate *= lrDecay

        optParams = wp.array(newParams.astype(np.float32), dtype=wp.float32, requires_grad=True)

        print(
            f"iter {it:03d} | "
            f"loss = {loss:.8f} | "
            f"tx={params[0]:.5f} ty={params[1]:.5f} tz={params[2]:.5f} | "
            f"rx={params[3]:.5f} ry={params[4]:.5f} rz={params[5]:.5f} | "
            f"gx={grads[0]:.3e} gy={grads[1]:.3e} gz={grads[2]:.3e} | "
            f"grx={grads[3]:.3e} gry={grads[4]:.3e} grz={grads[5]:.3e} | "
            f"lr={learningRate:.2e}"
        )

        if stallCounter >= patience:
            print("Early stopping: loss stalled.")
            break

    import matplotlib.pyplot as plt

    sensor_np    = normalizedBuffer.numpy()
    reference_np = referenceBuffer.numpy()

    vmin = 0.0
    vmax = max(sensor_np.max(), reference_np.max())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    im0 = axes[0].imshow(sensor_np, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[0].set_title("Sensor buffer")
    axes[0].set_xlabel("Pixel i")
    axes[0].set_ylabel("Pixel j")

    im1 = axes[1].imshow(reference_np, origin="lower", cmap="inferno", vmin=vmin, vmax=vmax)
    axes[1].set_title("Reference buffer")
    axes[1].set_xlabel("Pixel i")
    axes[1].set_ylabel("Pixel j")

    fig.colorbar(im0, ax=axes, label="Energy")

    plt.show()