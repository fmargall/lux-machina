import numpy as np
import warp  as wp

from accumulateSensor    import accumulateIdeal3DSensor
from generatePrimaryRays import generatePrimary3DRays
from intersections       import noIntersection3DRays

from structures import Intersection3D, LightSource3D, Primitive3D, Ray3D, Sensor3D


@wp.kernel
def transformEmitterSystem(
    poseParams    : wp.array(dtype=wp.float32, ndim=1),
    nbLightSources: wp.int32,
    
    emitterLightSourcesLocalBuffer: wp.array(dtype=LightSource3D, ndim=1),
    emitterLightSourcesWorldBuffer: wp.array(dtype=LightSource3D, ndim=1)
):
    tz    = poseParams[0]

    t = wp.vec3f(0., 0., tz)

    # Transforming light sources
    for lightSourceID in range(nbLightSources):
        lsLocal = emitterLightSourcesLocalBuffer[lightSourceID]

        if lsLocal.type == 1: # Parallelogram lambertian
            lsWorld = LightSource3D(
                type = lsLocal.type,
                v0   = lsLocal.v0 + t,
                v1   = lsLocal.v1,
                v2   = lsLocal.v2,
                f0   = lsLocal.f0
            )

        emitterLightSourcesWorldBuffer[lightSourceID] = lsWorld

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
    wp.config.verbose = False                    # Better to check potentiel problems
    wp.config.verify_autograd_array_access=False # Better to check potential problems

    # Initialisation
    tz = wp.float32(0.)

    # Since optParams is the array that will be optimised
    # at the end, it is thus needed to set 'requires_grad' 
    # as True
    optParams = wp.array(
        [tz], dtype=wp.float32, requires_grad=True
    )

    # Light source initialisation
    lightSource = LightSource3D()
    lightSource.type = 1 # Parallelogram Lambertian
    lightSource.v0 = wp.vec3(0.   , 0.   , 0.) # Center
    lightSource.v1 = wp.vec3(0.003, 0.   , 0.) # Tangent
    lightSource.v2 = wp.vec3(0.   , 0.003, 0.) # Bitangent
    lightSource.f0 = wp.float32(1.)

    lightSourcesList = [lightSource]

    # Initializing buffers
    emitterLightSourcesLocalBuffer = wp.array(    lightSourcesList , dtype=LightSource3D)
    emitterLightSourcesWorldBuffer = wp.zeros(len(lightSourcesList), dtype=LightSource3D, requires_grad=True)

    # Sensor initialization
    sensor = Sensor3D()
    sensor.type = 0 # Ideal 3D sensor
    sensor.v0 = wp.vec3(0.0, 0.0, 0.002)
    sensor.v1 = wp.vec3(0.0, 0.006, 0.0)
    sensor.v2 = wp.vec3(0.006, 0.0, 0.0)
    sensor.i0 = wp.int32(512)

    sensorBuffer = wp.zeros((512, 512), dtype=wp.float32, requires_grad=True)

    # Reference initialization
    N = 512

    reference_np = np.zeros((N, N), dtype=np.float32)

    cx = N // 2
    cy = N // 2
    half = N // 4

    reference_np[cx-half:cx+half, cy-half:cy+half] = 1.0

    reference_np /= reference_np.sum()

    referenceBuffer = wp.array(reference_np, dtype=wp.float32)

    # Parameters of the renderer
    nbParallelRays = 1_000_000

    # Parameters of the optimiser
    learningRate = 5e-4
    nIters = 50

    for it in range(nIters):
        sensorBuffer.zero_()

        tape = wp.Tape()
        with tape:
            # ----- Beginning of the optimizer -----

            # Emitter
            wp.launch(
                kernel  = transformEmitterSystem,
                dim     = 1,
                inputs  = [optParams, len(lightSourcesList),
                           emitterLightSourcesLocalBuffer],
                outputs = [emitterLightSourcesWorldBuffer]
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

            # Generate empty intersection buffer
            intersectionsBuffer = wp.empty((nbParallelRays,), dtype=Intersection3D, requires_grad=True)
            wp.launch(
                kernel  = noIntersection3DRays,
                dim     = nbParallelRays,
                inputs  = [raysBuffer],
                outputs = [intersectionsBuffer]
            )

            # Accumulate
            wp.launch(
                kernel  = accumulateIdeal3DSensor,
                dim     = nbParallelRays,
                inputs  = [intersectionsBuffer, sensor],
                outputs = [sensorBuffer]
            )

            totalEnergy = wp.zeros(1, dtype=wp.float32, requires_grad=True)

            wp.launch(
                computeSum,
                dim=sensorBuffer.shape,
                inputs=[sensorBuffer],
                outputs=[totalEnergy]
            )

            normalizedBuffer = wp.empty_like(sensorBuffer)

            wp.launch(
                normalizeSensor,
                dim=sensorBuffer.shape,
                inputs=[sensorBuffer, totalEnergy],
                outputs=[normalizedBuffer]
            )

            # Compute loss
            lossBuffer = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            wp.launch(
                computeLoss,
                dim=normalizedBuffer.shape,
                inputs=[normalizedBuffer, referenceBuffer],
                outputs=[lossBuffer]
            )

        # -------- End of the optimizer --------
        tape.backward(loss=lossBuffer, grads=None)

        """
        loss = lossBuffer.numpy()[0]
        grad = optParams.grad.numpy()

        optParams.numpy()[0] -= lr * grad

        print("Loss: ", loss, " | Gradient: ", grad, " | tz: ", optParams.numpy()[0])
        """
        loss = float(lossBuffer.numpy()[0])
        grad = float(optParams.grad.numpy()[0])
        tz   = float(optParams.numpy()[0])

        # Gradient descent update
        new_tz = tz - learningRate * grad
        optParams = wp.array([new_tz], dtype=wp.float32, requires_grad=True)

        print(f"iter {it:03d} | loss = {loss:.8f} | tz = {tz:.8f} | grad = {grad:.8f}")

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