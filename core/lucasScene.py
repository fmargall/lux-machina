import sys
import numpy as np
import warp as wp
from pyqtgraph.Qt import QtWidgets, QtCore

from _structures         import (_Ray, _LightSource, _Material, _Primitive, _Sensor, _Intersection)
from _generateRays       import _generateRays
from _intersect          import _intersect
from _accumulateOnSensor import _accumulateOnSensor
from _propagate          import _propagate
from _visualize          import _visualize, _visualizeRayPaths
from _ui                 import BufferDisplay, MainWindow


wp.init()
wp.config.verbose = True

# ──────────────────────────────────────────────────────────────────────────
# Scene setup
# ──────────────────────────────────────────────────────────────────────────

# Light source: a 1 W point source at the origin, emitting uniformly in a aperture
# of a disk with radius 0.009 m, centered on the optical axis (z) at a distance of
# z = 0.02616 m.
lightSource = _LightSource()
lightSource.type  = wp.int32(1) # Pointlight source
lightSource.v0    = wp.vec3f(0., 0., 0.) # Position
lightSource.v1    = wp.vec3f(0., 0., 1.) # Emission direction (normalized)
lightSource.f0    = wp.float32(0.331352) # Emission cone half-angle (radians)
lightSource.power = wp.float32(1.0)      # Total emitted power (Watts)

lightSourceArray = wp.array([lightSource], dtype=_LightSource)

bk7Material = _Material()
bk7Material.type = 0 # Perfect Fresnel
bk7Material.f0 = wp.float32(1.00)
bk7Material.f1 = wp.float32(1.52)
materialArray = wp.array([bk7Material], dtype=_Material)

# Thorlabs LB1757 lens
biconvexLens00 = _Primitive()
biconvexLens00.type       = 3 # Sphere (or spherical cap)
biconvexLens00.materialID = 0 # BK7
biconvexLens00.v0         = wp.vec3f(0., 0., 0.05565) # Center
biconvexLens00.v1         = wp.vec3f(0., 0., -1.)     # Direction of spherical cap pole (normalized)
biconvexLens00.f0         = wp.float32(0.0295)        # Radius
biconvexLens00.f1         = wp.float32(0.445056)      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

biconvexLens01 = _Primitive()
biconvexLens01.type = 4 # Cylinder
biconvexLens01.materialID = 0 # BK7
biconvexLens01.v0         = wp.vec3f(0., 0., 0.0290237) # Center of basis
biconvexLens01.v1         = wp.vec3f(0., 0., 1.)      # Axis (normalized)
biconvexLens01.f0         = wp.float32(0.02540) # Radius
biconvexLens01.f1         = wp.float32(0.002)   # Height

biconvexLens11 = _Primitive()
biconvexLens11.type       = 3 # Sphere (or spherical cap)
biconvexLens11.materialID = 0 # BK7
biconvexLens11.v0         = wp.vec3f(0., 0., 0.00435) # Center
biconvexLens11.v1         = wp.vec3f(0., 0., 1.)      # Direction of spherical cap pole (normalized)
biconvexLens11.f0         = wp.float32(0.0295)        # Radius
biconvexLens11.f1         = wp.float32(0.445056)      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

# Add hitbox sphere
hitboxSphere = _Primitive()
hitboxSphere.type = 3 # Sphere
hitboxSphere.materialID = -1 # Blocker
hitboxSphere.v0 = wp.vec3f(0., 0., 0.) # Center
hitboxSphere.v1 = wp.vec3f(0., 0., +1.)     # Direction of spherical cap pole (normalized)
hitboxSphere.f0 = wp.float32(5.)        # Radius
hitboxSphere.f1 = wp.half_pi      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

primitivesBuffer = wp.array([biconvexLens00, biconvexLens01, biconvexLens11, hitboxSphere], dtype=_Primitive)

sensorHalfSize = 0.01 # 2 cm sensor size (square)
sensorDist  = 0.06105
sensor      = _Sensor()
sensor.type = wp.int32(0)                  # flat radiometer
sensor.v0   = wp.vec3f(-sensorHalfSize, -sensorHalfSize, sensorDist)
sensor.v1   = wp.vec3f( sensorHalfSize, -sensorHalfSize, sensorDist)
sensor.v2   = wp.vec3f( sensorHalfSize,  sensorHalfSize, sensorDist)
sensor.v3   = wp.vec3f(-sensorHalfSize,  sensorHalfSize, sensorDist)
sensor.i0   = wp.int32(256)                # resX
sensor.i1   = wp.int32(256)                # resY


# ──────────────────────────────────────────────────────────────────────────
# Buffers allocation
# ──────────────────────────────────────────────────────────────────────────

N_RAYS = 1_000_000
RES_X  = int(sensor.i0)
RES_Y  = int(sensor.i1)

rayBuffer          = wp.zeros(N_RAYS, dtype=_Ray)
intersectionBuffer = wp.zeros(N_RAYS, dtype=_Intersection)
sensorBuffer       = wp.zeros((RES_X, RES_Y), dtype=wp.float32)

# ──────────────────────────────────────────────────────────────────────────
# Visualization sensor: top-down view of the optical bench
# Plane: y = 0 (so we look at the xz plane)
# u axis = x (horizontal in image)
# v axis = z (vertical in image, = optical axis)
# ──────────────────────────────────────────────────────────────────────────

VIZ_X_MIN, VIZ_X_MAX = -0.04, 0.04        # 8 cm horizontal extent
VIZ_Z_MIN, VIZ_Z_MAX = -0.01, 0.12        # 13 cm along the optical axis
VIZ_RES_X = 512
VIZ_RES_Z = 832

# Vertices of the visualization plane, in the y=0 plane:
# Looking from +y toward -y, we see x growing rightward and z growing upward.
vizSensor      = _Sensor()
vizSensor.type = wp.int32(0)
vizSensor.v0   = wp.vec3f(VIZ_X_MIN, 0.0, VIZ_Z_MIN)   # bottom-left
vizSensor.v1   = wp.vec3f(VIZ_X_MAX, 0.0, VIZ_Z_MIN)   # bottom-right → defines u (= x)
vizSensor.v2   = wp.vec3f(VIZ_X_MAX, 0.0, VIZ_Z_MAX)   # top-right
vizSensor.v3   = wp.vec3f(VIZ_X_MIN, 0.0, VIZ_Z_MAX)   # top-left    → defines v (= z)
vizSensor.i0   = wp.int32(VIZ_RES_X)
vizSensor.i1   = wp.int32(VIZ_RES_Z)

vizBuffer = wp.zeros((VIZ_RES_X, VIZ_RES_Z), dtype=wp.float32)


# ──────────────────────────────────────────────────────────────────────────
# Pipeline execution with visualization
# ──────────────────────────────────────────────────────────────────────────

INPUT_SEED    = wp.int32(42)
FRAME_ID      = wp.int32(0)
MAX_BOUNCES   = 10
NB_PRIMITIVES = wp.int32(len(primitivesBuffer))

def runOneFrame(frameID: int):
    """
    Run one wave of the wavefront pipeline.
    The buffers vizBuffer and sensorBuffer accumulate across multiple frames.
    """
    # Each frame uses a different frameID to decorrelate the RNG
    wp.launch(
        _generateRays,
        dim    = N_RAYS,
        inputs = [lightSourceArray, INPUT_SEED, wp.int32(frameID), rayBuffer],
    )

    for bounce in range(MAX_BOUNCES):
        wp.launch(_intersect,
                  dim=N_RAYS,
                  inputs=[rayBuffer, primitivesBuffer, NB_PRIMITIVES, intersectionBuffer])

        wp.launch(_visualize,
                  dim=N_RAYS,
                  inputs=[rayBuffer, intersectionBuffer, vizSensor, vizBuffer])

        # This visualization is optional, but it can be very helpful to debug ray paths
        # since this rasterizer is exact. Careful: if will be slower for large N_RAYS !
        """
        wp.launch(_visualizeRayPaths,
                  dim=vizBuffer.shape,
                  inputs=[rayBuffer, intersectionBuffer, N_RAYS, vizSensor, vizBuffer])
        """

        wp.launch(_accumulateOnSensor,
                  dim=N_RAYS,
                  inputs=[rayBuffer, intersectionBuffer, sensor, sensorBuffer])

        wp.launch(_propagate,
                  dim=N_RAYS,
                  inputs=[intersectionBuffer, primitivesBuffer, materialArray,
                          INPUT_SEED, wp.int32(frameID), rayBuffer])


# ──────────────────────────────────────────────────────────────────────────
# UI setup and run loop
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Build the two displays
    vizDisplay = BufferDisplay(
        extent   = (VIZ_X_MIN, VIZ_X_MAX, VIZ_Z_MIN, VIZ_Z_MAX),
        colormap = "inferno", transform = "log1p",
    )
    sensorDisplay = BufferDisplay(
        extent   = (-sensorHalfSize, sensorHalfSize, -sensorHalfSize, sensorHalfSize),
        colormap = "viridis", transform = "linear",
    )

    # Compose the main window
    window = MainWindow(
        leftDisplay  = vizDisplay,
        rightDisplay = sensorDisplay,
        title        = "Light Tracer",
    )
    window.show()
    window.adjustSize()

    # ── Frame loop driven by a QTimer ──
    # Each tick of the timer = one frame of the simulation.
    # The timer interval is set to 0, so it runs as fast as possible
    # while still letting Qt process events between frames.

    frameCounter = [0]   # mutable container for the closure

    def onTimerTick():
        """Called by QTimer at each interval."""
        # Run one simulation frame
        runOneFrame(frameID=frameCounter[0])
        wp.synchronize()

        # Pull buffers from GPU and refresh the UI
        window.updateViews(
            leftBuffer  = vizBuffer.numpy(),
            rightBuffer = sensorBuffer.numpy(),
            frameID     = frameCounter[0],
        )

        frameCounter[0] += 1

    timer = QtCore.QTimer()
    timer.timeout.connect(onTimerTick)
    timer.start(0)   # 0 ms = as fast as possible (after each event loop iteration)

    # ── Run the event loop ──
    exitCode = app.exec()

    # ── Save accumulated buffers to disk on close ──
    vizArray    = vizBuffer.numpy()
    sensorArray = sensorBuffer.numpy()
    nFrames     = frameCounter[0]

    np.save("vizBuffer.npy",    vizArray)
    np.save("sensorBuffer.npy", sensorArray)
    print(f"Saved vizBuffer {vizArray.shape} and sensorBuffer {sensorArray.shape} "
          f"— accumulated over {nFrames} frames ({nFrames * N_RAYS:,} rays total).")

    sys.exit(exitCode)