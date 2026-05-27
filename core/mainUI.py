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

# Light source: 1m x 1m lambertian parallelogram at z=1, normal toward -z.
# Vertex order chosen so that cross(v1-v0, v3-v0) points along -z.
lightSource       = _LightSource()
lightSource.type  = wp.int32(0)            # lambertian parallelogram
lightSource.v0    = wp.vec3f(-0.0015, -0.0015, 0.)
lightSource.v1    = wp.vec3f( 0.0015, -0.0015, 0.)
lightSource.v2    = wp.vec3f( 0.0015,  0.0015, 0.)
lightSource.v3    = wp.vec3f(-0.0015,  0.0015, 0.)
lightSource.power = wp.float32(1.0)        # 1 W total

lightSourceArray = wp.array([lightSource], dtype=_LightSource)

bk7Material = _Material()
bk7Material.iorPositive = wp.float32(1.00)
bk7Material.iorNegative = wp.float32(1.52)
materialArray = wp.array([bk7Material], dtype=_Material)

# Aspheric lens
asphericLens00 = _Primitive()
asphericLens00.type = 3 # Sphere (or spherical cap)
asphericLens00.materialID = 0 # BK7
asphericLens00.v0 = wp.vec3f(0., 0., 0.0718379) # Center
asphericLens00.v1 = wp.vec3f(0., 0., -1.)  # Direction of spherical cap pole (normalized)
asphericLens00.f0 = wp.float32(0.06999948) # Radius
asphericLens00.f1 = wp.float32(0.182440)   # Angle of the spherical cap: pi(/2) for an (hemi)sphere

asphericLens01 = _Primitive()
asphericLens01.type = 4 # Cylinder
asphericLens01.materialID = 0 # BK7
asphericLens01.v0 = wp.vec3f(0., 0., 0.0030) # Center of basis
asphericLens01.v1 = wp.vec3f(0., 0., 1.) # Axis (normalized)
asphericLens01.f0 = wp.float32(0.01270) # Radius
asphericLens01.f1 = wp.float32(0.0012) # Height

asphericLens11 = _Primitive()
asphericLens11.type = 5 # Asphere
asphericLens11.materialID = 0 # BK7
asphericLens11.v0 = wp.vec3f(0., 0., 0.0158) # Local axis origin
asphericLens11.v1 = wp.vec3f(0., 0., -0.001) # (z-axis unit vector in the local frame)
asphericLens11.f0 = wp.float32(8.818197) # Radius
asphericLens11.f1 = wp.float32(-0.9991715) # Conic constant
asphericLens11.f2 = wp.float32(8.682167e-5) # 1st coefficient, associated to 4th power
asphericLens11.f3 = wp.float32(6.3760123e-8) # 2nd coefficient, associated to 6th power
asphericLens11.f4 = wp.float32(2.4073084e-9) # 3rd coefficient, associated to 8th power
asphericLens11.f5 = wp.float32(-1.7189021e-11) # 4th coefficient, associated to 10th power
asphericLens11.f6 = wp.float32(12.7) # Maximum radius (millimeters)


# Biconvex lens
biconvexLens00 = _Primitive()
biconvexLens00.type       = 3 # Sphere (or spherical cap)
biconvexLens00.materialID = 0 # BK7
biconvexLens00.v0         = wp.vec3f(0., 0., 0.0984) # Center
biconvexLens00.v1         = wp.vec3f(0., 0., -1.)    # Direction of spherical cap pole (normalized)
biconvexLens00.f0         = wp.float32(0.0592)       # Radius
biconvexLens00.f1         = wp.float32(0.447189)     # Angle of the spherical cap: pi(/2) for an (hemi)sphere

biconvexLens01 = _Primitive()
biconvexLens01.type = 4 # Cylinder
biconvexLens01.materialID = 0 # BK7
biconvexLens01.v0         = wp.vec3f(0., 0., 0.04490) # Center of basis
biconvexLens01.v1         = wp.vec3f(0., 0., 1.)      # Axis (normalized)
biconvexLens01.f0         = wp.float32(0.02540) # Radius
biconvexLens01.f1         = wp.float32(0.003) # Height

biconvexLens11 = _Primitive()
biconvexLens11.type       = 3 # Sphere (or spherical cap)
biconvexLens11.materialID = 0 # BK7
biconvexLens11.v0         = wp.vec3f(0., 0., -0.0056) # Center
biconvexLens11.v1         = wp.vec3f(0., 0., +1.)     # Direction of spherical cap pole (normalized)
biconvexLens11.f0         = wp.float32(0.0592)        # Radius
biconvexLens11.f1         = wp.float32(0.447189)      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

# Add hitbox sphere
hitboxSphere = _Primitive()
hitboxSphere.type = 3 # Sphere
hitboxSphere.materialID = -1 # Blocker
hitboxSphere.v0 = wp.vec3f(0., 0., 0.) # Center
hitboxSphere.v1 = wp.vec3f(0., 0., +1.)     # Direction of spherical cap pole (normalized)
hitboxSphere.f0 = wp.float32(5.)        # Radius
hitboxSphere.f1 = wp.half_pi      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

#primitivesBuffer = wp.array([asphericLens00, asphericLens01, asphericLens11], dtype=_Primitive)
primitivesBuffer = wp.array([asphericLens00, asphericLens01, asphericLens11, biconvexLens00, biconvexLens01, biconvexLens11, hitboxSphere], dtype=_Primitive)
#primitivesBuffer = wp.array([hitboxSphere], dtype=_Primitive)

sensorHalfSize = 0.1
sensorDist = 0.8
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
MAX_BOUNCES   = 5
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
        colormap = "inferno",
    )
    sensorDisplay = BufferDisplay(
        extent   = (-sensorHalfSize, sensorHalfSize, -sensorHalfSize, sensorHalfSize),
        colormap = "viridis",
    )

    # Compose the main window
    window = MainWindow(
        leftDisplay  = vizDisplay,
        rightDisplay = sensorDisplay,
        title        = "Light Tracer",
    )
    window.show()

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

    sys.exit(app.exec())