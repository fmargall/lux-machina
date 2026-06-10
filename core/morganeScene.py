import sys
import numpy as np
import warp as wp
from pyqtgraph.Qt import QtWidgets, QtCore

from _structures         import (_Camera, _Ray, _LightSource, _Material, _Primitive, _Sensor, _Intersection)
from _generateRays       import _generateRays
from _intersect          import _intersect
from _accumulateOnSensor import _accumulateOnSensor, _checkLambertianPlateAndAccumulateOnCamera
from _propagate          import _propagate
from _visualize          import _visualize, _visualizeRayPaths
from _ui                 import BufferDisplay, MainWindow


wp.init()
wp.config.verbose = True

# ──────────────────────────────────────────────────────────────────────────
# Scene setup
# ──────────────────────────────────────────────────────────────────────────

superKCompact = _LightSource()
superKCompact.type  = wp.int32(2) # Gaussian beam
superKCompact.v0    = wp.vec3f(0., 0., 0.) # Waist at world origin
superKCompact.v1    = wp.vec3f(0., 0., 1.) # Propagation along +z
superKCompact.v2    = wp.vec3f(0., 0., 0.) # Unused
superKCompact.v3    = wp.vec3f(0., 0., 0.) # Unused
superKCompact.f0    = wp.float32(0.0005)    # w0 = 0.5 mm
superKCompact.power = wp.float32(0.020)    # 20 mW (visible band, see above)

lightSourceArray = wp.array([superKCompact], dtype=_LightSource)

nBAF10Material = _Material()
nBAF10Material.type = 0 # Perfect Fresnel
nBAF10Material.f0 = wp.float32(1.00)
nBAF10Material.f1 = wp.float32(1.6658)

nSF6HTMaterial = _Material()
nSF6HTMaterial.type = 0 # Perfect Fresnel
nSF6HTMaterial.f0 = wp.float32(1.00)
nSF6HTMaterial.f1 = wp.float32(1.7961)

nSF6HTnBAF10Material = _Material()
nSF6HTnBAF10Material.type = 0 # Perfect Fresnel
nSF6HTnBAF10Material.f0 = wp.float32(1.7961)
nSF6HTnBAF10Material.f1 = wp.float32(1.6658)

eBAF11Material = _Material()
eBAF11Material.type = 0 # Perfect Fresnel
eBAF11Material.f0 = wp.float32(1.00)
eBAF11Material.f1 = wp.float32(1.6626)

nSF11Material = _Material()
nSF11Material.type = 0 # Perfect Fresnel
nSF11Material.f0 = wp.float32(1.00)
nSF11Material.f1 = wp.float32(1.7760)

nSF11eBAF11Material = _Material()
nSF11eBAF11Material.type = 0 # Perfect Fresnel
nSF11eBAF11Material.f0 = wp.float32(1.7760)
nSF11eBAF11Material.f1 = wp.float32(1.6626)

nBK7Material = _Material()
nBK7Material.type = 0 # Perfect Fresnel
nBK7Material.f0 = wp.float32(1.00)
nBK7Material.f1 = wp.float32(1.5143)

nSF5Material = _Material()
nSF5Material.type = 0 # Perfect Fresnel
nSF5Material.f0 = wp.float32(1.00)
nSF5Material.f1 = wp.float32(1.6666)

nSF5nBK7Material = _Material()
nSF5nBK7Material.type = 0 # Perfect Fresnel
nSF5nBK7Material.f0 = wp.float32(1.6666)
nSF5nBK7Material.f1 = wp.float32(1.5143)

materialArray = wp.array([nBAF10Material, nSF6HTMaterial, nSF6HTnBAF10Material,
                          eBAF11Material, nSF11Material , nSF11eBAF11Material ,
                          nBK7Material  , nSF5Material  , nSF5nBK7Material    ], dtype=_Material)

ac254030a0 = _Primitive()
ac254030a0.type = 3 # Sphere (or spherical cap)
ac254030a0.materialID = 0 # nBAF10
ac254030a0.v0 = wp.vec3f(0., 0., 0.7709)  # Center (0.750 + R = 0.0209)
ac254030a0.v1 = wp.vec3f(0., 0., -1.)     # Direction of spherical cap pole (normalized)
ac254030a0.f0 = wp.float32(0.0209)        # Radius
ac254030a0.f1 = wp.float32(1.30621)      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

ac254030a1 = _Primitive()
ac254030a1.type = 4 # Cylinder
ac254030a1.materialID = 0 # nBAF10
ac254030a1.v0 = wp.vec3f(0., 0., 0.7665987979792217) # Center (0.750 + 0.016598797979221647)
ac254030a1.v1 = wp.vec3f(0., 0., 1.)     # Direction of cylindrical axis (normalized)
ac254030a1.f0 = wp.float32(0.02097) # Radius
ac254030a1.f1 = wp.float32(0.00184315)          # Length (not used for infinite cylinder)

ac254030a2 = _Primitive()
ac254030a2.type = 3 # Sphere (or spherical cap)
ac254030a2.materialID = 0 # nBAF10


# Add hitbox sphere
hitboxSphere = _Primitive()
hitboxSphere.type = 3 # Sphere
hitboxSphere.materialID = -1 # Blocker
hitboxSphere.v0 = wp.vec3f(0., 0., 0.) # Center
hitboxSphere.v1 = wp.vec3f(0., 0., 1.) # Direction of spherical cap pole (normalized)
hitboxSphere.f0 = wp.float32(5.)       # Radius
hitboxSphere.f1 = wp.half_pi      # Angle of the spherical cap: pi(/2) for an (hemi)sphere

primitivesBuffer = wp.array([ac254030a0, ac254030a1, hitboxSphere], dtype=_Primitive)

sensorHalfSizeX = 0.036 / 2. # 36 mm sensor width
sensorHalfSizeY = 0.024 / 2. # 24 mm sensor height
sensorDist = 1.96
sensor      = _Sensor()
sensor.type = wp.int32(0)                  # flat radiometer
sensor.v0   = wp.vec3f(-sensorHalfSizeX, -sensorHalfSizeY, sensorDist)
sensor.v1   = wp.vec3f( sensorHalfSizeX, -sensorHalfSizeY, sensorDist)
sensor.v2   = wp.vec3f( sensorHalfSizeX,  sensorHalfSizeY, sensorDist)
sensor.v3   = wp.vec3f(-sensorHalfSizeX,  sensorHalfSizeY, sensorDist)
sensor.i0   = wp.int32(6728)               # resX
sensor.i1   = wp.int32(4486)               # resY

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

VIZ_X_MIN, VIZ_X_MAX = -0.0005, 0.0005     # 8 cm horizontal extent
VIZ_Z_MIN, VIZ_Z_MAX = 0.749, 0.751        # 13 cm along the optical axis
VIZ_RES_X = 512
VIZ_RES_Z = 1024

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

INPUT_SEED    = wp.int32(0)
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
        colormap = "inferno", transform = "linear",
    )
    sensorDisplay = BufferDisplay(
        extent   = (-sensorHalfSizeX, sensorHalfSizeX, -sensorHalfSizeY, sensorHalfSizeY),
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

    sys.exit(app.exec())