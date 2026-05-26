"""
test_pipeline.py

Smoke test for the light tracer's generation + accumulation pipeline.

Scene:
  - Lambertian light source (1m x 1m parallelogram at z=1, emitting toward -z), 1 W total.
  - Flat sensor (2m x 2m parallelogram at z=0, normal +z), 256x256 pixels.

Expected output:
  - Sum of sensor pixels close to 1 W (light source's total power).
  - Peak intensity at the center of the image.
  - Image roughly symmetric and decreasing toward the edges.
"""

import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from _structures         import (_Ray, _LightSource, _Material, _Primitive, _Sensor, _Intersection)
from _generateRays       import _generateRays
from _intersect          import _intersect
from _accumulateOnSensor import _accumulateOnSensor
from _propagate          import _propagate


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
asphericLens01.f0 = wp.float32(0.0012) # Height

asphericLens11 = _Primitive()
asphericLens11.type = 5 # Asphere
asphericLens11.materialID = 0 # BK7
asphericLens11.v0 = wp.vec3f(0., 0., 0.0042) # Local axis origin
asphericLens11.v1 = wp.vec3f(0., 0., 1.) # (z-axis unit vector in the local frame)
asphericLens11.f0 = wp.float32(8.818197) # Radius
asphericLens11.f1 = wp.float32(-0.9991715) # Conic constant
asphericLens11.f2 = wp.float32(8.682167e-5) # 1st coefficient, associated to 4th power
asphericLens11.f3 = wp.float32(6.3760123e-8) # 2nd coefficient, associated to 6th power
asphericLens11.f4 = wp.float32(2.4073084e-9) # 3rd coefficient, associated to 8th power
asphericLens11.f5 = wp.float32(-1.7189021e-11) # 4th coefficient, associated to 10th power
asphericLens11.f6 = wp.float32(0.0127) # Maximum radius


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

primitivesBuffer = wp.array([asphericLens00, asphericLens01, asphericLens11, biconvexLens00, biconvexLens01, biconvexLens11], dtype=_Primitive)
#primitivesBuffer = wp.array([], dtype=_Primitive)

sensorHalfSize = 0.3
sensorDist = 0.1
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

N_RAYS = 112_500_000
RES_X  = int(sensor.i0)
RES_Y  = int(sensor.i1)

rayBuffer          = wp.zeros(N_RAYS, dtype=_Ray)
intersectionBuffer = wp.zeros(N_RAYS, dtype=_Intersection)
sensorBuffer       = wp.zeros((RES_X, RES_Y), dtype=wp.float32)

# Note: intersectionBuffer is zero-initialized, so intersection.t = 0.0
# The visibility test in _accumulateOnSensor is:
#     if intersection.t > 0.0 and intersection.t < tSensor: return
# Since 0.0 is not > 0.0, rays are not rejected.

"""
# ──────────────────────────────────────────────────────────────────────────
# Pipeline execution
# ──────────────────────────────────────────────────────────────────────────

INPUT_SEED = wp.int32(42)
FRAME_ID   = wp.int32(0)

print("Generating rays...")
wp.launch(
    _generateRays,
    dim    = N_RAYS,
    inputs = [lightSourceArray, INPUT_SEED, FRAME_ID, rayBuffer],
)

# Diagnose ray generation
rays_np = rayBuffer.numpy()

print("\n─── Ray diagnostics ────────────────────────────────────────────────")
print(f"  Origins x range  : [{rays_np['origin'][:,0].min():.4f}, {rays_np['origin'][:,0].max():.4f}]")
print(f"  Origins y range  : [{rays_np['origin'][:,1].min():.4f}, {rays_np['origin'][:,1].max():.4f}]")
print(f"  Origins z range  : [{rays_np['origin'][:,2].min():.4f}, {rays_np['origin'][:,2].max():.4f}]")
print(f"  Directions x     : [{rays_np['direction'][:,0].min():.4f}, {rays_np['direction'][:,0].max():.4f}]")
print(f"  Directions y     : [{rays_np['direction'][:,1].min():.4f}, {rays_np['direction'][:,1].max():.4f}]")
print(f"  Directions z     : [{rays_np['direction'][:,2].min():.4f}, {rays_np['direction'][:,2].max():.4f}]")
print(f"  Throughput       : [{rays_np['throughput'].min():.6e}, {rays_np['throughput'].max():.6e}]")
print(f"  NaN in origin    : {np.isnan(rays_np['origin']).any()}")
print(f"  NaN in direction : {np.isnan(rays_np['direction']).any()}")
print(f"  NaN in throughput: {np.isnan(rays_np['throughput']).any()}")
print(f"  Direction norms  : [{np.linalg.norm(rays_np['direction'], axis=1).min():.4f}, {np.linalg.norm(rays_np['direction'], axis=1).max():.4f}]")

#print("Accumulating on sensor...")
wp.launch(
    _accumulateOnSensor,
    dim    = N_RAYS,
    inputs = [rayBuffer, intersectionBuffer, sensor, sensorBuffer],
)

wp.synchronize()
"""
# ──────────────────────────────────────────────────────────────────────────
# Pipeline execution: wavefront path tracing
# ──────────────────────────────────────────────────────────────────────────

INPUT_SEED   = wp.int32(42)
FRAME_ID     = wp.int32(0)
MAX_BOUNCES  = 5
NB_PRIMITIVES = wp.int32(len(primitivesBuffer))   # number of primitives in the scene

print(f"Generating {N_RAYS:,} rays from light source...")
wp.launch(
    _generateRays,
    dim    = N_RAYS,
    inputs = [lightSourceArray, INPUT_SEED, FRAME_ID, rayBuffer],
)

for bounce in range(MAX_BOUNCES):
    print(f"  Bounce {bounce + 1}/{MAX_BOUNCES}...")

    # 1. Find next intersection for each alive ray
    wp.launch(
        _intersect,
        dim    = N_RAYS,
        inputs = [rayBuffer, primitivesBuffer, NB_PRIMITIVES, intersectionBuffer],
    )

    # 2. Accumulate on sensor (rays passing through sensor plane this bounce)
    wp.launch(
        _accumulateOnSensor,
        dim    = N_RAYS,
        inputs = [rayBuffer, intersectionBuffer, sensor, sensorBuffer],
    )

    # 3. Propagate: reflect/refract at intersection, or kill ray
    wp.launch(
        _propagate,
        dim    = N_RAYS,
        inputs = [
            intersectionBuffer,
            primitivesBuffer,
            materialArray,
            INPUT_SEED,
            FRAME_ID,
            rayBuffer,
        ],
    )

wp.synchronize()
print("Pipeline complete.")


# ──────────────────────────────────────────────────────────────────────────
# Sanity checks
# ──────────────────────────────────────────────────────────────────────────

image = sensorBuffer.numpy()
"""
total_flux = image.sum()
peak_value = image.max()
peak_pos   = np.unravel_index(np.argmax(image), image.shape)

print()
print("─── Sanity checks ──────────────────────────────────────────────────")
print(f"  Total flux on sensor : {total_flux:.4f} W  (expected ~0.85-1.0)")
print(f"  Peak pixel value     : {peak_value:.6f}")
print(f"  Peak position (i,j)  : {peak_pos}  (expected near ({RES_X//2}, {RES_Y//2}))")
print(f"  Image shape          : {image.shape}")
print(f"  Min / Mean / Max     : {image.min():.6e} / {image.mean():.6e} / {image.max():.6e}")
"""
# ──────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 2D image (transpose to get x horizontal, y vertical)
im = ax1.imshow(image.T, cmap="viridis", origin="lower",
                extent=[-1.0, 1.0, -1.0, 1.0])
ax1.set_title(f"Sensor image ({N_RAYS:,} rays)")
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
plt.colorbar(im, ax=ax1)

# Horizontal cross-section through center
center_row = image[RES_X // 2, :]
ax2.plot(np.linspace(-1.0, 1.0, RES_Y), center_row)
ax2.set_title("Cross-section through center")
ax2.set_xlabel("y (m)")
ax2.set_ylabel("Pixel value")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("test_pipeline.png", dpi=120)
plt.show()
print("\n  Image saved to: test_pipeline.png")