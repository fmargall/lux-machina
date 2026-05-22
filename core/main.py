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

from _structures         import (_Ray, _LightSource, _Primitive, _Sensor, _Intersection)
from _generateRays       import _generateRays
from _accumulateOnSensor import _accumulateOnSensor


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

# Biconvex lens
biconvexLens01 = _Primitive()
biconvexLens01.type = 3
biconvexLens01.v0   = wp.vec3f(0., 0., 0.0984)
biconvexLens01.v1   = wp.vec3f(0., 0., -1.)
biconvexLens01.f0   = wp.float32(0.0592)
biconvexLens01.f1   = wp.float32(0.447189)


# Sensor: 2m x 2m parallelogram at z=0, normal toward +z.
# Vertex order chosen so that cross(v1-v0, v3-v0) points along +z (toward source).
sensor      = _Sensor()
sensor.type = wp.int32(0)                  # flat radiometer
sensor.v0   = wp.vec3f(-0.003, -0.003, 0.000125)
sensor.v1   = wp.vec3f( 0.003, -0.003, 0.000125)
sensor.v2   = wp.vec3f( 0.003,  0.003, 0.000125)
sensor.v3   = wp.vec3f(-0.003,  0.003, 0.000125)
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
"""
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
"""
#print("Accumulating on sensor...")
wp.launch(
    _accumulateOnSensor,
    dim    = N_RAYS,
    inputs = [rayBuffer, intersectionBuffer, sensor, sensorBuffer],
)

wp.synchronize()


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