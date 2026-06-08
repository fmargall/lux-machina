import warp as wp

from _structures import _Camera, _Intersection, _Primitive, _Ray, _Sensor

@wp.kernel
def _accumulateOnSensor(
    # --- Input buffers ---
    rayBuffer         : wp.array(dtype=_Ray),
    intersectionBuffer: wp.array(dtype=_Intersection),

    # --- Scene data ---
    sensor: _Sensor,

    # --- Output buffer ---
    sensorBuffer: wp.array(dtype=wp.float32, ndim=2),
):
    ID           = wp.tid()
    ray          = rayBuffer[ID]
    intersection = intersectionBuffer[ID]

    # Skip dead rays
    if not ray.isAlive:
        return

    # Get sensor's support
    v0    = sensor.v0
    edgeU = sensor.v1 - v0
    edgeV = sensor.v3 - v0

    # Sensor plane normal
    sensorNormal = wp.normalize(wp.cross(edgeU, edgeV))

    # Ray-plane intersection
    denom = wp.dot(ray.direction, sensorNormal)

    # Reject parallel rays
    if wp.abs(denom) < 1.0e-8:
        return

    tSensor = wp.dot(v0 - ray.origin, sensorNormal) / denom

    # Reject rays going backward
    if tSensor <= 0.0:
        return

    # Visibility: scene blocked the ray before the sensor?
    if intersection.t > 0.0 and intersection.t < tSensor:
        return

    # Express hit point in the sensor's local (u, v) coordinates
    hitPoint = ray.origin + tSensor * ray.direction
    rel      = hitPoint - v0
    u        = wp.dot(rel, edgeU) / wp.dot(edgeU, edgeU)
    v        = wp.dot(rel, edgeV) / wp.dot(edgeV, edgeV)

    # Outside the sensor surface?
    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return

    # Continuous pixel coordinates (pixel centers at integer + 0.5)
    resX = sensorBuffer.shape[0]
    resY = sensorBuffer.shape[1]
    x    = u * wp.float32(resX) - 0.5
    y    = v * wp.float32(resY) - 0.5

    # Bilinear (tent) splatting on the 4 nearest pixels
    i0 = wp.int32(wp.floor(x))
    j0 = wp.int32(wp.floor(y))
    i1 = i0 + 1
    j1 = j0 + 1
    dx = x - wp.float32(i0)
    dy = y - wp.float32(j0)

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx         * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx         * dy

    e = ray.throughput

    if i0 >= 0 and i0 < resX and j0 >= 0 and j0 < resY:
        wp.atomic_add(sensorBuffer, i0, j0, w00 * e)
    if i1 >= 0 and i1 < resX and j0 >= 0 and j0 < resY:
        wp.atomic_add(sensorBuffer, i1, j0, w10 * e)
    if i0 >= 0 and i0 < resX and j1 >= 0 and j1 < resY:
        wp.atomic_add(sensorBuffer, i0, j1, w01 * e)
    if i1 >= 0 and i1 < resX and j1 >= 0 and j1 < resY:
        wp.atomic_add(sensorBuffer, i1, j1, w11 * e)



@wp.func
def _projectWorldToCamera(worldPoint: wp.vec3f,
                          camera    : _Camera) -> wp.vec3f:
    """
    Projects a world point onto the image plane via the OpenCV pinhole model
    Returns (u, v, depth). If depth is <= 0, the point is behind the camera.
    """
    # Extrinsics: from world to the camera frame:
    cameraPoint = camera.m0 * worldPoint + camera.v0

    # Check if point is behind the camera
    if cameraPoint[2] <= wp.float32(0.0):
        return wp.vec3f(0.0, 0.0, -1.0)

    # Perspective projection
    xPrime = cameraPoint[0] / cameraPoint[2]
    yPrime = cameraPoint[1] / cameraPoint[2]

    # Compute all radial distances squared
    r2 = xPrime * xPrime + yPrime * yPrime
    r4 = r2 * r2
    r6 = r4 * r2

    # Radial distortion (full 8-coefficient model)
    radialNumerator   = wp.float32(1.0) + camera.f4 * r2 + camera.f5 * r4 + camera.f6 * r6 # 1 + k1 * r2 + k2 * r4 + k3 * r6
    radialDenominator = wp.float32(1.0) + camera.f7 * r2 + camera.f8 * r4 + camera.f9 * r6 # 1 + k4 * r2 + k5 * r4 + k6 * r6
    radialDistortion  = radialNumerator / radialDenominator

    # Tangential distortion
    dxTangential = wp.float32(2.) * camera.f10 * xPrime * yPrime + camera.f11 * (r2 + wp.float32(2.) * xPrime * xPrime)
    dyTangential = camera.f10 * (r2 + wp.float32(2.) * yPrime * yPrime) + wp.float32(2.) * camera.f11 * xPrime * yPrime

    # Thin prism distortion
    dxPrism = camera.f12 * r2 + camera.f13 * r4
    dyPrism = camera.f14 * r2 + camera.f15 * r4

    # Combined distorted coordinates
    xDistorted = xPrime * radialDistortion + dxTangential + dxPrism
    yDistorted = yPrime * radialDistortion + dyTangential + dyPrism

    # Apply intrinsics: pixel coordinates
    u = camera.f0 * xDistorted + camera.f2 # fx * xDistorted + cx
    v = camera.f1 * yDistorted + camera.f3 # fy * yDistorted + cy

    return wp.vec3f(u, v, cameraPoint[2])

@wp.kernel
def _checkLambertianPlateAndAccumulateOnCamera(
    # --- Input buffers ---
    rayBuffer         : wp.array(dtype=_Ray),
    intersectionBuffer: wp.array(dtype=_Intersection),
    # --- Scene data ---
    camera            : _Camera,
    lambertianPlateHit: _Primitive,
    # --- Output buffer ---
    cameraBuffer: wp.array(dtype=wp.float32, ndim=2),
):
    """
    Simplified next-event estimation: for each ray, test if it reaches the
    Lambertian plate WITHOUT being blocked by another primitive first.
    If it does, project the hit point onto the camera and accumulate.

    Future improvements:
      - Add a shadow ray (visibility test) between hit point and camera.
      - Loop over multiple Lambertian primitives.
    """

    ID           = wp.tid()
    ray          = rayBuffer[ID]

    # Skip dead rays
    if not ray.isAlive:
        return

    # ── 1. Intersect ray with the Lambertian plate ──
    plateU      = lambertianPlate.v1 - lambertianPlate.v0
    plateV      = lambertianPlate.v3 - lambertianPlate.v0
    plateNormal = wp.normalize(wp.cross(plateU, plateV))

    denom = wp.dot(ray.direction, plateNormal)
    if wp.abs(denom) < 1.0e-8:
        return   # ray parallel to the plate

    tPlate = wp.dot(lambertianPlate.v0 - ray.origin, plateNormal) / denom
    if tPlate <= 0.0:
        return   # plate is behind the ray

    hitPoint = ray.origin + tPlate * ray.direction

    # Check that the hit is within the parallelogram bounds
    relPos = hitPoint - lambertianPlate.v0
    u      = wp.dot(relPos, plateU) / wp.dot(plateU, plateU)
    v      = wp.dot(relPos, plateV) / wp.dot(plateV, plateV)

    if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
        return   # outside the parallelogram

    # ── 2. Check if the ray is blocked BEFORE reaching the plate ──
    intersection = intersectionBuffer[ID]
    if intersection.t > 0.0 and intersection.t < tPlate:
        return   # blocked by another primitive

    # ── 3. Project the hit point onto the camera image plane ──
    proj = _projectWorldToCamera(hitPoint, camera)
    if proj[2] <= 0.0:
        return   # behind camera or outside image

    # ── 4. Compute Lambertian contribution ──
    cosIncident = wp.dot(ray.direction, plateNormal)
    if cosIncident > 0.0:
        plateNormal = -plateNormal
    cosTheta = -wp.dot(ray.direction, plateNormal)

    albedo       = wp.float32(1.0)
    brdf         = albedo / wp.pi
    contribution = ray.throughput * brdf * cosTheta

    # ── 5. Bilinear splat onto the camera buffer ──
    pixelU = proj[0]
    pixelV = proj[1]

    i0 = wp.int32(wp.floor(pixelU))
    j0 = wp.int32(wp.floor(pixelV))
    i1 = i0 + 1
    j1 = j0 + 1

    du = pixelU - wp.float32(i0)
    dv = pixelV - wp.float32(j0)

    w00 = (1.0 - du) * (1.0 - dv)
    w10 = du         * (1.0 - dv)
    w01 = (1.0 - du) * dv
    w11 = du         * dv

    W = camera.i0
    H = camera.i1

    if i0 >= 0 and i0 < W and j0 >= 0 and j0 < H:
        wp.atomic_add(cameraBuffer, j0, i0, w00 * contribution)
    if i1 >= 0 and i1 < W and j0 >= 0 and j0 < H:
        wp.atomic_add(cameraBuffer, j0, i1, w10 * contribution)
    if i0 >= 0 and i0 < W and j1 >= 0 and j1 < H:
        wp.atomic_add(cameraBuffer, j1, i0, w01 * contribution)
    if i1 >= 0 and i1 < W and j1 >= 0 and j1 < H:
        wp.atomic_add(cameraBuffer, j1, i1, w11 * contribution)