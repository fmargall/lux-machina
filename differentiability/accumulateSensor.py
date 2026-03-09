import warp as wp

from intersections import intersect3DRayWithParallelogram
from structures    import Intersection3D, Ray3D, Primitive3D, Sensor3D

"""
@wp.kernel
def accumulateIdeal3DSensor(
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    nbParallelRays     : wp.int32,
    sensor             : Sensor3D,
    
    sensorBuffer: wp.array(dtype=wp.float32, ndim=2)
):
    # Get pixel IDs
    iID, jID = wp.tid()
    i, j = wp.float32(iID), wp.float32(jID)

    nbPixelsTangent   = sensorBuffer.shape[0]
    nbPixelsBitangent = sensorBuffer.shape[1]

    sensorCenter   = sensor.v0
    pixelTangent   = sensor.v1 / wp.float32(nbPixelsTangent)
    pixelBitangent = sensor.v2 / wp.float32(nbPixelsBitangent)

    u = (i + wp.float32(0.5) - wp.float32(nbPixelsTangent)   / wp.float32(2.))
    v = (j + wp.float32(0.5) - wp.float32(nbPixelsBitangent) / wp.float32(2.))

    # Get associated pixel parallelogram
    pixelCenter = sensorCenter + u * pixelTangent + v * pixelBitangent
    pixel = Primitive3D(
        type = 5, # Parallelogram
        v0   = pixelCenter - pixelTangent / wp.float32(2.) - pixelBitangent / wp.float32(2.) ,
        v1   = pixelCenter + pixelTangent / wp.float32(2.) - pixelBitangent / wp.float32(2.) ,
        v2   = pixelCenter + pixelTangent / wp.float32(2.) + pixelBitangent / wp.float32(2.) ,
        v3   = pixelCenter - pixelTangent / wp.float32(2.) + pixelBitangent / wp.float32(2.) ,
        f0   = wp.float32(0.), f1 = wp.float32(0.), f2  = wp.float32(0.), f3 = wp.float32(0.),
        f4   = wp.float32(0.), f5 = wp.float32(0.), f6  = wp.float32(0.), f7 = wp.float32(0.),
        f8   = wp.float32(0.), f9 = wp.float32(0.), f10 = wp.float32(0.)
    )

    for rayID in range(nbParallelRays):
        ray = intersectionsBuffer[rayID].ray

        intersect = intersect3DRayWithParallelogram(ray, pixel)
        if intersect.hit:

            # The infinite ray intersects the pixel parallelogram.
            # We need to check if another intersection has occured
            # before reaching the sensor.
            if intersectionsBuffer[rayID].hit:
                # Is the intersection point before the sensor along the ray?
                distToHit    = wp.norm_l2(intersectionsBuffer[rayID].hitPoint - ray.origin)
                distToSensor = wp.norm_l2(intersect.hitPoint - ray.origin)

                if distToSensor < distToHit:
                    wp.atomic_add(sensorBuffer, wp.int32(i), wp.int32(j), ray.energy)

            else:
                wp.atomic_add(sensorBuffer, wp.int32(i), wp.int32(j), ray.energy)
"""

@wp.kernel
def accumulateIdeal3DSensor(
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    sensor             : Sensor3D,

    sensorBuffer       : wp.array(dtype=wp.float32, ndim=2)
):
    rayID = wp.tid()

    intersection = intersectionsBuffer[rayID]
    ray          = intersection.ray

    nbPixelsTangent   = sensorBuffer.shape[0]
    nbPixelsBitangent = sensorBuffer.shape[1]

    sensorCenter = sensor.v0
    sensorAxisU  = sensor.v1
    sensorAxisV  = sensor.v2

    sensorNormal = wp.normalize(wp.cross(sensorAxisU, sensorAxisV))

    # Ray / sensor plane intersection
    denom = wp.dot(ray.direction, sensorNormal)

    # Reject parallel rays and rays going away from the sensor plane
    if wp.abs(denom) < 1.0e-8:
        return

    tSensor = wp.dot(sensorCenter - ray.origin, sensorNormal) / denom

    if tSensor <= 0.0:
        return

    hitPoint = ray.origin + tSensor * ray.direction

    # Optional visibility test:
    # if another primitive has already been hit before the sensor, discard
    if intersection.hit:
        tHit = wp.norm_l2(intersection.hitPoint - ray.origin)
        if tHit < tSensor:
            return

    # Express hit point in the sensor basis
    rel = hitPoint - sensorCenter

    uu = wp.dot(rel, sensorAxisU) / wp.dot(sensorAxisU, sensorAxisU)
    vv = wp.dot(rel, sensorAxisV) / wp.dot(sensorAxisV, sensorAxisV)

    # Sensor is the parallelogram centered on sensorCenter
    # spanning [-0.5, +0.5] along both axes
    if uu < -0.5 or uu > 0.5 or vv < -0.5 or vv > 0.5:
        return

    # Continuous pixel coordinates
    x = (uu + 0.5) * wp.float32(nbPixelsTangent)   - 0.5
    y = (vv + 0.5) * wp.float32(nbPixelsBitangent) - 0.5

    i0 = wp.int32(wp.floor(x))
    j0 = wp.int32(wp.floor(y))
    i1 = i0 + 1
    j1 = j0 + 1

    dx = x - wp.float32(i0)
    dy = y - wp.float32(j0)

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    e = ray.energy

    if i0 >= 0 and i0 < nbPixelsTangent and j0 >= 0 and j0 < nbPixelsBitangent:
        wp.atomic_add(sensorBuffer, i0, j0, w00 * e)

    if i1 >= 0 and i1 < nbPixelsTangent and j0 >= 0 and j0 < nbPixelsBitangent:
        wp.atomic_add(sensorBuffer, i1, j0, w10 * e)

    if i0 >= 0 and i0 < nbPixelsTangent and j1 >= 0 and j1 < nbPixelsBitangent:
        wp.atomic_add(sensorBuffer, i0, j1, w01 * e)

    if i1 >= 0 and i1 < nbPixelsTangent and j1 >= 0 and j1 < nbPixelsBitangent:
        wp.atomic_add(sensorBuffer, i1, j1, w11 * e)