import warp as wp

from intersections import intersect3DRayWithParallelogram
from structures    import Intersection  , Ray  , Segment    , Sensor, \
                          Intersection3D, Ray3D, Primitive3D, Sensor3D


from rasterizer import segmentize, doSegmentsIntersect

@wp.kernel
def accumulateIdealSensor(
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    nbParallelRays     : wp.int32,
    sensor             : Sensor,
    
    sensorBuffer: wp.array(dtype=wp.float32, ndim=1)
):
    # Get pixel ID
    ID = wp.tid()

    # Get associated pixel segment, ie. the sensor
    # whole segment, divided by its pixels number.
    pixelUnit = (sensor.v1 - sensor.v0) / wp.float32(sensor.i0)
    
    pixelSegment = Segment()
    pixelSegment.v0 = sensor.v0 + wp.float32(ID)     * pixelUnit
    pixelSegment.v1 = sensor.v0 + wp.float32(ID + 1) * pixelUnit

    for rayID in range(nbParallelRays):
        ray = intersectionsBuffer[rayID].ray
        raySegment = segmentize(ray, intersectionsBuffer[rayID])

        if doSegmentsIntersect(pixelSegment, raySegment):
            wp.atomic_add(sensorBuffer, ID, ray.energy)

@wp.kernel
def accumulateIdealShackHartmannSensor(
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    nbParallelRays     : wp.int32,
    sensor             : Sensor,

    sensorCountBuffer: wp.array(dtype=wp.float32, ndim=1),
    sensorAngleBuffer: wp.array(dtype=wp.float32, ndim=1)
):
    # Get pixel ID
    ID = wp.tid()

    # Get associated pixel segment, ie. the sensor
    # whole segment, divided by its pixels number.
    pixelUnit = (sensor.v1 - sensor.v0) / wp.float32(sensor.i0)

    sensorTangent = wp.normalize(sensor.v1 - sensor.v0)
    sensorNormal  = wp.vec2(-sensorTangent.y, sensorTangent.x)
    
    pixelSegment = Segment()
    pixelSegment.v0 = sensor.v0 + wp.float32(ID)     * pixelUnit
    pixelSegment.v1 = sensor.v0 + wp.float32(ID + 1) * pixelUnit

    for rayID in range(nbParallelRays):
        ray = intersectionsBuffer[rayID].ray
        raySegment = segmentize(ray, intersectionsBuffer[rayID])

        if doSegmentsIntersect(pixelSegment, raySegment):
            # Add count to current pixel, as a normal sensor
            wp.atomic_add(sensorCountBuffer, ID, ray.energy)

            # Then save ray direction for ideal Shack-Hartmann
            angle = wp.acos(wp.dot(sensorNormal, ray.direction))
            wp.atomic_add(sensorAngleBuffer, ID, angle)

@wp.kernel
def accumulateIdealPlenopticSensor(
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    nbParallelRays     : wp.int32,
    sensor             : Sensor,

    plenopticSensorBuffer: wp.array(dtype=wp.float32, ndim=2)
):
    # Get current thread ID
    spatialBinID = wp.tid()

    # Get associated pixel segment, ie. the sensor
    # whole segment, divided by its pixels number.
    pixelUnit = (sensor.v1 - sensor.v0) / wp.float32(sensor.i0)

    sensorTangent = wp.normalize(sensor.v1 - sensor.v0)
    sensorNormal  = wp.vec2(-sensorTangent.y, sensorTangent.x)
    
    pixelSegment = Segment()
    pixelSegment.v0 = sensor.v0 + wp.float32(spatialBinID)     * pixelUnit
    pixelSegment.v1 = sensor.v0 + wp.float32(spatialBinID + 1) * pixelUnit

    for rayID in range(nbParallelRays):
        ray = intersectionsBuffer[rayID].ray
        raySegment = segmentize(ray, intersectionsBuffer[rayID])

        if doSegmentsIntersect(pixelSegment, raySegment):
            # Reading oriented angle between sensor normal and ray
            cos = wp.dot(sensorNormal   , ray.direction)
            sin =        sensorNormal.x * ray.direction.y - sensorNormal.y * ray.direction.x
            orientedAngle = wp.atan2(sin, cos)

            # Get associated angular bin ID
            if (orientedAngle >= -wp.half_pi and orientedAngle <= wp.half_pi):
                angularBinID = wp.int32(wp.float32(sensor.i1) * (orientedAngle + wp.half_pi) / wp.pi)

                wp.atomic_add(plenopticSensorBuffer, spatialBinID, angularBinID, ray.energy)

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
    pixel = Primitive3D()
    pixel.type = 5 # Parallelogram
    pixelCenter = sensorCenter + u * pixelTangent + v * pixelBitangent
    pixel.v0 = pixelCenter - pixelTangent / wp.float32(2.) - pixelBitangent / wp.float32(2.)
    pixel.v1 = pixelCenter + pixelTangent / wp.float32(2.) - pixelBitangent / wp.float32(2.)
    pixel.v2 = pixelCenter + pixelTangent / wp.float32(2.) + pixelBitangent / wp.float32(2.)
    pixel.v3 = pixelCenter - pixelTangent / wp.float32(2.) + pixelBitangent / wp.float32(2.)

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

@wp.kernel
def accumulateIdealPlenoptic3DSensor(
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    nbParallelRays     : wp.int32,
    sensor             : Sensor3D,

    plenopticSensorBuffer: wp.array(dtype=wp.float32, ndim=4)
):
    pass