import warp as wp

from structures import Intersection, Ray, Segment, Sensor

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


