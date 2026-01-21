import warp as wp

from structures import Intersection, Ray, Segment, Sensor

from rasterizer import segmentize, doSegmentsIntersect

@wp.kernel
def accumulateSensor(
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
