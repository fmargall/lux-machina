import warp as wp

from structures import Intersection, Primitive, Ray

@wp.kernel
def intersectRays(
    raysBuffer: wp.array(dtype=Ray, ndim=1),
    primitivesBuffer: wp.array(dtype=Primitive, ndim=1),
    nbPrimitives: wp.int32,

    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    raysStatusBuffer: wp.array(dtype=wp.bool, ndim=1)
):
    ID = wp.tid()

    ray = raysBuffer[ID]
    
    # Saving intersection properties
    intersection = Intersection()
    intersection.hitPoint = ray.origin + 1000.0 * ray.direction # No intersection, far away
    intersectionsBuffer[ID] = intersection

    # Saving ray status
    raysStatusBuffer[0] = ray.alive
