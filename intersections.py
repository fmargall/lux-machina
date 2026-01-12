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

    # Initializing intersection
    intersection = Intersection()
    intersection.hitPoint = ray.origin + 1.e12 * ray.direction # Initially no intersection
    intersectionsBuffer[ID] = intersection
    
    # Setting ray as dead by default
    raysStatusBuffer[0] = False

    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        if primitive.type == 0: # Segment
            continue
        else:
            continue


