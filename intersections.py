import warp as wp

from structures import Intersection, Primitive, Ray

@wp.func
def fresnel(
    theta1: wp.float32,
    n1    : wp.float32,
    n2    : wp.float32
) -> wp.float32:

    theta2 = wp.asin(n1 * wp.sin(theta1) / n2)

    rs = (n1 * wp.cos(theta1) - n2 * wp.cos(theta2)) / (n1 * wp.cos(theta1) + n2 * wp.cos(theta2))
    rp = (n1 * wp.cos(theta2) - n2 * wp.cos(theta1)) / (n1 * wp.cos(theta2) + n2 * wp.cos(theta1))

    return (wp.pow(wp.abs(rs), 2.) + wp.pow(wp.abs(rp), 2.)) / 2.

@wp.func
def raySegmentIntersection(
    ray    : Ray,
    segment: Primitive
) -> Intersection:
    intersection = Intersection()

    # Check if ray and segment intersect
    intersectionCatched = False

    if intersectionCatched:
        intersection.missed   = False
        #intersection.hitPoint = hitPoint
        #intersection.normal   = normal
        intersection.ni       = segment.ni
        intersection.no       = segment.no
    
    # If not, return a failed intersection
    intersection.missed = True

    return intersection

@wp.kernel
def intersectRays(
    raysBuffer      : wp.array(dtype=Ray, ndim=1),
    primitivesBuffer: wp.array(dtype=Primitive, ndim=1),
    nbPrimitives    : wp.int32,

    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    raysStatusBuffer   : wp.array(dtype=wp.bool, ndim=1)
):
    ID = wp.tid()

    ray = raysBuffer[ID]
    intersection = Intersection()
    intersection.missed = False

    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        # Different primitives types exist
        if   primitive.type == 0: # 0: segment
            tempIntersection = raySegmentIntersection(ray, primitive)
        #elif primitive.type == 1: # 1: circular arc
        #    pass
        else: # Unknown primitive
            pass

    intersectionsBuffer[ID] = intersection

    # Saving ray status
    raysStatusBuffer[0] = ray.alive

@wp.kernel
def propagateRays(
    raysBuffer         : wp.array(dtype=Ray, ndim=1),
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),

    seed: wp.int32
):
    ID  = wp.tid()
    ray = raysBuffer[ID]
    intersection = intersectionsBuffer[ID]

    seed = seed * raysBuffer.shape[0] + ID
    rand = wp.randf(wp.uint32(seed))

    if ray.alive:
        hitPoint = intersection.hitPoint
        normal   = intersection.normal

        thetaI = wp.acos(wp.dot(normal, -ray.direction))
        R = fresnel(thetaI, intersection.ni, intersection.no)

        if rand <= R:
            # The ray will be reflected
            ray.direction = ray.direction - 2.0 * wp.dot(ray.direction, normal) * normal
        else:
            # The ray will be refracted
            thetaO = wp.asin(intersection.ni * wp.sin(thetaI) / intersection.no)
            ray.direction = (intersection.ni / intersection.no) * (ray.direction + wp.cos(thetaI) * normal) - wp.cos(thetaO) * normal

        # Updating ray properties
        # Weight is not changed here, since the sampling of
        # reflection/refraction is already accounted in Fresnel
        ray.origin    = hitPoint
        ray.depth    += 1
        ray.alive     = True