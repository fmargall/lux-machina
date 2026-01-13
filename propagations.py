import warp as wp

from intersections import cross
from structures    import Intersection, Ray

@wp.func
def intersectLines(
    originA   : wp.vec2,
    directionA: wp.vec2,
    originB   : wp.vec2,
    directionB: wp.vec2
) -> wp.vec2:

    directionA = wp.normalize(directionA)
    directionB = wp.normalize(directionB)

    t = cross(originB - originA, directionB) / cross(directionA, directionB)
    return originA + t * directionA

@wp.func
def propagateThroughIdealLens(
    origin     : wp.vec2,
    direction  : wp.vec2, 
    v0         : wp.vec2, 
    v1         : wp.vec2, 
    focalLength: wp.float32
) -> wp.vec2:
    lensCenter  = (v0 + v1) / 2.
    lensTangent = wp.normalize(v1 - v0)
    lensNormal  = wp.vec2(-lensTangent.y, lensTangent.x)
    lensNormal *= wp.sign(wp.dot(direction, lensNormal))

    firstFocal  = lensCenter - focalLength * lensNormal
    secondFocal = lensCenter + focalLength * lensNormal

    # 1. Get position of ray on the focal plane
    rayOnFocalPlaneIntersection = intersectLines(origin, direction, firstFocal, lensTangent)

    # 2. Project this point on the lens plane, then create ghost ray through second focal
    ghostRayOnLensPlaneIntersection     = intersectLines(rayOnFocalPlaneIntersection, lensNormal, lensCenter, lensTangent)
    ghostRayThroughSecondFocalDirection = wp.normalize(secondFocal - ghostRayOnLensPlaneIntersection)

    # 3. Project the ray position on the focal plane through the center, without deviation
    ghostRayThroughCenterDirection = wp.normalize(lensCenter - rayOnFocalPlaneIntersection)

    # 4. Find the intersection of both
    ghostRaysIntersection = intersectLines(
        ghostRayOnLensPlaneIntersection, ghostRayThroughSecondFocalDirection,
        rayOnFocalPlaneIntersection    , ghostRayThroughCenterDirection)

    # 5. Get position of ray on the lens plane
    rayOnLensPlaneProjection = intersectLines(origin, direction, lensCenter, lensTangent)

    # 6. Output direction goes from rayOnLensPlaneProjection,
    #    to ghostRaysIntersection. Then normalize it and done
    result = wp.normalize(ghostRaysIntersection - rayOnLensPlaneProjection)

    # 7. Ideal lens transmits everything. So even if the intersection
    #    of the ghost rays is before the lens, the ray will continue.
    result *= wp.sign(wp.dot(result, lensNormal))

    return result
    

@wp.func
def fresnelReflection(
    direction: wp.vec2, normal: wp.vec2, 
    ni: wp.float32    , no: wp.float32
) -> wp.vec2:

    thetaI = wp.acos(wp.dot(-direction, normal))
    thetaT = wp.asin(ni / no * wp.sin(thetaI))

    rs = (ni * wp.cos(thetaI) - no * wp.cos(thetaT)) / (ni * wp.cos(thetaI) + no * wp.cos(thetaT))
    rp = (ni * wp.cos(thetaT) - no * wp.cos(thetaI)) / (ni * wp.cos(thetaT) + no * wp.cos(thetaI))

    Rs = wp.pow(wp.abs(rs), 2.)
    Rp = wp.pow(wp.abs(rp), 2.)

    return (Rs + Rp) / 2.

@wp.func
def reflect(origin: wp.vec2, normal: wp.vec2) -> wp.vec2:
    return origin - 2. * wp.dot(origin, normal) * normal

@wp.func
def refract(
    direction: wp.vec2, normal: wp.vec2, 
    ni: wp.float32    , no: wp.float32
) -> wp.vec2:

    thetaI = wp.acos(wp.dot(-direction, normal))
    thetaT = wp.asin(ni / no * wp.sin(thetaI))

    outputDirection = wp.cos(thetaT) * (-normal) + wp.sin(thetaT) * wp.vec2(-normal.y, normal.x)
    return wp.normalize(outputDirection)

@wp.kernel
def propagateRays(
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    seed: wp.int32,

    raysBuffer: wp.array(dtype=Ray, ndim=1)
):
    ID = wp.tid()
    
    seed = seed * raysBuffer.shape[0] + ID
    rand = wp.randf(wp.uint32(seed))

    intersection = intersectionsBuffer[ID]
    ray = intersection.ray

    if ray.isAlive:
        primitive = intersection.primitive
        
        if primitive.type == 0: # Ideal lens
            ray.direction = propagateThroughIdealLens(ray.origin, ray.direction, primitive.v0, primitive.v1, primitive.f0)
        """
        # Compute Fresnel reflection coefficient
        R = fresnelReflection(ray.direction, intersection.normal, intersection.ni, intersection.no)
        if rand < R:
            # Reflection
            ray.direction = reflect(ray.direction, intersection.normal)
        else:
            # Refraction
            ray.direction = refract(ray.direction, intersection.normal, intersection.ni, intersection.no)
        """

        ray.origin = intersection.hitPoint + 1.e-5 * ray.direction
        ray.depth += 1
        raysBuffer[ID] = ray
    else:
        ray.energy = 0.
        raysBuffer[ID] = ray