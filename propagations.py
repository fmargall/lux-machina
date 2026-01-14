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
) -> wp.float32:

    # For computation, normal and direction should have same orientation
    flip = wp.sign(wp.dot(wp.normalize(direction), wp.normalize(normal)))
    
    if flip == -1.:
        normal *= -1.
        nTemp   = ni
        ni      = no
        no      = nTemp

    thetaI = wp.acos(wp.dot(wp.normalize(direction), wp.normalize(normal)))

    # Check for a potential total internal reflection
    sin2ThetaT = wp.pow(ni / no * wp.sin(thetaI), 2.)
    if sin2ThetaT > 1.0:
        return 1.0

    thetaT = wp.asin(ni / no * wp.sin(thetaI))

    rs = (ni * wp.cos(thetaI) - no * wp.cos(thetaT)) / (ni * wp.cos(thetaI) + no * wp.cos(thetaT))
    rp = (ni * wp.cos(thetaT) - no * wp.cos(thetaI)) / (ni * wp.cos(thetaT) + no * wp.cos(thetaI))

    Rs = wp.pow(wp.abs(rs), 2.)
    Rp = wp.pow(wp.abs(rp), 2.)

    return (Rs + Rp) / 2.

@wp.func
def reflect(origin: wp.vec2, normal: wp.vec2) -> wp.vec2:
    # Normalizing
    origin = wp.normalize(origin)
    normal = wp.normalize(normal)

    # Normal and direction should have same orientation
    normal *= wp.sign(wp.dot(origin, normal))

    return origin - 2. * wp.dot(origin, normal) * normal

@wp.func
def refract(
    direction: wp.vec2, normal: wp.vec2, 
    ni: wp.float32    , no: wp.float32
) -> wp.vec2:
    
    # Normalizing
    direction = wp.normalize(direction)
    normal    = wp.normalize(normal)

    # For computation, normal and direction should have same orientation
    flip = wp.sign(wp.dot(direction, normal))
    
    if flip == -1.:
        normal *= -1.
        nTemp   = ni
        ni      = no
        no      = nTemp

    thetaI = wp.acos(wp.dot(direction, normal))
    
    # Check for a potential total internal reflection
    sin2ThetaT = wp.pow(ni / no * wp.sin(thetaI), 2.)
    if sin2ThetaT > 1.0:
        return reflect(direction, normal)

    thetaT = wp.asin(ni / no * wp.sin(thetaI))

    tangent = wp.normalize(direction - wp.dot(direction, normal) * normal)

    outputDirection = wp.cos(thetaT) * normal + wp.sin(thetaT) * tangent
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
        
        if   primitive.type == 0: # Ideal lens
            ray.direction = propagateThroughIdealLens(ray.origin, ray.direction, primitive.v0, primitive.v1, primitive.f0)
        elif primitive.type == 1: # Straight line interface
            ni = primitive.f0
            no = primitive.f1
        elif primitive.type == 2: # Circular arc interface
            ni = primitive.f3
            no = primitive.f4

        if (primitive.type == 1 or primitive.type == 2):
            # Compute Fresnel reflection coefficient
            R = fresnelReflection(ray.direction, intersection.normal, ni, no)
            if rand < R:
                # Reflection
                ray.direction = reflect(ray.direction, intersection.normal)
            else:
                # Refraction
                ray.direction = refract(ray.direction, intersection.normal, ni, no)

        ray.origin = intersection.hitPoint + 1.e-6 * ray.direction
        ray.depth += 1
        raysBuffer[ID] = ray
    else:
        ray.energy = 0.
        raysBuffer[ID] = ray