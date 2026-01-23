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
    rayDirection: wp.vec2, interfaceNormal: wp.vec2, 
    ni: wp.float32       , no: wp.float32
) -> wp.float32:
    # Normalizing 
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal) 

    # If the ray comes from the outside of the primitive 
    # the dot product will be negative. Then, we have to 
    # flip the normal and the refracive indexes. 
    if wp.dot(rayDirection, interfaceNormal) < 0.:
        interfaceNormal = - interfaceNormal
        ni, no = no, ni

    thetaI = wp.acos(wp.dot(rayDirection, interfaceNormal))

    # Check for a potential total internal reflection
    sineTransmissionAngle = (ni / no) * wp.sin(thetaI) 
    if wp.abs(sineTransmissionAngle) > wp.float32(1.0): 
        return  wp.float32(1.0)

    thetaT = wp.asin(sineTransmissionAngle)

    rs = (ni * wp.cos(thetaI) - no * wp.cos(thetaT)) / (ni * wp.cos(thetaI) + no * wp.cos(thetaT))
    rp = (ni * wp.cos(thetaT) - no * wp.cos(thetaI)) / (ni * wp.cos(thetaT) + no * wp.cos(thetaI))

    Rs = wp.pow(wp.abs(rs), 2.)
    Rp = wp.pow(wp.abs(rp), 2.)

    return (Rs + Rp) / 2.

@wp.func
def reflect(rayDirection: wp.vec2, interfaceNormal: wp.vec2) -> wp.vec2:
    # Normalizing
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal)

    return rayDirection - 2. * wp.dot(rayDirection, interfaceNormal) * interfaceNormal

"""
   By convention, interfaceNormal always points towards ni.
"""
@wp.func
def refract(rayDirection: wp.vec2, interfaceNormal: wp.vec2, 
            ni: wp.float32       , no: wp.float32
) -> wp.vec2: 
    # Normalizing 
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal) 
    
    # If the ray comes from the outside of the primitive 
    # the dot product will be negative. Then, we have to 
    # flip the normal and the refracive indexes. 
    if wp.dot(rayDirection, interfaceNormal) < 0.:
        interfaceNormal = - interfaceNormal
        ni, no = no, ni
        
    incidentAngle = wp.acos(wp.dot(rayDirection, interfaceNormal))
    
    sineTransmissionAngle = (ni / no) * wp.sin(incidentAngle) 
    # Check for a potential total internal reflection 
    if wp.abs(sineTransmissionAngle) > wp.float32(1.0): 
        return reflect(rayDirection, interfaceNormal) 
        
    # Tangent must follow the direction of the original ray direction: 
    interfaceTangent = wp.vec2(-interfaceNormal.y, interfaceNormal.x)
    interfaceTangent *= wp.sign(wp.dot(interfaceTangent, rayDirection))
    
    transmissionAngle = wp.asin(sineTransmissionAngle) 
    refractedDirection = wp.cos(transmissionAngle) * interfaceNormal \
        + wp.sin(transmissionAngle) * interfaceTangent 
        
    return wp.normalize(refractedDirection)

@wp.kernel
def propagateRays(
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    seed: wp.int32,

    raysBuffer: wp.array(dtype=Ray, ndim=1)
):
    ID = wp.tid()
    
    seed = (seed * raysBuffer.shape[0] + ID) * 1664525 + 1013904223
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
        elif primitive.type == 3: # Aspheric lens interface
            ni = primitive.f2
            no = primitive.f3

        if (primitive.type == 1 or primitive.type == 2 or primitive.type == 3):
            # Compute Fresnel reflection coefficient
            R = fresnelReflection(ray.direction, intersection.normal, ni, no)
            if rand < R:
                # Reflection
                ray.direction = reflect(ray.direction, intersection.normal)
            else:
                # Refraction
                ray.direction = refract(ray.direction, intersection.normal, ni, no)

        ray.origin = intersection.hitPoint + 5.e-7 * ray.direction
        ray.depth += 1
        raysBuffer[ID] = ray
    else:
        ray.energy = 0.
        raysBuffer[ID] = ray