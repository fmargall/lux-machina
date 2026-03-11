import warp as wp

from structures import Intersection3D, Ray3D

@wp.func
def fresnelReflection3D(
    rayDirection: wp.vec3f, interfaceNormal: wp.vec3f, 
    ni: wp.float32        , no: wp.float32
) -> wp.float32:
    # Normalizing 
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal) 

    """
    # If the ray comes from the outside of the primitive 
    # the dot product will be negative. Then, we have to 
    # flip the normal and the refracive indexes. 
    if wp.dot(rayDirection, interfaceNormal) < 0.:
        interfaceNormal = - interfaceNormal
        ni, no = no, ni

    thetaI = wp.acos(wp.dot(rayDirection, interfaceNormal))
    """
    cosThetaI = - wp.dot(rayDirection, interfaceNormal)
    if cosThetaI <= wp.float32(0.):
        interfaceNormal = - interfaceNormal
        cosThetaI       = - wp.dot(rayDirection, interfaceNormal)
        ni, no          =   no, ni

    thetaI = wp.acos(cosThetaI)

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

"""
   :param rayDirection: A light vector, pointing from the light source
                        towards the surface, either normalized or not.
   :type rayDirection: wp.vec3
   :param interfaceNormal: A vector normal to the interface, either 
                           normalized or not that points towards ni
   :type interfaceNormal: wp.vec3
   :return: A light vector, pointing from the interface towards the
            the reflected direction.
   :rtype: wp.vec3
"""
@wp.func
def reflect3D(rayDirection: wp.vec3f, interfaceNormal: wp.vec3f) -> wp.vec3f:
    # Normalizing
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal)

    return rayDirection - 2. * wp.dot(rayDirection, interfaceNormal) * interfaceNormal

"""
   :param rayDirection: A light vector, pointing from the light source
                        towards the surface, either normalized or not.
   :type rayDirection: wp.vec3
   :param interfaceNormal: A vector normal to the interface, either 
                           normalized or not that points towards ni
   :type interfaceNormal: wp.vec3
   :param ni: Index of refraction of domain where normal points at.
   :type ni: wp.float32
   :param no: Index of refraction of domain where normal goes out.
   :type no: wp.float32
   :return: A light vector, pointing from the interface towards the
            the reflected direction.
   :rtype: wp.vec3
"""
@wp.func
def refract3D(rayDirection: wp.vec3f, interfaceNormal: wp.vec3f, 
              ni: wp.float32        , no: wp.float32
) -> wp.vec3f: 
    # Wikipedia version
    # Normalizing
    rayDirection    = wp.normalize(rayDirection)
    interfaceNormal = wp.normalize(interfaceNormal)

    cosThetaI = - wp.dot(rayDirection, interfaceNormal)
    if cosThetaI <= wp.float32(0.):
        interfaceNormal = - interfaceNormal
        cosThetaI       = - wp.dot(rayDirection, interfaceNormal)
        ni, no          =   no, ni

    radicand  = 1. - (wp.pow((ni / no), 2.)) * (1. - wp.pow(cosThetaI, 2.))

    # Total internal reflection
    if radicand < wp.float32(0):
        return reflect3D(rayDirection, interfaceNormal) 

    cosThetaT = wp.sqrt(radicand)

    refractedDirection = (ni / no) * rayDirection + \
                        ((ni / no) * cosThetaI - cosThetaT) * interfaceNormal

    return wp.normalize(refractedDirection)

@wp.kernel
def propagate3DRays(
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    seed: wp.int32,

    raysBuffer: wp.array(dtype=Ray3D, ndim=1)
):
    ID = wp.tid()
    
    seed = (seed * raysBuffer.shape[0] + ID) * 1664525 + 1013904223
    rand = wp.randf(wp.uint32(seed))

    intersection = intersectionsBuffer[ID]
    ray = intersection.ray

    if ray.isAlive and intersection.hit:

        primitive = intersection.primitive

        ni = wp.float32(0.0)
        no = wp.float32(0.0)

        if primitive.type == 1:  # Cylinder
            ni = primitive.f1
            no = primitive.f2
        elif primitive.type == 2:  # Spherical cap
            ni = primitive.f1
            no = primitive.f2
        elif primitive.type == 3:  # Aspheric lens
            ni = primitive.f2
            no = primitive.f3

        # Fresnel
        R = fresnelReflection3D(ray.direction, intersection.normal, ni, no)

        newDirection = ray.direction

        if rand < R:
            newDirection = reflect3D(ray.direction, intersection.normal)
        else:
            newDirection = refract3D(ray.direction, intersection.normal, ni, no)

        newOrigin = intersection.hitPoint + 5.e-7 * newDirection

        raysBuffer[ID] = Ray3D(
            isAlive   = True,
            origin    = newOrigin,
            direction = newDirection,
            depth     = ray.depth + 1,
            energy    = ray.energy
        )

    else:
        raysBuffer[ID] = Ray3D(
            isAlive   = False,
            origin    = ray.origin,
            direction = ray.direction,
            depth     = ray.depth,
            energy    = wp.float32(0.0)
        )