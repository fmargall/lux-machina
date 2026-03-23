import warp as wp

from .structures import _Intersection3D, _Primitive3D, _Ray3D, _IN_OUT, _OUT_IN


@wp.func
def asphericLensProfile(
    x  : wp.float32,
    R  : wp.float32, k  : wp.float32,
    y0 : wp.float32, N  : wp.float32,
    a2 : wp.float32, a4 : wp.float32, a6 : wp.float32, a8 : wp.float32, a10: wp.float32
) -> wp.float32:
    
    # Scaling correction over x
    x *= N

    # Computing original profile
    num = wp.pow(x, 2.)
    den = R * (1. + wp.sqrt(1. - (1. + k) * wp.pow(x / R, 2.)))

    coefs = a2 * wp.pow(x, 2.) +  a4 * wp.pow(x, 4.) + a6 * wp.pow(x, 6.) \
          + a8 * wp.pow(x, 8.) + a10 * wp.pow(x, 10.) 

    res = (num / den) + coefs

    # Scaling correction over y
    return (y0 - res) / N

"""
   Analytical computation of the normal 
   using the derivative of the profile.
"""
@wp.func
def asphericLensProfileNormal(
    x  : wp.float32,
    R  : wp.float32, k  : wp.float32,
    y0 : wp.float32, N  : wp.float32,
    a2 : wp.float32, a4 : wp.float32, a6 : wp.float32, a8 : wp.float32, a10: wp.float32    
) -> wp.vec2:

    u = N * x

    derivative  = -       (u) / (R * wp.sqrt(1. - ((1. + k) * wp.pow(u, 2.)) / (R * R)))
    derivative += -        u      *  2. * a2
    derivative += - wp.pow(u, 3.) *  4. * a4
    derivative += - wp.pow(u, 5.) *  6. * a6
    derivative += - wp.pow(u, 7.) *  8. * a8
    derivative += - wp.pow(u, 9.) * 10. * a10

    return wp.normalize(wp.vec2(- derivative, 1.))


@wp.func
def _intersectRayWithDisk(
    ray      : _Ray3D,
    primitive: _Primitive3D
) -> _Intersection3D:

    center = primitive.v0
    normal = wp.normalize(primitive.v1)
    radius = wp.norm_l2(primitive.v1)

   # Default no hit
    noHit = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = primitive
    )

    denom = wp.dot(ray.direction, normal)

    # Ray parallel to disk
    if denom == 0.0:
        return noHit

    # Ray-plane intersection
    t = wp.dot(center - ray.origin, normal) / denom

    if t < 0.0:
        return noHit

    planeIntersection = ray.origin + t * ray.direction
    distanceToCenter  = wp.norm_l2(planeIntersection - center)

    if distanceToCenter <= radius:

        return _Intersection3D(
            hit       = True,
            hitPoint  = planeIntersection,
            normal    = normal,
            ray       = ray,
            primitive = primitive
        )

    return noHit

@wp.func
def _intersectRayWithAnnulus(
    ray      : _Ray3D,
    primitive: _Primitive3D
) -> _Intersection3D:
    
    origin      = primitive.v0
    normal      = wp.normalize(primitive.v1)
    innerRadius = primitive.f0
    outerRadius = primitive.f1

    # Default no intersection
    noHit = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = primitive
    )

    # Firstly, we need to check the intersection between
    # the ray and the plane where the annulus is defined
    t = wp.dot(origin - ray.origin, normal) / wp.dot(ray.direction, normal)

    if t < 0.:
        # Hit must be after ray origin
        return noHit

    planeIntersection = ray.origin + t * ray.direction
    distanceToCenter  = wp.norm_l2(planeIntersection - origin)

    if ((distanceToCenter >= innerRadius) and
        (distanceToCenter <= outerRadius)):
        
        return _Intersection3D(
            hit       = True,
            hitPoint  = planeIntersection,
            normal    = normal,
            ray       = ray,
            primitive = primitive
        )

    return noHit


@wp.func
def _intersectRayWithCylinder(
    ray      : _Ray3D,
    primitive: _Primitive3D
) -> _Intersection3D:
    
    centerOfBasis = primitive.v0
    normalOfBasis = wp.normalize(primitive.v1)
    height        = wp.norm_l2(primitive.v1)
    radius        = primitive.f0

    # Default no hit
    noHit = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = primitive
    )

    # The equations for an infinite cylinder are
    # ||a /\ (p - b)||² = r²     a: cylinder axis b: cylinder basis
    # 0 <= a * (p - b) <= h      r: radius        h: height:
    #                            p: any point on the cylinder
    # The equation for a ray is:
    # p = o + n t                # o: ray origin  n: ray direction

    b     = centerOfBasis - ray.origin
    cross = wp.cross(ray.direction, normalOfBasis)

    discriminant = wp.dot(cross, cross) * radius * radius - wp.pow(wp.dot(b, cross), 2.)

    if discriminant >= 0.:
        # The ray does intersect the infinite cylinder
        d1 = (wp.dot(cross, wp.cross(b, normalOfBasis)) + wp.sqrt(discriminant)) / (wp.dot(cross, cross)) 
        d2 = (wp.dot(cross, wp.cross(b, normalOfBasis)) - wp.sqrt(discriminant)) / (wp.dot(cross, cross))

        # Both d should be tested before chosing one
        smallestD = wp.float32(wp.inf)

        t1 = wp.dot(normalOfBasis, (ray.direction * d1 - b))
        t2 = wp.dot(normalOfBasis, (ray.direction * d2 - b))

        valid1 = (t1 >= 0.0) and (t1 <= height)
        valid2 = (t2 >= 0.0) and (t2 <= height)

        smallestD = wp.float32(wp.inf)

        if valid1:
            smallestD = d1

        if valid2 and d2 < smallestD:
            smallestD = d2

        if smallestD < wp.float32(wp.inf):
            # One solution has been found
            hitPoint = ray.origin + smallestD * ray.direction

            # Computing the normal
            t = wp.dot(normalOfBasis, (ray.direction * smallestD - b))
            normal = ray.direction * smallestD - normalOfBasis * t - b
            normal = wp.normalize(normal)

            return _Intersection3D(
                hit       = True,
                hitPoint  = hitPoint,
                normal    = normal,
                ray       = ray,
                primitive = primitive
            )

    return noHit


@wp.func
def _intersectRayWithSphericalCap(
    ray      : _Ray3D,
    primitive: _Primitive3D
) -> _Intersection3D:
    
    center   = primitive.v0
    radius   = wp.norm_l2(primitive.v1)
    capAngle = primitive.f0
    poleDirection = wp.normalize(primitive.v1)

    direction = wp.normalize(ray.direction)

    # Default no hit
    noHit = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = primitive
    )
    
    # First let's check the intersection with the sphere itself
    a = 1. # dot(ray.direction, ray.direction), by definition 1
    b = 2. * wp.dot(direction, ray.origin - center)
    c = wp.dot(ray.origin - center, ray.origin - center) - radius * radius

    discriminant = b * b - 4. * a * c

    if discriminant >= 0.:
        t1 = (- b + wp.sqrt(discriminant)) / (2. * a)
        t2 = (- b - wp.sqrt(discriminant)) / (2. * a)

        # ----- test first root -----
        valid1    = False
        hitPoint1 = wp.vec3f(0.0, 0.0, 0.0)

        if t1 >= 0.0:
            hp = ray.origin + t1 * ray.direction
            v  = wp.normalize(hp - center)

            angle = wp.acos(wp.dot(v, poleDirection))

            if angle <= capAngle:
                valid1    = True
                hitPoint1 = hp

        # ----- test second root -----
        valid2    = False
        hitPoint2 = wp.vec3f(0.0, 0.0, 0.0)

        if t2 >= 0.0:
            hp = ray.origin + t2 * ray.direction
            v  = wp.normalize(hp - center)

            angle = wp.acos(wp.dot(v, poleDirection))

            if angle <= capAngle:
                valid2    = True
                hitPoint2 = hp

        # ----- choose closest valid root -----
        t        = wp.float32(wp.inf)
        hitPoint = wp.vec3f(0.0, 0.0, 0.0)

        if valid1:
            t = t1
            hitPoint = hitPoint1

        if valid2 and t2 < t:
            t = t2
            hitPoint = hitPoint2

        if t == wp.float32(wp.inf):
            return noHit

        normal = wp.normalize(hitPoint - center)

        return _Intersection3D(
            hit       = True,
            hitPoint  = hitPoint,
            normal    = normal,
            ray       = ray,
            primitive = primitive
        )

    return noHit


@wp.func
def _intersectRayWithAsphericLens(
    ray      : _Ray3D,
    primitive: _Primitive3D
) -> _Intersection3D:
    
    localAxisOrigin = primitive.v0
    zAxisUnitVector = primitive.v1

    # Length unit
    u = wp.norm_l2(primitive.v1)

    # Default no hit
    noHit = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = primitive
    )

    # Before checking intersection, let's check
    # the bounding box that can be defined with
    # two disks and a cylinder.
    y0 = primitive.f4 / primitive.f5

    lowerBound = _Primitive3D(
        type = 6,  # Disk
        v0   = localAxisOrigin,
        v1   = zAxisUnitVector,
        v2   = wp.vec3f(0.0),
        v3   = wp.vec3f(0.0),
        f0   = wp.float32(0.0),
        f1   = wp.float32(0.0),
        f2   = wp.float32(0.0),
        f3   = wp.float32(0.0),
        f4   = wp.float32(0.0),
        f5   = wp.float32(0.0),
        f6   = wp.float32(0.0),
        f7   = wp.float32(0.0),
        f8   = wp.float32(0.0),
        f9   = wp.float32(0.0),
        f10  = wp.float32(0.0)
    )

    midstBound = _Primitive3D(
        type = 1,  # Cylinder
        v0   = localAxisOrigin,
        v1   = zAxisUnitVector * y0,
        v2   = wp.vec3f(0.0),
        v3   = wp.vec3f(0.0),
        f0   = wp.float32(u),
        f1   = wp.float32(0.0),
        f2   = wp.float32(0.0),
        f3   = wp.float32(0.0),
        f4   = wp.float32(0.0),
        f5   = wp.float32(0.0),
        f6   = wp.float32(0.0),
        f7   = wp.float32(0.0),
        f8   = wp.float32(0.0),
        f9   = wp.float32(0.0),
        f10  = wp.float32(0.0)
    )

    upperBound = _Primitive3D(
        type = 6,  # Disk
        v0   = localAxisOrigin + zAxisUnitVector * y0,
        v1   = zAxisUnitVector,
        v2   = wp.vec3f(0.0),
        v3   = wp.vec3f(0.0),
        f0   = wp.float32(0.0),
        f1   = wp.float32(0.0),
        f2   = wp.float32(0.0),
        f3   = wp.float32(0.0),
        f4   = wp.float32(0.0),
        f5   = wp.float32(0.0),
        f6   = wp.float32(0.0),
        f7   = wp.float32(0.0),
        f8   = wp.float32(0.0),
        f9   = wp.float32(0.0),
        f10  = wp.float32(0.0)
    )

    # Testing bounding box intersection
    closestPoint     = wp.vec3(wp.inf, wp.inf, wp.inf)
    closestDistance  = wp.float32(wp.inf)
    farthestPoint    = wp.vec3(wp.inf, wp.inf, wp.inf)
    farthestDistance = wp.float32(wp.inf)

    lowerIntersection = _intersectRayWithDisk(ray, lowerBound)
    if lowerIntersection.hit:
        distance = wp.norm_l2(lowerIntersection.hitPoint - ray.origin)
        if distance < closestDistance:
            farthestPoint    = closestPoint
            farthestDistance = closestDistance
            closestPoint     = lowerIntersection.hitPoint
            closestDistance  = distance
        elif distance < farthestDistance:
            farthestPoint    = lowerIntersection.hitPoint
            farthestDistance = distance

    midstIntersection = _intersectRayWithCylinder(ray, midstBound)
    if midstIntersection.hit:
        distance = wp.norm_l2(midstIntersection.hitPoint - ray.origin)
        if distance < closestDistance:
            farthestPoint    = closestPoint
            farthestDistance = closestDistance
            closestPoint     = midstIntersection.hitPoint
            closestDistance  = distance
        elif distance < farthestDistance:
            farthestPoint    = midstIntersection.hitPoint
            farthestDistance = distance

    upperIntersection = _intersectRayWithDisk(ray, upperBound)
    if upperIntersection.hit:
        distance = wp.norm_l2(upperIntersection.hitPoint - ray.origin)
        if distance < closestDistance:
            farthestPoint    = closestPoint
            farthestDistance = closestDistance
            closestPoint     = upperIntersection.hitPoint
            closestDistance  = distance
        elif distance < farthestDistance:
            farthestPoint    = upperIntersection.hitPoint
            farthestDistance = distance

    # Ray did not intersect lens
    if closestDistance == wp.float32(wp.inf):
        return noHit

    # Only one intersection occurred with the bounding box:
    # origin or ray is already inside.
    if farthestDistance == wp.float32(wp.inf):
        farthestPoint = closestPoint
        closestPoint  = ray.origin

    # Intersection will be studied using the bisection method
    segmentOrigin = closestPoint
    segmentVector = farthestPoint - closestPoint

    tMin  = wp.float32(0.0)
    tMax  = wp.float32(1.0)
    mean  = wp.float32(0.5)
    rMean = wp.float32(0.0)

    # Last test before bisection: maybe only the bounding box is hit
    pMin = segmentVector * tMin + segmentOrigin
    pMax = segmentVector * tMax + segmentOrigin

    yMin = wp.dot(pMin - localAxisOrigin, zAxisUnitVector) / (u * u)
    yMax = wp.dot(pMax - localAxisOrigin, zAxisUnitVector) / (u * u)

    rMinTest = wp.norm_l2(wp.cross(pMin - localAxisOrigin, zAxisUnitVector)) / (u * u)
    rMaxTest = wp.norm_l2(wp.cross(pMax - localAxisOrigin, zAxisUnitVector)) / (u * u)

    fMin = asphericLensProfile(
        rMinTest,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    ) - yMin

    fMax = asphericLensProfile(
        rMaxTest,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    ) - yMax

    if fMin * fMax > 0.0:
        return noHit

    maxIter = 64
    """ # Not a good way to handle differentiable rendering
    epsilon = 1.0e-10
    it      = wp.int32(0)

    while (tMax - tMin) > epsilon and it < maxIter:
        mean = (tMin + tMax) / 2.0

        linearRegtMin = segmentVector * tMin + segmentOrigin
        linearRegMean = segmentVector * mean + segmentOrigin

        linearRegtMinY = wp.dot(linearRegtMin - localAxisOrigin, zAxisUnitVector) / (u * u)
        linearRegMeanY = wp.dot(linearRegMean - localAxisOrigin, zAxisUnitVector) / (u * u)

        rMin = wp.norm_l2(wp.cross(linearRegtMin - localAxisOrigin, zAxisUnitVector)) / (u * u)
        rMean = wp.norm_l2(wp.cross(linearRegMean - localAxisOrigin, zAxisUnitVector)) / (u * u)

        asphericProfiletMin = asphericLensProfile(
            rMin,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )
        asphericProfileMean = asphericLensProfile(
            rMean,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )

        difftMin = asphericProfiletMin - linearRegtMinY
        diffMean = asphericProfileMean - linearRegMeanY

        if difftMin * diffMean <= 0.0:
            tMax = mean
        else:
            tMin = mean

        it += 1
    """
    for it in range(maxIter):
        mean = (tMin + tMax) / 2.0

        linearRegtMin = segmentVector * tMin + segmentOrigin
        linearRegMean = segmentVector * mean + segmentOrigin

        linearRegtMinY = wp.dot(linearRegtMin - localAxisOrigin, zAxisUnitVector) / (u * u)
        linearRegMeanY = wp.dot(linearRegMean - localAxisOrigin, zAxisUnitVector) / (u * u)

        rMin = wp.norm_l2(wp.cross(linearRegtMin - localAxisOrigin, zAxisUnitVector)) / (u * u)
        rMean = wp.norm_l2(wp.cross(linearRegMean - localAxisOrigin, zAxisUnitVector)) / (u * u)

        asphericProfiletMin = asphericLensProfile(
            rMin,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )
        asphericProfileMean = asphericLensProfile(
            rMean,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )

        difftMin = asphericProfiletMin - linearRegtMinY
        diffMean = asphericProfileMean - linearRegMeanY

        if difftMin * diffMean <= 0.0:
            tMax = mean
        else:
            tMin = mean

    hitPoint = segmentVector * mean + segmentOrigin

    rMean = wp.norm_l2(wp.cross(hitPoint - localAxisOrigin, zAxisUnitVector)) / (u * u)

    normal2D = asphericLensProfileNormal(
        rMean,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )

    zAxisVector = wp.normalize(zAxisUnitVector)
    radialVector = hitPoint - localAxisOrigin
    radialNorm = wp.norm_l2(radialVector)

    # Cas axial : la normale 3D est simplement l'axe local
    if radialNorm == 0.0:
        normal3D = zAxisVector
    else:
        tAxisVector = wp.normalize(wp.cross(zAxisVector, wp.normalize(radialVector)))
        hAxisVector = wp.normalize(wp.cross(tAxisVector, zAxisVector))
        normal3D    = wp.normalize(normal2D.x * hAxisVector + normal2D.y * zAxisVector)

    return _Intersection3D(
        hit       = True,
        hitPoint  = hitPoint,
        normal    = normal3D,
        ray       = ray,
        primitive = primitive
    )


@wp.kernel
def _intersectRays(
    raysBuffer      : wp.array(dtype=_Ray3D, ndim=1),
    primitivesBuffer: wp.array(dtype=_Primitive3D, ndim=1),
    nbPrimitives    : wp.int32,

    intersectionsBuffer: wp.array(dtype=_Intersection3D, ndim=1)
):
    ID = wp.tid()

    ray = raysBuffer[ID]

    # Initializing intersection to missed
    intersection = _Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
        ray       = ray,
        primitive = _Primitive3D()
    )

    # If several intersections are found,
    # the one we keep will be the closest
    minimumDistance = wp.float32(wp.inf)

    # Every primitive will be checked
    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        # Initialize current intersection to missed
        tempIntersection = _Intersection3D(
            hit       = False,
            hitPoint  = wp.vec3f(wp.inf, wp.inf, wp.inf),
            normal    = wp.vec3f(0.0   , 0.0   , 0.0   ),
            ray       = ray,
            primitive = _Primitive3D()
        ) 

        if   primitive.type == 0: # ---- Annulus blocker -----
            tempIntersection = _intersectRayWithAnnulus(ray, primitive)
        elif primitive.type == 1: # ---- Cylinder ------------
            tempIntersection = _intersectRayWithCylinder(ray, primitive)
        elif primitive.type == 2: # ---- Spherical cap -------
            tempIntersection = _intersectRayWithSphericalCap(ray, primitive)
        elif primitive.type == 3: # ---- Aspheric lens -------
            tempIntersection = _intersectRayWithAsphericLens(ray, primitive)
        elif primitive.type == 4: # ---- Cylinder blocker ----
            tempIntersection = _intersectRayWithCylinder(ray, primitive)
        else:
            # In the case where the primitive type number is
            # unknown, the primitive will simply be ignored.
            continue

        # Saving intersection if closer
        if tempIntersection.hit == True:
            tempDistance = wp.norm_l2(tempIntersection.hitPoint - ray.origin)
            
            isCloser = tempDistance < minimumDistance

            minimumDistance = wp.where(isCloser, tempDistance, minimumDistance)
            intersection    = wp.where(isCloser, tempIntersection, intersection)

    # Saving final intersection
    intersectionsBuffer[ID] = intersection