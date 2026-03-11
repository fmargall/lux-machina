import warp as wp

from structures import Intersection  , Primitive  , Ray  , \
                       Intersection3D, Primitive3D, Ray3D

@wp.func
def cross(a: wp.vec2, b: wp.vec2) -> wp.float32:
    return a.x * b.y - a.y * b.x

@wp.func
def intersectRayWithSegment(
    ray      : Ray, 
    primitive: Primitive
) -> Intersection:
    r = ray.direction
    s = primitive.v1 - primitive.v0
    q = primitive.v0 - ray.origin

    denominator = cross(r, s)

    t = cross(q, s) / denominator
    u = cross(q, r) / denominator

    intersection = Intersection()
    if denominator != 0. and t >= 0. and u >= 0. and u <= 1.:
        ray.isAlive = True

        intersection.hit       = True
        intersection.hitPoint  = ray.origin + t * ray.direction
        intersection.normal    = wp.vec2(-s.y, s.x) / wp.norm_l2(s)
        intersection.ray       = ray
        intersection.primitive = primitive
    else:
        intersection.hit = False

    return intersection

@wp.func
def intersectRayWithBlocker(
    ray      : Ray, 
    primitive: Primitive
) -> Intersection:
    # Blocker is nothing else but a segment blocking light
    intersection = intersectRayWithSegment(ray, primitive)
    intersection.ray.isAlive = False
    return intersection

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
def intersectRayWithAsphericLens(
    ray: Ray,
    primitive: Primitive
) -> Intersection:
    
    # Obtaining the local axis: localAxisOrigin corresponds
    # to the (0 ; 0) point in the lens profile referential,
    # xUnit corresponds to (1, 0) vector, then yUnit stands
    # for the (0, 1) vector.
    localAxisOrigin = (primitive.v1 + primitive.v0) / 2.
    # v0, in the lens profile referential, is
    # (-1 ; 0) and v1 corresponds to (1 ; 0).
    xUnit = (primitive.v1 - primitive.v0) / 2.
    yUnit = wp.vec2(-xUnit.y, xUnit.x)

    yIntercept = primitive.f4
    normalizationFactor = primitive.f5
    
    # Obtaining the bounding box
    lowerSegment = Primitive()
    lowerSegment.type = 1
    lowerSegment.v0   = primitive.v0
    lowerSegment.v1   = primitive.v1

    upperSegment = Primitive()
    upperSegment.type = 1
    upperSegment.v0   = primitive.v0 + yUnit * yIntercept / normalizationFactor
    upperSegment.v1   = primitive.v1 + yUnit * yIntercept / normalizationFactor

    leftSegment = Primitive()
    leftSegment.type = 1
    leftSegment.v0   = primitive.v0
    leftSegment.v1   = primitive.v0 + yUnit * yIntercept / normalizationFactor

    rightSegment = Primitive()
    rightSegment.type = 1
    rightSegment.v0   = primitive.v1
    rightSegment.v1   = primitive.v1 + yUnit * yIntercept / normalizationFactor
    
    # Testing bounding box intersection
    closestPoint     = wp.vec2(wp.inf, wp.inf)
    closestDistance  = wp.float32(wp.inf)
    farthestPoint    = wp.vec2(wp.inf, wp.inf)
    farthestDistance = wp.float32(wp.inf) 
    
    lowerIntersection = intersectRayWithSegment(ray, lowerSegment)
    
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

    upperIntersection = intersectRayWithSegment(ray, upperSegment)
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

    leftIntersection  = intersectRayWithSegment(ray, leftSegment)
    if leftIntersection.hit:
        distance = wp.norm_l2(leftIntersection.hitPoint - ray.origin)
        if distance < closestDistance:
            farthestPoint    = closestPoint
            farthestDistance = closestDistance
            closestPoint     = leftIntersection.hitPoint
            closestDistance  = distance
        elif distance < farthestDistance:
            farthestPoint    = leftIntersection.hitPoint
            farthestDistance = distance

    rightIntersection = intersectRayWithSegment(ray, rightSegment)
    if rightIntersection.hit:
        distance = wp.norm_l2(rightIntersection.hitPoint - ray.origin)
        if distance < closestDistance:
            farthestPoint    = closestPoint
            farthestDistance = closestDistance
            closestPoint     = rightIntersection.hitPoint
            closestDistance  = distance
        elif distance < farthestDistance:
            farthestPoint    = rightIntersection.hitPoint
            farthestDistance = distance

    # Ray did not intersect lens
    if closestDistance == wp.inf:
        intersection = Intersection()
        intersection.hit = False

        return intersection

    # Only one intersection occured
    # with the bounding box: origin
    # or ray is already inside.
    elif farthestDistance == wp.inf:
        farthestPoint = closestPoint
        closestPoint  = ray.origin

    # We can now project these two points on the x-axis
    # of the aspheric lens profile, going from -1 to 1.
    # This means solving the following 2D system:
    # x = x0 + xProj xX + yProj yX
    # y = y0 + xProj xY + yProj yY
    # We can then isolate xProj which is our unknown, either with
    # one of the following method : choosing the one with highest
    # denominator allows to avoid some numerical errors
    # xProj = (x - x0 - (yX / yY) (y - y0)) / (xX - (yX / yY) xY)
    #         (xDelta)  (yDivide) (yDelta)    (   denominator   )
    # xProj = (y - y0 - (yY / yX) (x - x0)) / (xY - (yY / yX) xX)
    #         (yDelta)  (yDivide) (xDelta)    (   denominator   )
    # Following the same method, we will also need to
    # have the projection of these two points for the
    # linear regression of the ray.

    if wp.abs(yUnit.x) < wp.abs(yUnit.y):
        yDivide     = yUnit.x / yUnit.y
        denominator = xUnit.x - yDivide * xUnit.y
    else:
        yDivide     = yUnit.y / yUnit.x
        denominator = xUnit.y - yDivide * xUnit.x

    xDelta = closestPoint.x - localAxisOrigin.x
    yDelta = closestPoint.y - localAxisOrigin.y
    if wp.abs(yUnit.x) < wp.abs(yUnit.y):
        xClosest = (xDelta - yDivide  * yDelta ) / denominator
        yClosest = (yDelta - xClosest * xUnit.y) / yUnit.y
    else:
        xClosest = (yDelta - yDivide  * xDelta ) / denominator
        yClosest = (xDelta - xClosest * xUnit.x) / yUnit.x

    xDelta = farthestPoint.x - localAxisOrigin.x
    yDelta = farthestPoint.y - localAxisOrigin.y
    if wp.abs(yUnit.x) < wp.abs(yUnit.y):
        xFarthest = (xDelta - yDivide   * yDelta ) / denominator
        yFarthest = (yDelta - xFarthest * xUnit.y) / yUnit.y
    else:
        xFarthest = (yDelta - yDivide   * xDelta ) / denominator
        yFarthest = (xDelta - xFarthest * xUnit.x) / yUnit.x

    xMin = wp.min(xClosest, xFarthest)
    xMax = wp.max(xClosest, xFarthest)
    if xMin == xClosest:
        yMin = yClosest
        yMax = yFarthest
    else:
        yMin = yFarthest
        yMax = yClosest

    # Ray can be then represented by linear regression
    m = (yMax - yMin) / (xMax - xMin)
    p =  yMax - m * xMax

    # Before running the bisection methode, we need to
    # be sure that there is indeed an intersection, if
    # not the algorithm will return a fake hit.
    rayAtMin = m * xMin + p
    rayAtMax = m * xMax + p

    profileAtMin = asphericLensProfile(xMin,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )
    profileAtMax = asphericLensProfile(xMax,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )

    # If there is no intersection, returning missed hit
    diffMin = profileAtMin - rayAtMin
    diffMax = profileAtMax - rayAtMax
    if diffMin * diffMax > 0.0:
        intersection = Intersection()
        intersection.hit = False
        return intersection
    
    # Mean value should be initialized first. If not,
    # a strange behaviour may appear after while loop
    # maybe due to obscure Warp optimization when the
    # kernel compilation is made.
    mean = (xMin + xMax) / 2.

    # Intersection can be found using bisection method
    epsilon = 1.e-10; maxIter = 64; it = wp.int32(0.)
    while (xMax - xMin) > epsilon and it < maxIter:
        mean = (xMin + xMax) / 2.

        # Let's compute the ray linear regression
        linearRegxMin = m * xMin + p
        linearRegMean = m * mean + p

        # Let's compute the aspheric lens profile
        asphericProfilexMin = asphericLensProfile(xMin,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )
        asphericProfileMean = asphericLensProfile(mean,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )

        # Let's compute the difference between the ray
        # linear regression and aspheric lens profile.
        diffxMin = asphericProfilexMin - linearRegxMin
        diffMean = asphericProfileMean - linearRegMean

        if (diffxMin * diffMean <= 0.):
            xMax = mean
        else:
            xMin = mean

        it += 1

    # Ensure final profile value is computed from the final, well-defined mean
    # The explanation for this can be found in the comment above associated to
    # the initilization of the 'mean' variable before the loop
    asphericProfileMean = asphericLensProfile(mean,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )
    
    # DEPRECATED: This version was using numerical computation
    #             for the normal direction. This may be reused
    #             in the future for other primitives.
    """
    # We now have the intersection of the ray and the profile
    # We can compute its tangent, then its associated normal.
    if wp.abs(mean) < epsilon:
        # Caution : close-to-zero value on x-axis profile
        # may cause rounding errors and should be treated
        asphericProfileMean = primitive.f4 / primitive.f5
        normal              = wp.vec2(0., 1.)
    else:
        if mean > xMin:
            tangent = wp.normalize(wp.vec2(mean - xMin, asphericProfileMean - asphericProfilexMin))
        else:
            tangent = wp.normalize(wp.vec2(xMin - mean, asphericProfilexMin - asphericProfileMean))
        normal = wp.vec2(-tangent.y, tangent.x)
    """
    
    # Normal vector is computed analytically
    normal = asphericLensProfileNormal(mean,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )
    
    # We are still in the referential of the profile, we needq
    # to go back to the original referential before returning
    hitPoint = localAxisOrigin + mean * xUnit + asphericProfileMean * yUnit
    normal   = wp.normalize(normal.x * xUnit + normal.y * yUnit)

    ray.isAlive = True

    intersection = Intersection()
    intersection.hit = True
    intersection.hitPoint  = hitPoint
    intersection.normal    = normal
    intersection.ray       = ray
    intersection.primitive = primitive

    return intersection

"""
   Since everything is always computed in screen coordinates,
   such as x in [0 ; 1] and y in [0 ; 1], the unique bounding
   box is always defined as four segments.
"""
@wp.func
def intersectRayWithBoundingBox(
    ray      : Ray,
    primitive: Primitive
) -> Intersection:

    aspectRatio = primitive.f0

    # Test for the uppper border: (0 ; 1) - (1 ; 1)
    # or for an aspectRatio != 1: (0 ; 1) - (aspectRatio ; 1)
    upperBorder = Primitive()
    upperBorder.type = 1
    upperBorder.v0 = wp.vec2(0.         , 1.)
    upperBorder.v1 = wp.vec2(aspectRatio, 1.)
    intersection = intersectRayWithSegment(ray, upperBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    # Test for the right  border: (1 ; 0) - (1 ; 1)
    # or for an aspectRatio != 1: (aspectRatio ; 0) - (aspectRatio ; 1)
    rightBorder = Primitive()
    rightBorder.type = 1
    rightBorder.v0 = wp.vec2(aspectRatio, 0.)
    rightBorder.v1 = wp.vec2(aspectRatio, 1.)
    intersection = intersectRayWithSegment(ray, rightBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    # Test for the lower  border: (0 ; 0) - (1 ; 0)
    # or for an aspectRatio != 1: (0 ; 0) - (aspectRatio ; 0)
    lowerBorder = Primitive()
    lowerBorder.type = 1
    lowerBorder.v0 = wp.vec2(0.         , 0.)
    lowerBorder.v1 = wp.vec2(aspectRatio, 0.)
    intersection = intersectRayWithSegment(ray, lowerBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    # Test for the left   border: (0 ; 0) - (0 ; 1)    
    leftBorder = Primitive()
    leftBorder.type = 1
    leftBorder.v0 = wp.vec2(0., 0.)
    leftBorder.v1 = wp.vec2(0., 1.) 
    intersection = intersectRayWithSegment(ray, leftBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    noIntersection = Intersection()
    noIntersection.hit = False
    return noIntersection

@wp.func
def intersectRayWithCircularArc(
    ray      : Ray,
    primitive: Primitive
) -> Intersection:
    
    center = primitive.v0
    radius = primitive.f0
    theta1 = primitive.f1
    theta2 = primitive.f2

    #    The ray is defined as  r(t) = ray.origin + t * ray.direction
    # The circle is defined as (r(t) - center)² = R²
    # Substituting first equation into second gives:
    # (ray.origin + t * ray.direction - center)² - R² = 0

    # Expanding this equation gives us:
    #     (ray.direction * t)²
    # + 2 (ray.direction (ray.origin - center)) t
    # +   (ray.origin - center)² - R² = 0

    # We then obtain an equation of the form
    # a * t² + b * t + c = 0

    # Computing the discrimnant
    a =      wp.dot(ray.direction      , ray.direction)
    b = 2. * wp.dot(ray.direction      , ray.origin - center)
    c =      wp.dot(ray.origin - center, ray.origin - center) - radius * radius

    discriminant = b * b - 4. * a * c

    intersection = Intersection()
    intersection.hit = False
    if discriminant >= 0.:
        # Main circle is touched twice by input ray
        t1 = (- b - wp.sqrt(discriminant)) / (2. * a)
        t2 = (- b + wp.sqrt(discriminant)) / (2. * a)

        # Both t should be tested before chosing one
        smallestT = wp.float32(wp.inf)

        ts = wp.vec2(t1, t2)
        for tID in range(2):
            # Both solutions should be tried and compared
            t = ts[tID] 

            # t cannot be negative : the intersection must
            # be detected after the ray origin, not before
            if t < 0.: continue

            hitPoint = ray.origin + t * ray.direction

            # Let's move the hitPoint to the trigonometric
            # circle referential, in order to check if the
            # angle is in the circular arc.
            hitPointMC = hitPoint - center

            phi = wp.atan2(hitPointMC.y, hitPointMC.x)
            if (phi < 0): phi += 6.2831853071795864769

            hitCircularArc = False
            if (theta1 <= theta2):
                hitCircularArc = (phi >= theta1 and phi <= theta2)
            else:
                hitCircularArc = (phi >= theta1 or  phi <= theta2)

            if (hitCircularArc and t < smallestT): smallestT = t

        if smallestT < wp.float32(wp.inf):
            # One solution has been found
            ray.isAlive = True

            intersection.hit = True
            intersection.hitPoint = ray.origin + smallestT * ray.direction
            intersection.normal = wp.normalize(intersection.hitPoint - center)

            intersection.ray       = ray
            intersection.primitive = primitive

    return intersection

@wp.func
def intersect3DRayWithParallelogram(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:

    v0 = primitive.v0 # Connected to 1 and 3
    v1 = primitive.v1 # Connected to 0 and 2
    v2 = primitive.v2 # Connected to 1 and 3
    v3 = primitive.v3 # Connected to 0 and 2

    u = v1 - v0
    v = v3 - v0
    n = wp.normalize(wp.cross(u, v))

    intersection = Intersection3D()
    intersection.hit = False

    # First let's check where, and 
    # if, the ray intersects plane
    t = wp.dot((v0 - ray.origin), n) / wp.dot(ray.direction, n)

    # Intersection cannot be before ray origin
    if t < wp.float32(0.):
        return Intersection3D(
            hit       = False,
            hitPoint  = wp.vec3f(0., 0., 0.),
            normal    = wp.vec3f(0., 0., 0.),
            ray       = ray,
            primitive = primitive
        )

    # Intersection point is inside the parallelogram if it verifies:
    # intersection = alpha * u + beta * v | (alpha, beta) in [0, 1]²
    hitPoint = ray.origin + t * ray.direction

    w = hitPoint - v0

    A = wp.dot(u, u) # a11
    B = wp.dot(u, v) # a12 = a21
    D = wp.dot(v, v) # a22

    b1 = wp.dot(w, u)
    b2 = wp.dot(w, v)

    det = A * D - B * B

    # If degenerate (parallelogram collapsed), no intersection
    if det == 0.0:
        return Intersection3D(
            hit       = False,
            hitPoint  = wp.vec3f(0., 0., 0.),
            normal    = wp.vec3f(0., 0., 0.),
            ray       = ray,
            primitive = primitive
        )

    alpha =  (D * b1 - B * b2) / det
    beta  = (-B * b1 + A * b2) / det

    # Intersection point is not on the parallelogram.
    if ((alpha < 0.0 or alpha > 1.0) or
        (beta  < 0.0 or beta  > 1.0)):
        return Intersection3D(
            hit       = False,
            hitPoint  = wp.vec3f(0., 0., 0.),
            normal    = wp.vec3f(0., 0., 0.),
            ray       = ray,
            primitive = primitive
        )

    # Intersection is inside the parallelogram
    return Intersection3D(
        hit       = True,
        hitPoint  = hitPoint,
        normal    = n,
        ray       = ray,
        primitive = primitive
    )

@wp.func
def intersect3DRayWithDisk(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:
    
    center = primitive.v0
    normal = wp.normalize(primitive.v1)
    radius = wp.norm_l2(primitive.v1)

    # Dead ray
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    # Default no hit
    noHit = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
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

        aliveRay = Ray3D(
            isAlive   = True,
            origin    = ray.origin,
            direction = ray.direction,
            depth     = ray.depth,
            energy    = ray.energy
        )

        return Intersection3D(
            hit       = True,
            hitPoint  = planeIntersection,
            normal    = normal,
            ray       = aliveRay,
            primitive = primitive
        )

    return noHit

@wp.func
def intersect3DRayWithAnnulusBlocker(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:

    origin = primitive.v0
    normal = wp.normalize(primitive.v1)
    innerRadius = primitive.f0
    outerRadius = primitive.f1

    # Default dead ray
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    # Default no intersection
    noHit = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
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
        
        # Ray killed by blocker
        killedRay = Ray3D(
            isAlive   = False,
            origin    = ray.origin,
            direction = ray.direction,
            depth     = ray.depth,
            energy    = ray.energy
        )

        return Intersection3D(
            hit       = True,
            hitPoint  = planeIntersection,
            normal    = normal,
            ray       = killedRay,
            primitive = primitive
        )

    return noHit

@wp.func
def intersect3DRayWithCylinder(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:

    centerOfBasis = primitive.v0
    normalOfBasis = wp.normalize(primitive.v1)
    height        = wp.norm_l2(primitive.v1)
    radius        = primitive.f0

    # Default dead ray
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    # Default no hit
    noHit = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
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

        """ # This part may not be differentiable
        ds = wp.vec2(d1, d2)
        for dID in range(2):
            # Both solutions should be tried and compared
            d = ds[dID]

            # Computing the distance from cylinder basis
            t = wp.dot(normalOfBasis, (ray.direction * d - b))

            # t needs to be between 0 and cylinder height
            if t < 0. or t > height: continue

            # If t fits d gives us the right hitPoint
            if d < smallestD: smallestD = d
        """ # Here we propose a corrected version
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

            aliveRay = Ray3D(
                isAlive   = True,
                origin    = ray.origin,
                direction = ray.direction,
                depth     = ray.depth,
                energy    = ray.energy
            )

            return Intersection3D(
                hit       = True,
                hitPoint  = hitPoint,
                normal    = normal,
                ray       = aliveRay,
                primitive = primitive
            )

    return noHit

@wp.func
def intersect3DRayWithSphericalCap(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:

    center   = primitive.v0
    radius   = wp.norm_l2(primitive.v1)
    capAngle = primitive.f0
    poleDirection = wp.normalize(primitive.v1)

    direction = wp.normalize(ray.direction)

    # Default dead ray
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    # Default no hit
    noHit = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
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
        valid1 = False
        hitPoint1 = wp.vec3(0.0,0.0,0.0)

        if t1 >= 0.0:
            hp = ray.origin + t1 * ray.direction
            v  = wp.normalize(hp - center)

            angle = wp.acos(wp.dot(v, poleDirection))

            if angle <= capAngle:
                valid1 = True
                hitPoint1 = hp

        # ----- test second root -----
        valid2 = False
        hitPoint2 = wp.vec3(0.0,0.0,0.0)

        if t2 >= 0.0:
            hp = ray.origin + t2 * ray.direction
            v  = wp.normalize(hp - center)

            angle = wp.acos(wp.dot(v, poleDirection))

            if angle <= capAngle:
                valid2 = True
                hitPoint2 = hp

        # ----- choose closest valid root -----
        t = wp.float32(wp.inf)
        hitPoint = wp.vec3(0.0,0.0,0.0)

        if valid1:
            t = t1
            hitPoint = hitPoint1

        if valid2 and t2 < t:
            t = t2
            hitPoint = hitPoint2

        if t == wp.float32(wp.inf):
            return noHit

        normal = wp.normalize(hitPoint - center)

        aliveRay = Ray3D(
            isAlive   = True,
            origin    = ray.origin,
            direction = ray.direction,
            depth     = ray.depth,
            energy    = ray.energy
        )

        return Intersection3D(
            hit       = True,
            hitPoint  = hitPoint,
            normal    = normal,
            ray       = aliveRay,
            primitive = primitive
        )

    return noHit

""" # Old version: no compatibility with differentiable rendering
@wp.func
def intersect3DRayWithAsphericLens(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:
    
    localAxisOrigin = primitive.v0
    zAxisUnitVector = primitive.v1

    # Length unit
    u = wp.norm_l2(primitive.v1)

    # Before checking intersection, let's check
    # the bounding box that can be defined with
    # two disks and a cylinder.
    
    lowerBound = Primitive3D()
    lowerBound.type = 6 # Disk
    lowerBound.v0 = localAxisOrigin # Center
    lowerBound.v1 = zAxisUnitVector # Normal, with radius as norm
    y0 = primitive.f4 / primitive.f5

    midstBound = Primitive3D()
    midstBound.type = 1 # Cylinder
    midstBound.v0 = localAxisOrigin # Center of basis
    midstBound.v1 = zAxisUnitVector * y0 # Normal of basis, with height as norm
    midstBound.f0 = wp.float32(u) # Radius

    upperBound = Primitive3D()
    upperBound.type = 6 # Disk
    upperBound.v0 = localAxisOrigin + zAxisUnitVector * y0 # Center
    upperBound.v1 = zAxisUnitVector # Normal, with radius as norm

    # Testing bounding box intersection
    closestPoint     = wp.vec3(wp.inf, wp.inf, wp.inf)
    closestDistance  = wp.float32(wp.inf)
    farthestPoint    = wp.vec3(wp.inf, wp.inf, wp.inf)
    farthestDistance = wp.float32(wp.inf)

    lowerIntersection = intersect3DRayWithDisk(ray, lowerBound)
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

    midstIntersection = intersect3DRayWithCylinder(ray, midstBound)
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

    upperIntersection = intersect3DRayWithDisk(ray, upperBound)
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
    if closestDistance == wp.inf:
        intersection = Intersection3D()
        intersection.hit = False

        return intersection

    # Only one intersection occured
    # with the bounding box: origin
    # or ray is already inside.
    elif farthestDistance == wp.inf:
        farthestPoint = closestPoint
        closestPoint  = ray.origin

    # Intersection will be studied using the bisection method
    # Since we know the limits of the ray inside the bounding
    # box, we can move along the ray and check if we're above
    # or below the aspheric lens profile.

    segmentOrigin = closestPoint
    segmentVector = farthestPoint - closestPoint

    # Mean value should be initialized first. If not,
    # a strange behaviour may appear after while loop
    # maybe due to obscure Warp optimization when the
    # kernel compilation is made.
    tMin  = wp.float32(0.0)
    tMax  = wp.float32(1.0)
    mean  = wp.float32(0.5)
    rMin  = wp.float32(0.0)
    rMean = wp.float32(0.0)

    # Last test before bisection method should be to check if
    # only the bounding box is hit, but not the aspheric lens
    pMin = segmentVector * tMin + segmentOrigin
    pMax = segmentVector * tMax + segmentOrigin

    yMin = wp.dot(pMin - localAxisOrigin, zAxisUnitVector) / (u * u)
    yMax = wp.dot(pMax - localAxisOrigin, zAxisUnitVector) / (u * u)

    rMinTest = wp.norm_l2(wp.cross(pMin - localAxisOrigin, zAxisUnitVector)) / (u * u)
    rMaxTest = wp.norm_l2(wp.cross(pMax - localAxisOrigin, zAxisUnitVector)) / (u * u)

    fMin = asphericLensProfile(rMinTest,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    ) - yMin

    fMax = asphericLensProfile(rMaxTest,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    ) - yMax

    if fMin * fMax > 0.:
        intersection = Intersection3D()
        intersection.hit = False
        return intersection

    epsilon = 1.e-10; maxIter = 64; it = wp.int32(0.)
    while (tMax - tMin) > epsilon and it < maxIter:
        mean = (tMin + tMax) / 2.

        # Let's compute the ray linear regression
        linearRegtMin = segmentVector * tMin + segmentOrigin
        linearRegMean = segmentVector * mean + segmentOrigin

        # By definition of the local axis system, should be between 0 and y0 / N
        linearRegtMinY = wp.dot(linearRegtMin - localAxisOrigin, zAxisUnitVector) / (u * u)
        linearRegMeanY = wp.dot(linearRegMean - localAxisOrigin, zAxisUnitVector) / (u * u)

        # To compute the aspheric lens profile, we need
        # to project these two points on the local axis
        # system of the aspheric lens.

        # Point-to-axis distance (ie. [0..1], this is why we need another u at the denominator)
        rMin  = wp.norm_l2(wp.cross(linearRegtMin - localAxisOrigin, zAxisUnitVector)) / (u * u)
        rMean = wp.norm_l2(wp.cross(linearRegMean - localAxisOrigin, zAxisUnitVector)) / (u * u)

        # Let's compute the aspheric lens profile
        asphericProfiletMin = asphericLensProfile(rMin,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )
        asphericProfileMean = asphericLensProfile(rMean,
            primitive.f0, primitive.f1, primitive.f4, primitive.f5,
            primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
        )

        # Let's compute the difference between the ray
        # linear regression and aspheric lens profile.
        difftMin = asphericProfiletMin - linearRegtMinY
        diffMean = asphericProfileMean - linearRegMeanY

        if (difftMin * diffMean <= 0.):
            tMax = mean
        else:
            tMin = mean

        it += 1

    # Ensure final profile value is computed from the final, well-defined mean
    # The explanation for this can be found in the comment above associated to
    # the initilization of the 'mean' variable before the loop
    hitPoint = segmentVector * mean + segmentOrigin

    # Point-to-axis distance (ie. [0..1], thus why we need another u at the denominator)
    rMean = wp.norm_l2(wp.cross(hitPoint - localAxisOrigin, zAxisUnitVector)) / (u * u)
    asphericProfileMean = asphericLensProfile(rMean,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )

    # 2D normal vector is computed analytically
    normal2D = asphericLensProfileNormal(rMean,
        primitive.f0, primitive.f1, primitive.f4, primitive.f5,
        primitive.f6, primitive.f7, primitive.f8, primitive.f9, primitive.f10
    )

    zAxisVector = wp.normalize(zAxisUnitVector)
    tAxisVector = wp.normalize(wp.cross(zAxisVector, wp.normalize(hitPoint - localAxisOrigin)))
    hAxisVector = wp.normalize(wp.cross(tAxisVector, zAxisVector))
    normal3D = wp.normalize(normal2D.x * hAxisVector + normal2D.y * zAxisVector)

    ray.isAlive = True

    intersection = Intersection3D()
    intersection.hit = True
    intersection.hitPoint  = hitPoint
    intersection.normal    = normal3D
    intersection.ray       = ray
    intersection.primitive = primitive

    return intersection
"""
@wp.func
def intersect3DRayWithAsphericLens(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:
    
    localAxisOrigin = primitive.v0
    zAxisUnitVector = primitive.v1

    # Length unit
    u = wp.norm_l2(primitive.v1)

    # Default dead ray
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    # Default no hit
    noHit = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
        primitive = primitive
    )

    # Before checking intersection, let's check
    # the bounding box that can be defined with
    # two disks and a cylinder.
    y0 = primitive.f4 / primitive.f5

    lowerBound = Primitive3D(
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

    midstBound = Primitive3D(
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

    upperBound = Primitive3D(
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

    lowerIntersection = intersect3DRayWithDisk(ray, lowerBound)
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

    midstIntersection = intersect3DRayWithCylinder(ray, midstBound)
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

    upperIntersection = intersect3DRayWithDisk(ray, upperBound)
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
        normal3D = wp.normalize(normal2D.x * hAxisVector + normal2D.y * zAxisVector)

    aliveRay = Ray3D(
        isAlive   = True,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )

    return Intersection3D(
        hit       = True,
        hitPoint  = hitPoint,
        normal    = normal3D,
        ray       = aliveRay,
        primitive = primitive
    )

@wp.func
def intersect3DRayWithCylinderBlocker(
    ray      : Ray3D,
    primitive: Primitive3D
) -> Intersection3D:
    # Blocker is nothing else but cylinder that blocks light.
    intersection = intersect3DRayWithCylinder(ray, primitive)
    if not intersection.hit:
        return intersection

    deadRay = Ray3D(
        isAlive   = False,
        origin    = intersection.ray.origin,
        direction = intersection.ray.direction,
        depth     = intersection.ray.depth,
        energy    = intersection.ray.energy
    )

    return Intersection3D(
        hit       = True,
        hitPoint  = intersection.hitPoint,
        normal    = intersection.normal,
        ray       = deadRay,
        primitive = intersection.primitive
    )

@wp.kernel
def noIntersection3DRays(
    raysBuffer         : wp.array(dtype=Ray3D, ndim=1),
    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1)
):
    ID = wp.tid()

    intersectionsBuffer[ID] = Intersection3D(
        hit = False,
        hitPoint  = wp.vec3(0., 0., 0.),
        normal    = wp.vec3(0., 0., 0.),
        ray       = raysBuffer[ID],
        primitive = Primitive3D()
    )

@wp.kernel
def intersect3DRays(
    raysBuffer      : wp.array(dtype=Ray3D, ndim=1),
    primitivesBuffer: wp.array(dtype=Primitive3D, ndim=1),
    nbPrimitives    : wp.int32,

    intersectionsBuffer: wp.array(dtype=Intersection3D, ndim=1),
    raysStatusBuffer   : wp.array(dtype=wp.bool, ndim=1)
):
    ID = wp.tid()

    ray = raysBuffer[ID]

    # Setting ray as dead by default
    deadRay = Ray3D(
        isAlive   = False,
        origin    = ray.origin,
        direction = ray.direction,
        depth     = ray.depth,
        energy    = ray.energy
    )
    raysStatusBuffer[ID] = False

    # Initializing intersection to missed
    intersection = Intersection3D(
        hit       = False,
        hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
        normal    = wp.vec3(0.0, 0.0, 0.0),
        ray       = deadRay,
        primitive = Primitive3D()
    )

    # If several intersections are found,
    # the one we keep will be the closest
    minimumDistance = wp.float32(wp.inf)

    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        tempIntersection = Intersection3D(
            hit       = False,
            hitPoint  = wp.vec3(wp.inf, wp.inf, wp.inf),
            normal    = wp.vec3(0.0, 0.0, 0.0),
            ray       = deadRay,
            primitive = Primitive3D()
        ) 

        if   primitive.type == 0: # Annulus blocker
            tempIntersection = intersect3DRayWithAnnulusBlocker(ray, primitive)
        elif primitive.type == 1: # Cylinder
            tempIntersection = intersect3DRayWithCylinder(ray, primitive)
        elif primitive.type == 2: # Spherical cap
            tempIntersection = intersect3DRayWithSphericalCap(ray, primitive)
        elif primitive.type == 3: # Aspheric lens
            tempIntersection = intersect3DRayWithAsphericLens(ray, primitive)
        elif primitive.type == 4: # Cylinder blocker
            tempIntersection = intersect3DRayWithCylinderBlocker(ray, primitive)
        else:
            continue

        # Saving intersection if closer
        if tempIntersection.hit == True:
            tempDistance = wp.norm_l2(tempIntersection.hitPoint - ray.origin)
            
            isCloser = tempDistance < minimumDistance

            minimumDistance = wp.where(isCloser, tempDistance, minimumDistance)
            intersection    = wp.where(isCloser, tempIntersection, intersection)
        
    # Saving final intersection
    intersectionsBuffer[ID] = intersection
    
    # Saving ray status
    if intersection.hit == True:
        raysStatusBuffer[ID] = intersection.ray.isAlive