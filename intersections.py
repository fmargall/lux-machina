import warp as wp

from structures import Intersection, Primitive, Ray

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
    
    # Intersection can be found using bisection method
    epsilon = 1.e-10; maxIter = 64; it = wp.int32(0)
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
    
    # We now have the intersection of the ray and the profile
    # We can compute its tangent, then its associated normal.
    if wp.abs(mean) < epsilon:
        # Caution : close-to-zero value on x-axis profile
        # may cause rounding errors and should be treated
        asphericProfileMean = primitive.f4 / primitive.f5
    else:
        if mean > xMin:
            tangent = wp.normalize(wp.vec2(mean - xMin, asphericProfileMean - asphericProfilexMin))
        else:
            tangent = wp.normalize(wp.vec2(xMin - mean, asphericProfilexMin - asphericProfileMean))
        normal = wp.vec2(-tangent.y, tangent.x)

    # We are still in the referential of the profile, we need
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
        smallestT = wp.float32(1.e15)

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

        if smallestT < wp.float32(1.e15):
            # One solution has been found
            ray.isAlive = True

            intersection.hit = True
            intersection.hitPoint = ray.origin + smallestT * ray.direction
            intersection.normal = wp.normalize(intersection.hitPoint - center)

            intersection.ray       = ray
            intersection.primitive = primitive

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

    # Initializing intersection
    intersection = Intersection()
    intersection.hit = False # No intersection
    intersection.hitPoint = wp.vec2(wp.inf, wp.inf) 
    intersectionsBuffer[ID] = intersection

    # If several intersections are found,
    # the one we keep will be the closest
    minimumDistance = wp.float32(wp.inf)
    
    # Setting ray as dead by default
    ray.isAlive          = False
    raysStatusBuffer[ID] = False

    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        tempIntersection = Intersection()
        tempIntersection.hit      = False
        tempIntersection.hitPoint = wp.vec2(wp.inf, wp.inf) 

        if   primitive.type == -1: # Bounding box
            tempIntersection = intersectRayWithBoundingBox(ray, primitive)
        elif primitive.type ==  0: # Ideal lens (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type ==  1: # Straight line interface (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type ==  2: # Circular arc interface
            tempIntersection = intersectRayWithCircularArc(ray, primitive)
        elif primitive.type ==  3: # Aspheric lens interface
            tempIntersection = intersectRayWithAsphericLens(ray, primitive)
        elif primitive.type ==  4: # Blocker
            tempIntersection = intersectRayWithBlocker(ray, primitive)
        else:
            continue

        # Saving intersection if closer
        if tempIntersection.hit == True:
            tempDistance = wp.norm_l2(tempIntersection.hitPoint - ray.origin)
            if tempDistance < minimumDistance:
                minimumDistance = tempDistance
                intersection = tempIntersection
        
    # Saving final intersection
    intersectionsBuffer[ID] = intersection
    
    # Saving ray status
    if intersection.hit == True:
        raysStatusBuffer[ID] = intersection.ray.isAlive