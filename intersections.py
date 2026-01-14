import warp as wp

from structures import Intersection, Primitive, Ray

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
    den = R * wp.sqrt(1. - (1. + k) * wp.pow(x / R, 2.))

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
    intersection = Intersection()
    intersection.hit = False

    # TO BE DONE

    return intersection

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

"""
   Since everything is always computed in screen coordinates,
   such as x in [0 ; 1] and y in [0 ; 1], the unique bounding
   box is always defined as four segments.
"""
@wp.func
def intersectRayWithBoundingBox(
    ray: Ray
) -> Intersection:
    # Test for the uppper border: (0 ; 1) - (1 ; 1)
    upperBorder = Primitive()
    upperBorder.type = 1
    upperBorder.v0 = wp.vec2(0., 1.)
    upperBorder.v1 = wp.vec2(1., 1.)
    intersection = intersectRayWithSegment(ray, upperBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    # Test for the right  border: (1 ; 0) - (1 ; 1)
    rightBorder = Primitive()
    rightBorder.type = 1
    rightBorder.v0 = wp.vec2(1., 0.)
    rightBorder.v1 = wp.vec2(1., 1.)
    intersection = intersectRayWithSegment(ray, rightBorder)
    if intersection.hit:
        ray.isAlive = False # Touching bounding box kills ray
        intersection.ray = ray
        return intersection

    # Test for the lower  border: (0 ; 0) - (1 ; 0)
    lowerBorder = Primitive()
    lowerBorder.type = 1
    lowerBorder.v0 = wp.vec2(0., 0.)
    lowerBorder.v1 = wp.vec2(1., 0.)
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
            tempIntersection = intersectRayWithBoundingBox(ray)
        elif primitive.type ==  0: # Ideal lens (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type ==  1: # Straight line interface (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type ==  2: # Circular arc interface
            tempIntersection = intersectRayWithCircularArc(ray, primitive)
        elif primitive.type ==  3: # Aspheric lens interface
            tempIntersection = intersectRayWithAsphericLens(ray, primitive)
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