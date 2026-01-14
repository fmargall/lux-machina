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
def intersectRayWithCircularArc(
    ray      : Ray,
    primitive: Primitive
) -> Intersection:
    
    center = primitive.v0
    radius = primitive.f0
    theta1 = primitive.f1
    theta2 = primitive.f2

    # Let's write and solve the equation
    oc = ray.origin - center

    # Computing the discriminant
    a = wp.dot(ray.direction, ray.direction)
    b = 2. * wp.dot(ray.direction, oc)
    c = wp.dot(oc, oc) - radius * radius

    discriminant = b * b - 4. * a * c

    intersection = Intersection()
    intersection.hit = False
    if discriminant >= 0.:
        t1 = - b - wp.sqrt(discriminant) / (2. * a)
        t2 = - b + wp.sqrt(discriminant) / (2. * a)

        # Let's try both solutions
        smallestT = wp.float32(1.e15)
        solutions = wp.vec2(t1, t2)
        for tID in range(2):
            t = solutions[tID]

            # Cannot be before the ray origin
            if t < 0.: continue

            p   = ray.origin + t * ray.direction
            pc  = p - center 
            phi = wp.atan2(pc.y, pc.x)
            if (phi < 0): phi += 6.28318530717958

            hit = False
            if (theta1 <= theta2):
                hit = (phi >= theta1 and phi <= theta2)
            else:
                hit = (phi >= theta1 or  phi <= theta2)

            # Saving the best configuration
            if (hit and t < smallestT): smallestT = t

        # One solution found
        if smallestT < 1.e15:
            ray.isAlive = True

            intersection.hit = True
            intersection.hitPoint  = ray.origin + smallestT * ray.direction
            intersection.normal    = wp.normalize(intersection.hitPoint - center)
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
    intersection.hit = True # Needs to be set to True to be then rendered by the rasterizer
    intersection.hitPoint = ray.origin + 1.e12 * ray.direction # Initially, no intersection
    intersectionsBuffer[ID] = intersection

    # If several intersections are found,
    # the one we keep will be the closest
    minimumDistance = wp.float32(1.e12)
    
    # Setting ray as dead by default
    ray.isAlive = False
    raysStatusBuffer[ID] = False

    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        tempIntersection = Intersection()
        tempIntersection.hit = False

        if   primitive.type == 0: # Ideal lens (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type == 1: # Straight line interface (ie. segment)
            tempIntersection = intersectRayWithSegment(ray, primitive)
        elif primitive.type == 2: # Circular arc interface
            tempIntersection = intersectRayWithCircularArc(ray, primitive)
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
        raysStatusBuffer[ID] = True