import warp as wp

from structures import Intersection3D, Primitive3D, Ray3D


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