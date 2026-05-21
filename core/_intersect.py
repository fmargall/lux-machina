import warp as wp

from _structures import _Intersection, _Primitive, _Ray

@wp.func
def _isFlat(primitive: _Primitive) -> wp.bool:
    #            Triangle == 0              Quad == 1
    #        Disk/Annulus == 2
    return primitive.type == 0 or primitive.type == 1 \
        or primitive.type == 2

@wp.func
def _intersectTriangle(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    # Not implemented yet

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Fallbacks to no hit
    return noIntersection

@wp.func
def _intersectQuad(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    # Not implemented yet

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Fallbacks to no hit
    return noIntersection

@wp.func
def _intersectDisk(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    center = primitive.v0
    normal = primitive.v1
    innerRadius = primitive.f0
    outerRadius = primitive.f1

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Firstly, we need to check the intersection between
    # the ray and the plane where the annulus is defined
    denominator = wp.dot(ray.direction, normal)
    if wp.abs(denominator) < wp.float32(1.e-8):
        # Reject parallel ray
        return noIntersection

    t = wp.dot(center - ray.origin, normal) / denominator
    if t <= wp.float32(0.):
        # Reject backward ray
        return noIntersection

    # TO-DO: we can optimize this by comparing squared distances
    # rather than computing the square root for the norm. We may
    # test it and benchmark it later in the future.
    # Even pre-computed the radius squared on the CPU before the
    # GPU run would be even better.
    planeIntersection = ray.origin + t * ray.direction
    distanceToCenter  = wp.norm_l2(planeIntersection - center)

    if ((distanceToCenter >= innerRadius) and
        (distanceToCenter <= outerRadius)):
        return _Intersection(
            t           = t,
            normal      = normal,
            primitiveID = primitiveID
        )

    # Fallbacks to no hit
    return noIntersection

@wp.func
def _intersectSphere(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    # For a full sphere: capAngle = 180° (or π   in radians)
    # For an hemisphere: capAngle =  90° (or π/2 in radians)
    center        = primitive.v0
    poleDirection = primitive.v1  # (unit vector)
    radius        = primitive.f0
    capAngle      = primitive.f1

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Solving the following quadratic equation:
    # | origin + t * direction - center |² = R²
    # This gives us the following coefficients:
    oc = ray.origin - center
    b  = 2.0 * wp.dot(ray.direction, oc)
    c  = wp.dot(oc, oc) - radius * radius

    discriminant = b * b - 4.0 * c
    # No real solution, the ray missed
    if discriminant < wp.float32(0.0):
        return noIntersection

    sqrtDisc = wp.sqrt(discriminant)
    t1 = (- b - sqrtDisc) * wp.float32(0.5) # near root
    t2 = (- b + sqrtDisc) * wp.float32(0.5) # far  root

    if t1 > 0.: # Check closest hit in first place
        hitPoint = ray.origin + t1 * ray.direction
        normal   = (hitPoint - center) / radius # normalize
        # Test if the hitpoint is within the cap
        cosAngle = wp.dot(normal, poleDirection)
        if cosAngle >= wp.cos(capAngle):
            return _Intersection(
                t           = t1,
                normal      = normal,
                primitiveID = primitiveID,
            )

    if t2 > 0.: # Check far hit if t1 was rejected
        hitPoint = ray.origin + t2 * ray.direction
        normal   = (hitPoint - center) / radius # normalize
        # Test if the hitpoint is within the cap
        cosAngle = wp.dot(normal, poleDirection)
        if cosAngle >= wp.cos(capAngle):
            return _Intersection(
                t           = t2,
                normal      = normal,
                primitiveID = primitiveID,
            )

    # Fallbacks to no hit
    return noIntersection

@wp.func
def _intersectCylinder(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    baseCenter = primitive.v0
    axis       = primitive.v1 # (unit vector)
    radius     = primitive.f0
    height     = primitive.f1

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Project ray origin on the plane
    # perpendicular to cylinder axis.
    ob      = ray.origin - baseCenter
    obDotA  = wp.dot(ob, axis)
    obPerp  = ob - obDotA * axis

    # Project ray direction on the plane
    dDotA = wp.dot(ray.direction, axis)
    dPerp = ray.direction - dDotA * axis

    # Solve the quadratic equation
    # | obPerp + t * dPerp |² = R²
    a = wp.dot(dPerp, dPerp)
    b = 2.0 * wp.dot(obPerp, dPerp)
    c = wp.dot(obPerp, obPerp) - radius * radius

    # Ray parallel, so no hit
    if a < wp.float32(1.e-8):
        return noIntersection

    discriminant = b * b - 4.0 * a * c
    # No real solution, the ray missed
    if discriminant < wp.float32(0.0):
        return noIntersection

    sqrtDisc = wp.sqrt(discriminant)
    tFactor  = wp.float32(0.5) / a
    t1 = (- b - sqrtDisc) * tFactor # near root
    t2 = (- b + sqrtDisc) * tFactor # far  root

    if t1 > 0.: # Check closest hit in first place
        hitAlongCylinderAxis = obDotA + t1 * dDotA
        if hitAlongCylinderAxis >= 0.0 and hitAlongCylinderAxis <= height:
            hitPoint = ray.origin + t1 * ray.direction
            radial   = hitPoint - baseCenter - hitAlongCylinderAxis * axis
            normal   = radial / radius # normalize
            return _Intersection(
                t           = t1,
                normal      = normal,
                primitiveID = primitiveID,
            )

    if t2 > 0.: # Check far hit if t1 was rejected
        hitAlongCylinderAxis = obDotA + t2 * dDotA
        if hitAlongCylinderAxis >= 0.0 and hitAlongCylinderAxis <= height:
            hitPoint = ray.origin + t2 * ray.direction
            radial   = hitPoint - baseCenter - hitAlongCylinderAxis * axis
            normal   = radial / radius # normalize
            return _Intersection(
                t           = t2,
                normal      = normal,
                primitiveID = primitiveID,
            )

    # Fallbacks to no hit
    return noIntersection

@wp.func
def _intersectAsphere(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    # Not implemented yet

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Fallbacks to no hit
    return noIntersection

@wp.kernel
def _intersect(
    # --- Input buffers ---
    raysBuffer      : wp.array(dtype=_Ray      , ndim=1),
    primitivesBuffer: wp.array(dtype=_Primitive, ndim=1),
    nbPrimitives    : wp.int32,

     # --- Output buffer ---
    intersectionsBuffer: wp.array(dtype=_Intersection, ndim=1)
):
    ID  = wp.tid()
    ray = raysBuffer[ID]

    # Initialize intersection to no-hit
    closestIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # Skipping dead rays
    if not ray.isAlive:
        intersectionsBuffer[ID] = closestIntersection
        return

    # Loop over all primitives
    for primitiveID in range(nbPrimitives):
        primitive = primitivesBuffer[primitiveID]

        # Fast ID exclusion: only valid for flat primitives.
        # For non-flat ones we rely on Wächter-Binder offset
        # that will then be done later during propagation.
        if primitiveID == ray.sourcePrimitiveID and _isFlat(primitive):
            continue

        # Initialize candidate intersection
        candidateIntersection = _Intersection(
            t           = wp.float32(-1.0),
            normal      = wp.vec3f(0.0),
            primitiveID = wp.int32(-1)
        )

        if   primitive.type == 0: # Triangle
            candidateIntersection = _intersectTriangle(ray, primitive, primitiveID)
        elif primitive.type == 1: # Quad
            candidateIntersection = _intersectQuad(ray, primitive, primitiveID)
        elif primitive.type == 2: # Disk (or annulus)
            candidateIntersection = _intersectDisk(ray, primitive, primitiveID)
        elif primitive.type == 4: # Sphere (or spherical cap)
            candidateIntersection = _intersectSphere(ray, primitive, primitiveID)
        elif primitive.type == 5: # Cylinder
            candidateIntersection = _intersectCylinder(ray, primitive, primitiveID)
        elif primitive.type == 6: # Asphere
            candidateIntersection = _intersectAsphere(ray, primitive, primitiveID)
        else:
            continue

        if candidateIntersection.t > wp.float32(0.0):
            # We have a hit! Check if the closest one
            if closestIntersection.t < wp.float32(0.0) or candidateIntersection.t < closestIntersection.t:
                closestIntersection = candidateIntersection

    intersectionsBuffer[ID] = closestIntersection