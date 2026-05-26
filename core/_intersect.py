import warp as wp

from _structures import _isFlat
from _structures import _Intersection, _Primitive, _Ray


@wp.func
def _sag(primitive: _Primitive, r: wp.float32) -> wp.float32:
    rSquared = r * r
    radicand = 1. - (1. + primitive.f1) * rSquared / (primitive.f0 * primitive.f0)
    radicand = wp.max(0.0, radicand) # Defensive clamp

    rFourth = rSquared * rSquared
    rSixth  = rFourth  * rSquared
    rEighth = rSixth   * rSquared
    rTenth  = rEighth  * rSquared

    sag   = (rSquared / (primitive.f0 * (1. + wp.sqrt(radicand))))
    sag  += primitive.f2 * rFourth
    sag  += primitive.f3 * rSixth
    sag  += primitive.f4 * rEighth
    sag  += primitive.f5 * rTenth

    return sag

@wp.func
def _sagDerivative(primitive: _Primitive, r: wp.float32) -> wp.float32:
    rSquared = r * r

    rThird   = r        * rSquared
    rFifth   = rThird   * rSquared
    rSeventh = rFifth   * rSquared
    rNinth   = rSeventh * rSquared

    radicand = 1. - (1. + primitive.f1) * rSquared / (primitive.f0 * primitive.f0)
    radicand = wp.max(0.0, radicand) # Defensive clamp

    derivative  = r / (primitive.f0 * wp.sqrt(radicand))
    derivative += 4.  * primitive.f2 * rThird
    derivative += 6.  * primitive.f3 * rFifth
    derivative += 8.  * primitive.f4 * rSeventh
    derivative += 10. * primitive.f5 * rNinth

    return derivative

@wp.func
def _sagResidual(ray: _Ray, primitive: _Primitive, t: wp.float32) -> wp.float32:
    # f(t) = z_local(t) - sag(r_local(t))
    scale    = wp.norm_l2(primitive.v1)
    axisUnit = primitive.v1 / scale

    point  = ray.origin + t * ray.direction

    relPos = point - primitive.v0

    zLocal = wp.dot(relPos, axisUnit) / scale
    rVec   = relPos - wp.dot(relPos, axisUnit) * axisUnit
    rLocal = wp.norm_l2(rVec) / scale

    return zLocal - _sag(primitive, rLocal)


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
def _cylinderBoundingInterval(ray: _Ray, primitive: _Primitive) -> wp.vec2f:
    baseCenter = primitive.v0
    axis       = primitive.v1 # (unit vector)
    radius     = primitive.f0
    height     = primitive.f1

    zMin = wp.min(0.0, height)
    zMax = wp.max(0.0, height)

    # Project ray origin on the plane
    # perpendicular to cylinder axis.
    ob      = ray.origin - baseCenter
    obDotA  = wp.dot(ob, axis)
    obPerp  = ob - obDotA * axis

    # Project ray direction on the plane
    dDotA = wp.dot(ray.direction, axis)
    dPerp = ray.direction - dDotA * axis

    a = wp.dot(dPerp, dPerp)

    # ------ Testing axial slab intersection ------
    if wp.abs(dDotA) < 1.0e-8:
        # The ray is parallel to the disks
        if obDotA < zMin or obDotA > zMax:
            return wp.vec2f(-1.0, -1.0)
        tSlabMin = -wp.inf
        tSlabMax =  wp.inf
    else:
        t1 = (zMin - obDotA) / dDotA
        t2 = (zMax - obDotA) / dDotA
        tSlabMin = wp.min(t1, t2)
        tSlabMax = wp.max(t1, t2)

    # --- Testing lateral cylinder intersection ---
    if a < wp.float32(1.0e-8):
        # the ray is parallel to cylinder axis
        radialSquared = wp.dot(obPerp, obPerp)
        if radialSquared > radius * radius:
            return wp.vec2f(-1.0, -1.0)
        tCylMin = -wp.inf
        tCylMax =  wp.inf
    else:
        b = wp.float32(2.0) * wp.dot(dPerp, obPerp)
        c = wp.dot(obPerp, obPerp) - radius * radius

        disc = b * b - 4.0 * a * c
        if disc < wp.float32(0.0):
            # ray miss infinite cylinder
            return wp.vec2f(-1.0, -1.0)

        sqrtDisc = wp.sqrt(disc)
        inv2a    = wp.float32(0.5) / a
        tCylMin  = (-b - sqrtDisc) * inv2a
        tCylMax  = (-b + sqrtDisc) * inv2a

    # Intersecting the two intervals:
    tEnter = wp.max(tSlabMin, tCylMin)
    tExit  = wp.min(tSlabMax, tCylMax)

    if tEnter >= tExit:
        # Intervals don't overlap
        return wp.vec2f(-1., -1.)

    # The entire interval is behind the initial ray origin
    if tExit < wp.float32(0.): return wp.vec2f(-1.0, -1.0)

    # Clamp tEnter to 0 if ray starts inside the cylinder
    if tEnter < wp.float32(0.0): tEnter = wp.float32(0.0)

    return wp.vec2f(tEnter, tExit)

@wp.func
def _intersectAsphere(ray: _Ray, primitive: _Primitive, primitiveID: wp.int32) -> _Intersection:
    localAxisOrigin = primitive.v0
    zAxisUnitVector = primitive.v1 # (unit vector in the local frame)
    R    = primitive.f0 # Radius
    K    = primitive.f1 # Conic constant
    a4   = primitive.f2 # 4th order aspheric coefficient
    a6   = primitive.f3 # 6th order aspheric coefficient
    a8   = primitive.f4 # 8th order aspheric coefficient
    a10  = primitive.f5 # 10th order aspheric coefficient
    rMax = primitive.f6 # Maximum radius of the asphere

    scale = wp.norm_l2(zAxisUnitVector)
    sagittaLocal = _sag(primitive, rMax)
    sagittaWorld = sagittaLocal * scale
    rMaxWorld    = rMax * scale
    zAxisNorm    = zAxisUnitVector / scale

    # By default, no intersection
    noIntersection = _Intersection(
        t           = wp.float32(-1.0),
        normal      = wp.vec3f(0.0),
        primitiveID = wp.int32(-1)
    )

    # --- Bounding box intersection test ---
    # The aspheric lens profile can be quite
    # flat or bulged. The best we can choose
    # is a cylinder and two disks.
    boundingCylinder    = _Primitive(
        type = 4, # Cylinder
        materialID = -2, # Bounding box
        v0 = localAxisOrigin,
        v1 = zAxisNorm,
        v2 = wp.vec3f(0.), # Dummy variable
        v3 = wp.vec3f(0.), # Dummy variable
        f0 = rMaxWorld,
        f1 = sagittaWorld,
        f2 = wp.float32(0.), # Dummy variable
        f3 = wp.float32(0.), # Dummy variable
        f4 = wp.float32(0.), # Dummy variable
        f5 = wp.float32(0.), # Dummy variable
        f6 = wp.float32(0.)  # Dummy variable
    )

    interval = _cylinderBoundingInterval(ray, boundingCylinder)

    if interval[0] < wp.float32(0.):
        # Missed bounding box
        return noIntersection

    tEnter = interval[0]
    tExit  = interval[1]

    fEnter = _sagResidual(ray, primitive, tEnter)
    fExit  = _sagResidual(ray, primitive, tExit)

    if fEnter * fExit > wp.float32(0.):
        return noIntersection

    # ---------- Bisection method ----------
    tMin = tEnter
    tMax = tExit
    fMin = fEnter

    maxIter = 64
    for it in range(maxIter):
        tMid = (tMin + tMax) /  wp.float32(2.0)
        fMid = _sagResidual(ray, primitive, tMid)

        if fMin * fMid <= wp.float32(0.0):
            tMax = tMid
        else:
            tMin = tMid
            fMin = fMid

    tHit     = (tMin + tMax) / wp.float32(2.0)
    hitPoint = ray.origin + tHit * ray.direction

    # Last check to see if we are in
    relPos = hitPoint - primitive.v0
    rVec   = relPos - wp.dot(relPos, zAxisNorm) * zAxisNorm
    rLocal = wp.length(rVec) / scale

    if rLocal > rMax:
        # Outside of the lens
        return noIntersection

    sgn  = wp.sign(primitive.f0) # Sign of R
    dSag = _sagDerivative(primitive, rLocal)

    if rLocal < wp.float32(1.e-8):
        # See comment below for sgn
        normal3D = -sgn * zAxisNorm
    else:
        radialUnit = rVec / wp.norm_l2(rVec)
        # The normal should be handled carefully: equation
        # of the sag is convex when R is taken as positive
        normal3D = sgn * (dSag * radialUnit - zAxisNorm)
        normal3D = wp.normalize(normal3D)

    return _Intersection(
        t           = tHit,
        normal      = normal3D,
        primitiveID = primitiveID,
    )

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
        elif primitive.type == 3: # Sphere (or spherical cap)
            candidateIntersection = _intersectSphere(ray, primitive, primitiveID)
        elif primitive.type == 4: # Cylinder
            candidateIntersection = _intersectCylinder(ray, primitive, primitiveID)
        elif primitive.type == 5: # Asphere
            candidateIntersection = _intersectAsphere(ray, primitive, primitiveID)
        else:
            continue

        if candidateIntersection.t > wp.float32(0.0):
            # We have a hit! Check if the closest one
            if closestIntersection.t < wp.float32(0.0) or candidateIntersection.t < closestIntersection.t:
                closestIntersection = candidateIntersection

    intersectionsBuffer[ID] = closestIntersection