import warp as wp

from _structures import _Ray, _Intersection, _Sensor

@wp.func
def _clipSegmentToWindow(
    x0: wp.float32, y0: wp.float32,
    x1: wp.float32, y1: wp.float32,
    xMin: wp.float32, yMin: wp.float32,
    xMax: wp.float32, yMax: wp.float32,
) -> wp.vec4f:
    """
    Liang-Barsky 2D segment clipping against an axis-aligned window.

    Returns (clippedX0, clippedY0, clippedX1, clippedY1).
    If the segment is fully outside the window, returns the original endpoints
    (the caller should test if t_enter > t_exit to detect this).

    Standard approach: parametrize the segment as p(t) = p0 + t * (p1 - p0), t ∈ [0, 1].
    Find the t interval [tEnter, tExit] where the point is inside the window.
    """
    dx = x1 - x0
    dy = y1 - y0

    tEnter = wp.float32(0.0)
    tExit  = wp.float32(1.0)

    # Test each of the 4 window edges
    # Edge: x >= xMin → dx * t >= xMin - x0
    # Sign of dx determines if it's an "entering" or "exiting" constraint

    # Left edge: x = xMin
    if dx != 0.0:
        t = (xMin - x0) / dx
        if dx > 0.0:
            # Entering through the left edge
            if t > tEnter:
                tEnter = t
        else:
            # Exiting through the left edge
            if t < tExit:
                tExit = t
    elif x0 < xMin:
        # Vertical segment outside the window
        return wp.vec4f(0.0, 0.0, 0.0, 0.0)   # signal: no visible portion

    # Right edge: x = xMax
    if dx != 0.0:
        t = (xMax - x0) / dx
        if dx > 0.0:
            if t < tExit:
                tExit = t
        else:
            if t > tEnter:
                tEnter = t
    elif x0 > xMax:
        return wp.vec4f(0.0, 0.0, 0.0, 0.0)

    # Bottom edge: y = yMin
    if dy != 0.0:
        t = (yMin - y0) / dy
        if dy > 0.0:
            if t > tEnter:
                tEnter = t
        else:
            if t < tExit:
                tExit = t
    elif y0 < yMin:
        return wp.vec4f(0.0, 0.0, 0.0, 0.0)

    # Top edge: y = yMax
    if dy != 0.0:
        t = (yMax - y0) / dy
        if dy > 0.0:
            if t < tExit:
                tExit = t
        else:
            if t > tEnter:
                tEnter = t
    elif y0 > yMax:
        return wp.vec4f(0.0, 0.0, 0.0, 0.0)

    # If tEnter > tExit, segment is outside the window
    if tEnter > tExit:
        return wp.vec4f(0.0, 0.0, 0.0, 0.0)

    # Compute clipped endpoints
    return wp.vec4f(
        x0 + tEnter * dx,
        y0 + tEnter * dy,
        x0 + tExit  * dx,
        y0 + tExit  * dy,
    )

@wp.kernel
def _visualize(
    raysBuffer         : wp.array(dtype=_Ray,           ndim=1),
    intersectionsBuffer: wp.array(dtype=_Intersection,  ndim=1),
    sensor             : _Sensor,
    sensorBuffer       : wp.array(dtype=wp.float32,   ndim=2),
):
    ID = wp.tid()

    ray = raysBuffer[ID]
    if not ray.isAlive:
        return

    intersection = intersectionsBuffer[ID]
    if intersection.t <= 0.0:
        return

    p0 = ray.origin
    p1 = ray.origin + intersection.t * ray.direction

    # ── Build the sensor's local frame ──
    sensorU      = sensor.v1 - sensor.v0
    sensorV      = sensor.v3 - sensor.v0
    sensorWidth  = wp.length(sensorU)
    sensorHeight = wp.length(sensorV)
    invWidth2    = 1.0 / (sensorWidth  * sensorWidth)
    invHeight2   = 1.0 / (sensorHeight * sensorHeight)

    # ── Project both endpoints to pixel coordinates ──
    relP0 = p0 - sensor.v0
    u0    = wp.dot(relP0, sensorU) * invWidth2
    v0    = wp.dot(relP0, sensorV) * invHeight2

    relP1 = p1 - sensor.v0
    u1    = wp.dot(relP1, sensorU) * invWidth2
    v1    = wp.dot(relP1, sensorV) * invHeight2

    resX = wp.float32(sensorBuffer.shape[0])
    resY = wp.float32(sensorBuffer.shape[1])

    x0 = u0 * resX - 0.5
    y0 = v0 * resY - 0.5
    x1 = u1 * resX - 0.5
    y1 = v1 * resY - 0.5

    # ── Clip the segment to the image window ──
    # Slightly expanded window to ensure we catch all pixels at the boundary
    clipped = _clipSegmentToWindow(
        x0, y0, x1, y1,
        wp.float32(-0.5), wp.float32(-0.5),
        wp.float32(resX - 0.5), wp.float32(resY - 0.5),
    )

    # Detect "no visible portion" (all zeros from clipping)
    if clipped[0] == 0.0 and clipped[1] == 0.0 and clipped[2] == 0.0 and clipped[3] == 0.0:
        return

    cx0 = clipped[0]
    cy0 = clipped[1]
    cx1 = clipped[2]
    cy1 = clipped[3]

    # ── DDA-style rasterization on the clipped segment ──
    dx = cx1 - cx0
    dy = cy1 - cy0

    nSteps = wp.int32(wp.max(wp.abs(dx), wp.abs(dy)) + 1.0)
    # No cap needed now: clipped segment is at most diagonal of image (~1000 pixels)

    invNSteps = 1.0 / wp.float32(nSteps)

    for step in range(nSteps):
        t = wp.float32(step) * invNSteps
        x = cx0 + t * dx
        y = cy0 + t * dy

        xi = wp.int32(wp.round(x))
        yi = wp.int32(wp.round(y))

        # Safety check (should always be in bounds after clipping)
        if xi >= 0 and xi < sensorBuffer.shape[0] and yi >= 0 and yi < sensorBuffer.shape[1]:
            wp.atomic_add(sensorBuffer, xi, yi, wp.float32(1.0))

"""
@wp.kernel
def _visualize(
    raysBuffer         : wp.array(dtype=_Ray,           ndim=1),
    intersectionsBuffer: wp.array(dtype=_Intersection,  ndim=1),
    sensor             : _Sensor,
    sensorBuffer       : wp.array(dtype=wp.float32,   ndim=2),
):
    ID = wp.tid()

    ray = raysBuffer[ID]
    if not ray.isAlive:
        return

    intersection = intersectionsBuffer[ID]
    if intersection.t <= 0.0:
        return

    p0 = ray.origin
    p1 = ray.origin + intersection.t * ray.direction

    # ── Build the sensor's local frame ──
    sensorU      = sensor.v1 - sensor.v0
    sensorV      = sensor.v3 - sensor.v0
    sensorWidth  = wp.length(sensorU)
    sensorHeight = wp.length(sensorV)
    invWidth2    = 1.0 / (sensorWidth  * sensorWidth)
    invHeight2   = 1.0 / (sensorHeight * sensorHeight)

    # ── Project both endpoints onto sensor plane ──
    relP0 = p0 - sensor.v0
    u0    = wp.dot(relP0, sensorU) * invWidth2
    v0    = wp.dot(relP0, sensorV) * invHeight2

    relP1 = p1 - sensor.v0
    u1    = wp.dot(relP1, sensorU) * invWidth2
    v1    = wp.dot(relP1, sensorV) * invHeight2

    # ── Convert to pixel coordinates ──
    resX = wp.float32(sensorBuffer.shape[0])
    resY = wp.float32(sensorBuffer.shape[1])

    x0 = u0 * resX - 0.5
    y0 = v0 * resY - 0.5
    x1 = u1 * resX - 0.5
    y1 = v1 * resY - 0.5

    # ── DDA-style rasterization ──
    dx = x1 - x0
    dy = y1 - y0

    # Number of steps = max axial distance in pixels (+1)
    nSteps = wp.int32(wp.max(wp.abs(dx), wp.abs(dy)) + 1.0)

    # Each splat contributes 1/nSteps, so total contribution is 1.0 (independent of length)
    contribution = 1.0 / wp.float32(nSteps)

    # Cap for GPU performance
    MAX_STEPS = wp.int32(8192*8192)
    if nSteps > MAX_STEPS:
        nSteps = MAX_STEPS

    invNSteps = 1.0 / wp.float32(nSteps)

    for step in range(nSteps):
        t = wp.float32(step) * invNSteps
        x = x0 + t * dx
        y = y0 + t * dy

        xi = wp.int32(wp.round(x))
        yi = wp.int32(wp.round(y))

        if xi >= 0 and xi < sensorBuffer.shape[0] and yi >= 0 and yi < sensorBuffer.shape[1]:
            wp.atomic_add(sensorBuffer, xi, yi, contribution)
"""

@wp.func
def _orientation2D(a: wp.vec2f, b: wp.vec2f, c: wp.vec2f) -> wp.float32:
    """
    Returns the signed cross product of (b - a) and (c - a).
    Positive = c is to the left of a→b; negative = right; zero = collinear.
    """
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


@wp.func
def _doSegmentsIntersect2D(a0: wp.vec2f, a1: wp.vec2f,
                            b0: wp.vec2f, b1: wp.vec2f) -> wp.bool:
    """
    Returns True if segments [a0, a1] and [b0, b1] intersect in 2D.
    Handles general case (no special collinear handling — sufficient for debug).
    """
    o1 = _orientation2D(a0, a1, b0)
    o2 = _orientation2D(a0, a1, b1)
    o3 = _orientation2D(b0, b1, a0)
    o4 = _orientation2D(b0, b1, a1)

    # General case: signs of o1, o2 differ AND signs of o3, o4 differ
    return (o1 * o2 < 0.0) and (o3 * o4 < 0.0)


@wp.func
def _pointInRect(p: wp.vec2f,
                  rectMin: wp.vec2f,
                  rectMax: wp.vec2f) -> wp.bool:
    """Returns True if point p is inside the axis-aligned rectangle."""
    return (p[0] >= rectMin[0] and p[0] <= rectMax[0] and
            p[1] >= rectMin[1] and p[1] <= rectMax[1])


@wp.func
def _segmentIntersectsRect(s0: wp.vec2f, s1: wp.vec2f,
                            rectMin: wp.vec2f, rectMax: wp.vec2f) -> wp.bool:
    """
    Tests if a 2D segment [s0, s1] touches an axis-aligned rectangle.
    Returns True if:
      - either endpoint is inside the rectangle, OR
      - the segment crosses at least one of the 4 edges.
    """
    # Quick check: endpoint inside?
    if _pointInRect(s0, rectMin, rectMax):
        return True
    if _pointInRect(s1, rectMin, rectMax):
        return True

    # Otherwise: check intersection with each of the 4 edges
    p00 = rectMin
    p10 = wp.vec2f(rectMax[0], rectMin[1])
    p11 = rectMax
    p01 = wp.vec2f(rectMin[0], rectMax[1])

    if _doSegmentsIntersect2D(s0, s1, p00, p10):  return True
    if _doSegmentsIntersect2D(s0, s1, p10, p11):  return True
    if _doSegmentsIntersect2D(s0, s1, p11, p01):  return True
    if _doSegmentsIntersect2D(s0, s1, p01, p00):  return True

    return False

@wp.kernel
def _visualizeRayPaths(
    raysBuffer         : wp.array(dtype=_Ray,          ndim=1),
    intersectionsBuffer: wp.array(dtype=_Intersection, ndim=1),
    nbRays             : wp.int32,
    sensor             : _Sensor,
    sensorBuffer       : wp.array(dtype=wp.float32, ndim=2),
):
    """
    Pixel-parallel exact rasterization of ray paths.
    Each thread = one pixel. It loops over all rays and tests if the projected
    segment [ray.origin, ray.origin + t * ray.direction] crosses its pixel.
    Accumulates into the existing buffer content.
    """
    i, j = wp.tid()

    # Build sensor frame
    sensorU      = sensor.v1 - sensor.v0
    sensorV      = sensor.v3 - sensor.v0
    sensorWidth  = wp.length(sensorU)
    sensorHeight = wp.length(sensorV)
    invWidth2    = 1.0 / (sensorWidth  * sensorWidth)
    invHeight2   = 1.0 / (sensorHeight * sensorHeight)

    # Pixel bounds in (u, v)
    resX = wp.float32(sensorBuffer.shape[0])
    resY = wp.float32(sensorBuffer.shape[1])
    rectMin = wp.vec2f(wp.float32(i) / resX, wp.float32(j) / resY)
    rectMax = wp.vec2f(wp.float32(i + 1) / resX, wp.float32(j + 1) / resY)

    # Loop over rays
    accumulator = wp.float32(0.0)

    for rayID in range(nbRays):
        ray = raysBuffer[rayID]
        if not ray.isAlive:
            continue

        intersection = intersectionsBuffer[rayID]
        if intersection.t <= 0.0:
            continue

        # Project segment
        p0 = ray.origin
        p1 = ray.origin + intersection.t * ray.direction

        relP0 = p0 - sensor.v0
        relP1 = p1 - sensor.v0

        s0 = wp.vec2f(wp.dot(relP0, sensorU) * invWidth2,
                       wp.dot(relP0, sensorV) * invHeight2)
        s1 = wp.vec2f(wp.dot(relP1, sensorU) * invWidth2,
                       wp.dot(relP1, sensorV) * invHeight2)

        if _segmentIntersectsRect(s0, s1, rectMin, rectMax):
            accumulator += wp.float32(1.0)

    # Accumulate (no atomic needed: each thread writes its own pixel)
    sensorBuffer[i, j] = sensorBuffer[i, j] + accumulator