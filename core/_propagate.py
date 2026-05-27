import warp as wp

from _structures import _isFlat
from _structures import _Intersection, _Material, _Primitive, _Ray


# ─────────────────────────────────────────────────────────────────────
# Native CUDA intrinsics for bit reinterpretation
# ─────────────────────────────────────────────────────────────────────

@wp.func_native("""
return __float_as_int(x);
""")
def _floatToInt(x: float) -> int:
    ...


@wp.func_native("""
return __int_as_float(x);
""")
def _intToFloat(x: int) -> float:
    ...


# ─────────────────────────────────────────────────────────────────────
# Wächter-Binder offset (Ray Tracing Gems 2019, Chapter 6)
# ─────────────────────────────────────────────────────────────────────

@wp.func
def _wbOffsetComponent(p: float, n: float) -> float:
    """
    Compute the offset for a single coordinate component, following
    Wächter & Binder (2019).

    p: coordinate of the hit point (one component)
    n: corresponding component of the surface normal

    Returns the offset coordinate, safely moved 256 ULPs away from p
    along the direction of n (or by a fixed near-field offset for small p).
    """
    ORIGIN_THRESHOLD = wp.float32(1.0 / 32.0)
    FLOAT_SCALE      = wp.float32(1.0 / 65536.0)
    INT_SCALE        = wp.float32(256.0)

    # Integer offset proportional to the normal magnitude
    intOffset = wp.int32(INT_SCALE * n)

    # Far-field offset: bit-manipulate the float to add intOffset ULPs
    # Direction of offset depends on the sign of p
    bits = _floatToInt(p)
    if p < 0.0:
        farFieldOffset = _intToFloat(bits - intOffset)
    else:
        farFieldOffset = _intToFloat(bits + intOffset)

    # Near-field offset: additive, used when |p| is small
    nearFieldOffset = p + FLOAT_SCALE * n

    # Select based on magnitude of p
    if wp.abs(p) < ORIGIN_THRESHOLD:
        return nearFieldOffset
    else:
        return farFieldOffset


@wp.func
def _wachterBinderOffset(point: wp.vec3f, normal: wp.vec3f) -> wp.vec3f:
    """
    Robust ray origin offset following Wächter & Binder (2019).
    Scale-invariant, handles near-zero coordinates, requires no parameter tuning.

    point  : hit point on the surface (world coordinates)
    normal : surface normal at the hit (should point in the desired offset direction)

    Returns the offsetted point, safely on the side of `normal`.
    """
    return wp.vec3f(
        _wbOffsetComponent(point[0], normal[0]),
        _wbOffsetComponent(point[1], normal[1]),
        _wbOffsetComponent(point[2], normal[2]),
    )


@wp.func
def _reflect(rayDirection: wp.vec3f, normal: wp.vec3f) -> wp.vec3f:
    # Ray direction and interface normal are both supposed normalized
    return rayDirection - 2.0 * wp.dot(rayDirection, normal) * normal

@wp.func
def _refract(rayDirection: wp.vec3f, normal: wp.vec3f, nRatio: wp.float32) -> wp.vec3f:
    cosTheta1  = -wp.dot(rayDirection, normal)
    cosTheta2  = wp.sqrt(1.0 - nRatio * nRatio * (1.0 - cosTheta1 * cosTheta1))

    return nRatio * rayDirection + (nRatio * cosTheta1 - cosTheta2) * normal


@wp.kernel
def _propagate(
    # --- Input buffer ---
    intersectionsBuffer: wp.array(dtype=_Intersection, ndim=1),
    primitivesBuffer   : wp.array(dtype=_Primitive,    ndim=1),
    materialsBuffer    : wp.array(dtype=_Material,     ndim=1),

    # --- RNG control ---
    inputSeed: wp.int32,
    frameID  : wp.int32,

    # --- Input/output buffer ---
    raysBuffer: wp.array(dtype=_Ray, ndim=1)
):
    ID = wp.tid()

    ray = raysBuffer[ID]
    # Stop for dead rays
    if not ray.isAlive:
        return

    intersection = intersectionsBuffer[ID]
    # If there's no intersection, stop ray
    if intersection.t <= wp.float32(0.0):
        updatedRay = _Ray(
            origin     = ray.origin,
            direction  = ray.direction,
            throughput = ray.throughput,
            pdf        = ray.pdf,
            wavelength = ray.wavelength,

            isAlive    = False,
            depth      = ray.depth,

            sourcePrimitiveID = intersection.primitiveID
        )
        raysBuffer[ID] = updatedRay
        return

    primitive = primitivesBuffer[intersection.primitiveID]
    # A block may be intersected
    if primitive.materialID < 0:
        updatedRay = _Ray(
            origin     = ray.origin,
            direction  = ray.direction,
            throughput = ray.throughput,
            pdf        = ray.pdf,
            wavelength = ray.wavelength,

            isAlive    = False,
            depth      = ray.depth,

            sourcePrimitiveID = intersection.primitiveID
        )
        raysBuffer[ID] = updatedRay
        return

    hitPoint    = ray.origin + intersection.t * ray.direction
    cosIncident = wp.dot(ray.direction, intersection.normal)

    if cosIncident < wp.float32(0.0):
        # Ray arrives from the +normal side
        normalForward = intersection.normal
        material      = materialsBuffer[primitive.materialID]
        n1            = material.f0 # Refractive index in the +normal domain
        n2            = material.f1 # Refractive index in the -normal domain
    else:
        # Ray arrives from the -normal side
        normalForward = -intersection.normal
        material      = materialsBuffer[primitive.materialID]
        n1            = material.f1 # Refractive index in the -normal domain
        n2            = material.f0 # Refractive index in the +normal domain

    # --- Computation of the Fresnel reflectance ---
    cosTheta1  = -wp.dot(ray.direction, normalForward)
    nRatio     = n1 / n2
    sin2Theta2 = nRatio * nRatio * (1.0 - cosTheta1 * cosTheta1)

    if sin2Theta2 > wp.float32(1.0):
        # Total internal reflection
        reflectance = wp.float32(1.0)
        cosTheta2   = wp.float32(0.0)
    else:
        cosTheta2 = wp.sqrt(1.0 - sin2Theta2)
        rs = (n1 * cosTheta1 - n2 * cosTheta2) / (n1 * cosTheta1 + n2 * cosTheta2)
        rp = (n1 * cosTheta2 - n2 * cosTheta1) / (n1 * cosTheta2 + n2 * cosTheta1)
        reflectance = 0.5 * (rs * rs + rp * rp)

    # Pseudo-random state generation
    N        = raysBuffer.shape[0]
    rngState = wp.rand_init(inputSeed + ray.depth + 1, ID + frameID * N)
    u        = wp.randf(rngState)

    if u < reflectance:
        # Reflection
        offsetNormal = normalForward
        newDirection = _reflect(ray.direction, normalForward)
        # Throughput unchanged for reflection
        # PDF multiplied by event probability
        throughput = ray.throughput
        pdf = reflectance
    else:
        # Refraction
        offsetNormal = -normalForward
        newDirection = _refract(ray.direction, normalForward, nRatio)
        # Throughput scaled by (n2/n1)² for the radiance compression
        # (we are doing light tracing: light propagates from source)
        throughput = ray.throughput / (nRatio * nRatio)
        # PDF multiplied by the event probability
        pdf = wp.float32(1.0) - reflectance

    # Apply Wächter-Binder offset only for non-flat primitives
    if _isFlat(primitive):
        newOrigin = hitPoint
    else:
        newOrigin = _wachterBinderOffset(hitPoint, offsetNormal)

    updatedRay = _Ray(
        origin     = newOrigin,
        direction  = newDirection,
        throughput = throughput,
        pdf        = ray.pdf * pdf,
        wavelength = ray.wavelength,

        isAlive    = True,
        depth      = ray.depth + 1,

        sourcePrimitiveID = intersection.primitiveID
    )

    raysBuffer[ID] = updatedRay