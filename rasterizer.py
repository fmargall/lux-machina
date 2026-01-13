import warp as wp

from structures import Intersection, Ray, Segment

"""
   Creates a segment from a ray and an intersection point
"""
@wp.func
def segmentize(
    ray         : Ray,
    intersection: Intersection
) -> Segment:
    return Segment(ray.origin, intersection.hitPoint, ray.energy)

@wp.func
def ccw(a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.float32: 
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

@wp.func
def doSegmentsIntersect(segment0: Segment, segment1: Segment) -> wp.bool:
    return (ccw(segment0.v0, segment0.v1, segment1.v0) * ccw(segment0.v0, segment0.v1, segment1.v1) <= 0.0 and
            ccw(segment1.v0, segment1.v1, segment0.v0) * ccw(segment1.v0, segment1.v1, segment0.v1) <= 0.0)

"""
   We are here already working in screen coordinates;
   Pixel (i,j) defined by corners p00, p01, p10, p11:
    (0 : 1) --------------------------------- (1 : 1)
       |                                         |
       |             (p01)     (p11)             |
       |                  *---*                  |
       |                  |   |                  |
       |                  *---*                  |
       |             (p00)     (p10)             |
       |                                         |
    (0 : 0) --------------------------------- (1 : 0)

   What that means is that every light source, every primitive,
   once defined by the user needs to be transformed into screen
   space coordinates before launching the rasterization kernel,
   i.e. in a square defined on [0 ; 1] in both axis.
"""
@wp.kernel
def rasterize(
    raysBuffer         : wp.array(dtype=Ray, ndim=1),
    intersectionsBuffer: wp.array(dtype=Intersection, ndim=1),
    nbParallelRays     : wp.int32,
    
    imageBufferHeight: wp.int32,
    imageBufferWidth : wp.int32,
    imageBuffer      : wp.array(dtype=wp.float32, ndim=2)
):
    # Get pixel IDs
    i, j = wp.tid()

    halfDeltaX = wp.float32(1.) / (2. * wp.float32(imageBufferWidth))
    halfDeltaY = wp.float32(1.) / (2. * wp.float32(imageBufferHeight))

    # Careful: since i is the rowID and j colID,
    # i defines the position on y, then j on x.
    pixelCenter = wp.vec2(
                            (wp.float32(j) / wp.float32(imageBufferWidth))  + halfDeltaX,
        (wp.float32(imageBufferHeight - i) / wp.float32(imageBufferHeight)) + halfDeltaY
    )

    p00 = pixelCenter + wp.vec2(-halfDeltaX, -halfDeltaY);
    p01 = pixelCenter + wp.vec2(-halfDeltaX,  halfDeltaY);
    p10 = pixelCenter + wp.vec2( halfDeltaX, -halfDeltaY);
    p11 = pixelCenter + wp.vec2( halfDeltaX,  halfDeltaY);

    for ID in range(nbParallelRays):
        segment = segmentize(raysBuffer[ID], intersectionsBuffer[ID])
        hit = False

        if doSegmentsIntersect(segment, Segment(p00, p10, 1.)): hit = True
        if doSegmentsIntersect(segment, Segment(p10, p11, 1.)): hit = True
        if doSegmentsIntersect(segment, Segment(p11, p01, 1.)): hit = True
        if doSegmentsIntersect(segment, Segment(p01, p00, 1.)): hit = True

        if hit:
            wp.atomic_add(imageBuffer, i, j, segment.f0)