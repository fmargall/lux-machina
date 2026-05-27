import warp as wp

from _structures import _Ray, _Intersection, _Sensor

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
    
    # Cap for GPU performance
    MAX_STEPS = wp.int32(512)
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
            wp.atomic_add(sensorBuffer, xi, yi, wp.float32(1.0))