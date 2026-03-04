import warp as wp

goldenRatio = wp.float32(1.6180339)

@wp.func
def fibonacciDiskSampling(n: wp.int32, N: wp.int32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    r   = wp.sqrt(n / N)
    psi = 2. * wp.pi * n / goldenRatio

    return wp.vec2(r, psi)

@wp.func
def fibonacciHemisphereSampling(n: wp.int32, N: wp.int32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    theta = wp.acos(1. - 2. * n / N) / 2.
    phi   = 2. * wp.pi * n / goldenRatio
    
    return wp.vec2(theta, phi)

@wp.func
def fibonacciHemisphericalCapSampling(n: wp.int32, N: wp.int32, thetaMax: wp.float32) -> wp.vec2:
    n = wp.float32(n)
    N = wp.float32(N)

    theta = wp.acos(1. - n * (1. - wp.cos(2. * thetaMax))) / 2.
    phi   = 2. * wp.pi * n / goldenRatio

    return wp.vec2(theta, phi)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    p = 6  # p-ième plus proche voisin

    goldenRatio = (1 + np.sqrt(5)) / 2

    Ns = []
    deltas = []
    scaled = []

    plt.ion()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    line_delta, = ax1.plot([], [], marker='o', label=f"delta_{p}(N)")
    line_ref,   = ax1.plot([], [], linestyle="--", label="1/sqrt(N)")
    line_scaled,= ax2.plot([], [], color="red", label=f"sqrt(N)*delta_{p}(N)")

    ax1.set_xlabel("N")
    ax1.set_ylabel("delta_p(N)")
    ax2.set_ylabel("sqrt(N)*delta_p(N)")
    ax1.set_title("Fibonacci disk neighbour spacing")
    ax1.grid(True)

    for N in range(1, 1024):

        if N < p + 1:
            continue

        pts = []

        for n in range(N):
            r   = np.sqrt(n / N)
            psi = 2.0 * np.pi * n / goldenRatio

            x = r * np.cos(psi)
            y = r * np.sin(psi)

            pts.append([x, y])

        pts = np.array(pts)

        delta = np.inf

        for i in range(N):

            d = np.linalg.norm(pts[i] - pts, axis=1)
            d[i] = np.inf

            d_sorted = np.sort(d)
            d_p = d_sorted[p-1]

            delta = min(delta, d_p)

        Ns.append(N)
        deltas.append(delta)
        scaled.append(np.sqrt(N) * delta)

        Ns_arr = np.array(Ns)

        line_delta.set_data(Ns_arr, deltas)
        line_ref.set_data(Ns_arr, 1.0 / np.sqrt(Ns_arr))
        line_scaled.set_data(Ns_arr, scaled)

        ax1.relim()
        ax1.autoscale_view()

        ax2.relim()
        ax2.autoscale_view()

        plt.draw()
        plt.pause(0.01)

    plt.ioff()
    plt.show()