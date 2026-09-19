"""
3D transient heat conduction model of a rectangular furnace wall.

The temperature field is solved using an explicit finite-difference method,
considering conduction only under simplified thermal boundary conditions.

    python furnace_heat_3d.py             run and show the plots
    python furnace_heat_3d.py --no-show   run, only save the plots
    python furnace_heat_3d.py --test      run the verification tests
"""

from pathlib import Path
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- CHANGE THESE --------------------------
LX, LY, LZ = 0.155, 0.30, 0.30   # thickness, width, height (m)
NX, NY, NZ = 20, 12, 12          # intervals; node counts are NX+1, NY+1, NZ+1
TOTAL_TIME = 3600.0              # s
K = 0.45                         # thermal conductivity, W/(m.K)
RHO = 900.0                      # density, kg/m3
CP = 1050.0                      # specific heat capacity, J/(kg.K)
T_INIT = 290.0                   # degC (an already warm wall, NOT room temperature)
T_LEFT = 100.0                   # degC, face x = 0
T_HOT = 1250.0                   # degC, hot patch on face x = L
T_BACKGROUND = 100.0             # degC, face x = L outside the hot patch
HOT_PATCH = True                 # False = the whole x = L face is at T_HOT
PATCH_WIDTH_FRACTION = 0.40      # patch width as a fraction of the width, centered
PATCH_HEIGHT_FRACTION = 0.40     # patch height as a fraction of the height, centered
# ------------------------------------------------------------------


def solve(lx, ly, lz, nx, ny, nz, total_time, alpha,
          T_start, left_face, right_face, number_of_snapshots):
    """Explicit finite differences in 3D (conduction only).

    left_face and right_face can be one number or a 2D array.
    Returns: snapshots (full 3D fields, including t = 0), times, dt, mx, my, mz
    """
    dx = lx / nx
    dy = ly / ny
    dz = lz / nz

    # Stability of the explicit scheme: mx + my + mz <= 0.5.
    # We use 90% of the largest stable dt, then adjust it so that
    # the last step lands exactly on total_time.
    dt_max = 1.0 / (2.0 * alpha * (1 / dx**2 + 1 / dy**2 + 1 / dz**2))
    steps = math.ceil(total_time / (0.9 * dt_max))
    dt = total_time / steps
    mx = alpha * dt / dx**2
    my = alpha * dt / dy**2
    mz = alpha * dt / dz**2

    # Starting field, with the two x faces imposed (they never change)
    T = T_start.copy()
    T[0, :, :] = left_face
    T[nx, :, :] = right_face

    # Steps at which a snapshot is kept (step 0 is the starting field)
    snapshot_steps = []
    for n in range(number_of_snapshots):
        snapshot_steps.append(int(steps * n / (number_of_snapshots - 1)))
    snapshots = [T.copy()]
    times = [0.0]

    for step in range(1, steps + 1):
        old = T.copy()     # previous time level

        for i in range(1, nx):            # x = 0 and x = L are fixed, so skip them
            for j in range(ny + 1):
                # Insulated sides (zero gradient): at the edge, the missing
                # neighbour is replaced by the mirror node (reflected ghost node).
                jm = j - 1 if j > 0 else j + 1
                jp = j + 1 if j < ny else j - 1
                for k in range(nz + 1):
                    km = k - 1 if k > 0 else k + 1
                    kp = k + 1 if k < nz else k - 1

                    c = old[i][j][k]
                    T[i, j, k] = (c
                                  + mx * (old[i + 1][j][k] - 2 * c + old[i - 1][j][k])
                                  + my * (old[i][jp][k] - 2 * c + old[i][jm][k])
                                  + mz * (old[i][j][kp] - 2 * c + old[i][j][km]))

        if step in snapshot_steps:
            snapshots.append(T.copy())
            times.append(step * dt)

    return snapshots, times, dt, mx, my, mz


def self_test():
    """Verification tests. They use the same solve() as the real run."""

    # Tests 1 and 2: exact 3D transient solution + grid refinement.
    # Unit cube, alpha = 1, x faces fixed (100 and 200), other faces insulated.
    # Exact: T = 100 + 100*x + 10*sin(pi*x)*cos(pi*y)*cos(pi*z)*exp(-3*pi^2*t)
    t_end = 0.05
    decay = math.exp(-3 * math.pi**2 * t_end)   # time factor at t = t_end
    errors = []

    for n in (8, 16):                            # intervals in each direction
        h = 1.0 / n
        mode = np.zeros((n + 1, n + 1, n + 1))   # spatial shape of the transient part
        T_start = np.zeros((n + 1, n + 1, n + 1))
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    x = i * h
                    y = j * h
                    z = k * h
                    mode[i, j, k] = math.sin(math.pi * x) * math.cos(math.pi * y) * math.cos(math.pi * z)
                    T_start[i, j, k] = 100 + 100 * x + 10 * mode[i, j, k]

        snapshots, times, dt, mx, my, mz = solve(1.0, 1.0, 1.0, n, n, n, t_end, 1.0,
                                                 T_start, 100.0, 200.0, 2)
        final = snapshots[-1]

        # Largest difference between numerical and exact solution
        largest_error = 0.0
        for i in range(n + 1):
            for j in range(n + 1):
                for k in range(n + 1):
                    exact = 100 + 100 * (i * h) + 10 * mode[i, j, k] * decay
                    largest_error = max(largest_error, abs(final[i, j, k] - exact))
        errors.append(largest_error)

        assert mx + my + mz <= 0.5 + 1e-12       # stability criterion

        # In every snapshot: fixed faces stay fixed, and temperatures stay
        # between the starting minimum and maximum (no new hot or cold spots)
        for snapshot in snapshots:
            assert np.all(snapshot[0, :, :] == 100.0)
            assert np.all(snapshot[n, :, :] == 200.0)
            assert snapshot.min() >= T_start.min() - 1e-10
            assert snapshot.max() <= T_start.max() + 1e-10

    # Halving the grid spacing must cut the error by about 4 (second-order scheme)
    assert errors[1] < errors[0] / 3, errors
    print(f"PASS: analytical 3D transient; errors {errors[0]:.6f}, {errors[1]:.6f} degC")
    print(f"PASS: refinement (error ratio {errors[0] / errors[1]:.2f}), stability, prescribed faces, temperature bounds")

    # Test 3: uniform end-face temperatures must match a plain 1D calculation
    nx, ny, nz = 10, 6, 6
    alpha = 0.45 / (900.0 * 1050.0)
    T_start = np.full((nx + 1, ny + 1, nz + 1), 290.0)
    snapshots, times, dt, mx, my, mz = solve(0.155, 0.3, 0.3, nx, ny, nz, 3600.0, alpha,
                                             T_start, 100.0, 1250.0, 2)
    final = snapshots[-1]

    one_d = [290.0] * (nx + 1)                   # independent 1D calculation, same dt
    one_d[0] = 100.0
    one_d[nx] = 1250.0
    for step in range(round(3600.0 / dt)):
        old_1d = one_d.copy()
        for i in range(1, nx):
            one_d[i] = old_1d[i] + mx * (old_1d[i + 1] - 2 * old_1d[i] + old_1d[i - 1])

    biggest_difference = 0.0                     # every (j, k) line must equal the 1D result
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                biggest_difference = max(biggest_difference, abs(final[i, j, k] - one_d[i]))
    assert biggest_difference < 1e-9, biggest_difference
    print("PASS: uniform-face case matches independent 1D calculation")

    # Test 4: a straight-line profile between the two faces is the steady state,
    # so it must not change
    T_start = np.zeros((nx + 1, ny + 1, nz + 1))
    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                T_start[i, j, k] = 100.0 + (1250.0 - 100.0) * i / nx
    snapshots, times, dt, mx, my, mz = solve(0.155, 0.3, 0.3, nx, ny, nz, 3600.0, alpha,
                                             T_start, 100.0, 1250.0, 2)
    biggest_change = np.abs(snapshots[-1] - T_start).max()
    assert biggest_change < 1e-9, biggest_change
    print("PASS: analytical steady-state profile remains stationary")


def main(show_plots):
    alpha = K / (RHO * CP)
    dy = LY / NY
    dz = LZ / NZ

    # Face x = L: hot patch in the middle, background temperature around it
    right_face = np.full((NY + 1, NZ + 1), T_HOT)
    if HOT_PATCH:
        right_face[:, :] = T_BACKGROUND
        for j in range(NY + 1):
            for k in range(NZ + 1):
                in_patch = (abs(j * dy - LY / 2) <= PATCH_WIDTH_FRACTION * LY / 2 and
                            abs(k * dz - LZ / 2) <= PATCH_HEIGHT_FRACTION * LZ / 2)
                if in_patch:
                    right_face[j, k] = T_HOT

    T_start = np.full((NX + 1, NY + 1, NZ + 1), T_INIT)
    snapshots, times, dt, mx, my, mz = solve(LX, LY, LZ, NX, NY, NZ, TOTAL_TIME, alpha,
                                             T_start, T_LEFT, right_face, 6)
    final = snapshots[-1]
    history = np.array(snapshots)   # shape: (snapshot, x, y, z)

    print(f"Grid: {NX + 1} x {NY + 1} x {NZ + 1} nodes")
    print(f"alpha = {alpha:.6e} m²/s | dt = {dt:.6f} s")
    print(f"Mx, My, Mz = [{mx:.8f} {my:.8f} {mz:.8f}]")
    print(f"sum = {mx + my + mz:.6f} <= 0.5")
    print(f"Number of time steps: {round(TOTAL_TIME / dt)}")
    print(f"Center temperature: {final[NX // 2, NY // 2, NZ // 2]:.3f} °C")

    # ---------------- save the numbers ----------------
    x = np.linspace(0.0, LX, NX + 1)
    y = np.linspace(0.0, LY, NY + 1)
    z = np.linspace(0.0, LZ, NZ + 1)

    output_folder = Path(__file__).resolve().parent / "results_3d"
    output_folder.mkdir(exist_ok=True)

    np.savez_compressed(output_folder / "temperature_snapshots.npz",
                        x=x, y=y, z=z, times_s=times, T_degC=history,
                        dt_s=dt, fourier_numbers=[mx, my, mz])

    with open(output_folder / "final_temperature.csv", "w") as file:
        file.write("x_m,y_m,z_m,T_degC\n")
        for i in range(NX + 1):
            for j in range(NY + 1):
                for k in range(NZ + 1):
                    file.write(f"{x[i]:.8f},{y[j]:.8f},{z[k]:.8f},{final[i, j, k]:.8f}\n")

    vmin = history.min()   # same colour scale for the slices and the 3D view
    vmax = history.max()

    # ---------------- figure 1: temperature slices ----------------
    fig1, axs = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    slices = [
        (x, y, final[:, :, NZ // 2].T, "x (thickness, m)", "y (width, m)", f"z = {z[NZ // 2]:.3f} m"),
        (x, z, final[:, NY // 2, :].T, "x (thickness, m)", "z (height, m)", f"y = {y[NY // 2]:.3f} m"),
        (y, z, final[NX // 2, :, :].T, "y (width, m)", "z (height, m)", f"x = {x[NX // 2]:.3f} m"),
        (y, z, final[NX, :, :].T, "y (width, m)", "z (height, m)", "Prescribed hot face, x = L"),
    ]
    for ax, (a, b, data, xlabel, ylabel, title) in zip(axs.flat, slices):
        mesh = ax.pcolormesh(a, b, data, shading="auto", cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.set_aspect("equal")
    fig1.colorbar(mesh, ax=list(axs.flat), label="Temperature (degC)", shrink=0.8)
    fig1.suptitle(f"3D wall: temperature slices at t = {TOTAL_TIME:g} s")
    fig1.savefig(output_folder / "temperature_slices.png", dpi=150)

    # ---------------- figure 2: 3D cutaway node view ----------------
    # One corner (an octant) is left out so that the inside can be seen.
    px, py, pz, pt = [], [], [], []
    for i in range(NX + 1):
        for j in range(NY + 1):
            for k in range(NZ + 1):
                if x[i] > LX / 2 and y[j] < LY / 2 and z[k] > LZ / 2:
                    continue   # hidden corner
                px.append(x[i])
                py.append(y[j])
                pz.append(z[k])
                pt.append(final[i, j, k])

    fig2 = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax3 = fig2.add_subplot(111, projection="3d")
    points = ax3.scatter(px, py, pz, c=pt, s=10, cmap="inferno",
                         vmin=vmin, vmax=vmax, alpha=0.65)
    ax3.set(xlabel="x (m)", ylabel="y (m)", zlabel="z (m)",
            title="Final 3D field - cutaway node view")
    ax3.set_box_aspect((LX, LY, LZ))
    fig2.colorbar(points, ax=ax3, label="Temperature (degC)", shrink=0.7)
    fig2.savefig(output_folder / "temperature_3d.png", dpi=150)

    # ---------------- figure 3: centreline through the wall ----------------
    fig3, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for n in range(len(times)):
        ax.plot(x, history[n, :, NY // 2, NZ // 2], label=f"{times[n]:.0f} s")
    ax.set(xlabel="x (thickness, m)", ylabel="Temperature (degC)",
           title="Temperature along the central line through the wall")
    ax.grid(alpha=0.25)
    ax.legend(title="Time")
    fig3.savefig(output_folder / "centerline_history.png", dpi=150)

    print(f"Saved 3 plots, final CSV, and snapshot NPZ in: {output_folder}")
    if show_plots:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="run the verification tests")
    parser.add_argument("--no-show", action="store_true", help="save the plots without opening windows")
    args = parser.parse_args()

    if args.test:
        self_test()
    else:
        main(show_plots=not args.no_show)