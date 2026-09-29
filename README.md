# Gas Simulations

Interactive visualisations of how simple microscopic rules — particles bouncing off each other, atoms pulling
and pushing through the Lennard-Jones potential — give rise to macroscopic gas behaviour such as diffusion,
osmosis and the Maxwell–Boltzmann speed distribution.

The project is meant for exploring and demonstrating physics, not for quantitative molecular dynamics: the 2D
simulations use pixels and seconds as units, and parameters are chosen to look good and run in real time.

| Script | What it shows | Library |
| --- | --- | --- |
| [`atoms.py`](atoms.py) | A 2D box of gas with toggleable collisions, Lennard-Jones forces, gravity, a dividing wall and a semi-permeable (osmotic) membrane | pygame |
| [`boltzmann_billiard.py`](boltzmann_billiard.py) | One fast particle among stationary ones — collisions share out its energy until a Maxwell–Boltzmann speed distribution appears | pygame |
| [`boltzmann_billiard_record.py`](boltzmann_billiard_record.py) | The same experiment with 2025 particles, rendered to a 30-second MP4 | pygame + OpenCV |
| [`LJ_potential_curve_argon.py`](LJ_potential_curve_argon.py) | One argon atom oscillating in the Lennard-Jones well of another, with turning points, time-averaged position and effective stiffness | matplotlib + SciPy |

## Installation

Requires Python 3.10 or newer (developed on 3.14) and works on macOS, Linux and Windows.

```bash
git clone https://github.com/BChevallier/Gas_Simulations.git
cd Gas_Simulations
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy scipy matplotlib pygame opencv-python
```

`opencv-python` is only needed for the recording script, and `scipy` only for the argon script.

## Usage

Each script is standalone — run it directly:

```bash
python atoms.py
python boltzmann_billiard.py
python boltzmann_billiard_record.py
python LJ_potential_curve_argon.py
```

There are no command-line options. Every script is configured through the UPPER_CASE constants at the top of the
file (window size, particle count, which physics is enabled at start-up, and so on); edit them and re-run.
The pygame scripts print their starting parameters and controls to the terminal when they launch.

---

## `atoms.py` — the gas box

400 disks start on a jittered grid with random velocities inside an 800×800 window, bouncing elastically off the
walls. The physics is switched on and off live from the keyboard, so the same gas can be used for several
experiments.

### Controls

| Key | Action |
| --- | --- |
| `C` | Toggle hard-disk (elastic billiard-ball) collisions. Turns Lennard-Jones forces off. |
| `F` | Toggle Lennard-Jones forces. Turns collisions off. Unavailable while the wall is up. |
| `G` | Toggle gravity (only acts while Lennard-Jones forces are off) |
| `D` | Toggle damping — velocities are multiplied by 0.9999 every physics step, slowly cooling the gas |
| `B` | Toggle a solid wall down the middle of the box. Unavailable while forces are on. |
| `O` | Toggle an osmotic membrane down the middle (see below) |
| `R` | Move every particle into the left half with fresh random velocities |
| `H` | Toggle colouring by half: left half blue, right half red. The window title shows the count on each side. |
| `E` | Print the average kinetic, Lennard-Jones and total energy per particle |
| `Esc` | Quit |

### Experiments to try

- **Free expansion / diffusion** — press `B` to raise the wall, `R` to pack the gas into the left half, then `B`
  again to remove the wall. Turn on `H` to watch the counts in the title bar even out.
- **Osmosis** — every particle is randomly one of two species. With `O` on, the membrane blocks the red species
  but lets the blue one through, and the title bar tracks how many particles are on each side.
- **Condensation** — turn on Lennard-Jones forces with `F`, then slowly remove energy with `D` and watch the
  attraction between atoms take over as the gas cools.
- **Gravity** — turn on collisions and gravity (`C`, `G`) and watch the gas pile up toward the bottom of the box.

### Physics notes

- Integration uses velocity Verlet with `STEPS_PER_FRAME` physics steps per rendered frame.
- Lennard-Jones: σ = 2.2 × particle radius, cut off at 3σ. ε is chosen so that the average initial kinetic
  energy is `TARGET_RATIO` × ε, which sets whether the gas behaves more like a gas or a liquid.
- Pair searches use a uniform spatial grid (cell size ≥ the cut-off), and the force calculation is vectorised with
  NumPy, so a few hundred particles run in real time.
- Hard-disk collisions are resolved with position correction plus an impulse with a coefficient of restitution of 1
  (perfectly elastic).
- The Lennard-Jones potential energy reported by `E` is shifted so it is zero at the cut-off.
- Two particles on exactly the same spot have an undefined Lennard-Jones force, and the simulation stops with a
  `ZeroDivisionError`. Setting `CLAMP = True` gives such pairs a random push direction instead, at the cost of
  energy conservation — the `E` read-out warns about this.

---

## `boltzmann_billiard.py` — how a speed distribution emerges

400 particles start at rest, except one that moves at (1200, 1200) px/s. Each elastic collision shares energy,
and within seconds the whole gas is moving. The histogram under the box shows the speed distribution in real
time: it starts as a single spike at zero and settles into the 2D Maxwell–Boltzmann (Rayleigh) shape, without that
distribution ever being put into the code.

Particles are coloured by speed, from **blue (slow)** to **red (fast)**, and the histogram bars use the same scale.

| Key | Action |
| --- | --- |
| `C` | Toggle hard-disk collisions (turns forces off) |
| `F` | Toggle Lennard-Jones forces (turns collisions off) |
| `D` | Toggle damping |
| `E` | Print the energy per particle |
| `Esc` | Quit |

Useful settings at the top of the file: `ATOM_NUMBER`, `BIN_NUMBER` and `HIST_V_CUTOFF` (histogram resolution
and range), `MIN_COLOR_SPEED` / `MAX_COLOR_SPEED` (colour scale), `TIME_SCALE` (simulation speed).

---

## `boltzmann_billiard_record.py` — video export

Runs the billiard experiment at a larger scale (1000×1000 box, 2025 particles, fast particle starting at
(2000, 2000) px/s) and writes a video next to the script:

- `boltzmann_billiard.mp4`, 1000×1200 (box plus histogram), 60 FPS
- 5 seconds of the still starting state (a countdown shows in the window title but not in the video), then 25
  seconds of simulation

The simulation is rendered frame by frame, not in real time: expect several minutes for the 30-second clip. The
final message reports how many frames were written. Pressing `Esc` stops early but keeps the frames written so
far. The file is large (~230 MB) and uses the MPEG-4 Part 2 (`mp4v`) codec, which many browsers and social
platforms don't play; convert it to H.264 before sharing:

```bash
ffmpeg -i boltzmann_billiard.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 boltzmann_billiard_h264.mp4
```

To record without a window (e.g. on a server), use SDL's dummy video driver:

```bash
SDL_VIDEODRIVER=dummy python boltzmann_billiard_record.py
```

Video length, resolution, frame rate and particle count are set by `VIDEO_SECONDS`, `COUNTDOWN_SECONDS`, `WIDTH`,
`FPS` and `ATOM_NUMBER`. `COLOR_PROPAGATION = False` turns off speed colouring, leaving only the initially fast
particle in red.

---

## `LJ_potential_curve_argon.py` — one argon atom in a Lennard-Jones well

A one-dimensional model of an argon dimer: one atom is held fixed and the other oscillates in the 12-6
Lennard-Jones potential

U(r) = 4ε [ (σ/r)¹² − (σ/r)⁶ ]

using real argon parameters (σ = 3.405 Å, ε/k_B = 119.8 K, m = 39.948 u). The top panel plots the potential and
the bottom panel the force F = −dU/dr, both in SI units against the reduced separation r/σ.

**Sliders**

- **Temperature (K)** — the energy k_B·T given to the atom above the bottom of the well. The atom starts at
  rest at the inner turning point, so this is its total vibrational energy. At or above ε/k_B (119.8 K by default)
  the atom is unbound: it escapes, and the markers and averages disappear.
- **Epsilon (J)** — the depth of the well, from 0.4× to 1.6× argon's value.

**What's drawn**

- The atom (red marker) moving along both curves in real time
- The selected total energy (dashed horizontal line)
- The inner and outer turning points (grey dashed lines), found with `scipy.optimize.brentq`
- The time-averaged separation (red dotted line), computed by weighting each position by how long the atom spends
  there, with `scipy.integrate.quad`. Because the well is asymmetric, it moves outward as the temperature rises —
  a one-atom picture of **thermal expansion**.
- The time-averaged stiffness ⟨d²U/dr²⟩ in N/m (bottom right of the force panel, also printed to the terminal
  whenever a slider changes)

Internally everything runs in reduced Lennard-Jones units (r/σ, E/ε, t/τ with τ = σ√(m/ε)), integrated with
velocity Verlet; the `reduce_*` helpers convert to and from SI. `BALL = False` shows the curves without the
moving atom, and `DISPLAY_FORCE = False` hides the force panel.

**Plot backend:** matplotlib picks the native window backend for your OS. If you run the script from PyCharm, its
built-in plot pane can't animate or use sliders, so the script switches to a normal window automatically
(macOS backend on a Mac, Tk elsewhere). To force a backend, set `MPLBACKEND`, e.g.
`MPLBACKEND=QtAgg python LJ_potential_curve_argon.py`.

---

## Project layout

```
.
├── atoms.py                      # interactive gas box
├── boltzmann_billiard.py         # speed-distribution demo with live histogram
├── boltzmann_billiard_record.py  # the same demo rendered to MP4
├── LJ_potential_curve_argon.py   # argon atom in a Lennard-Jones well (matplotlib)
└── LICENSE
```

The three pygame scripts are self-contained and share much of their physics code (particle class, spatial grid,
Lennard-Jones forces, collision handling) by copying rather than importing, so a change to the physics in one
isn't automatically picked up by the others.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
