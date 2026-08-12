import math
import random as rnd
from collections.abc import Iterator

import numpy as np
import pygame

import time

# PARAMS
COLLISIONS = True
FORCES = False
DAMPING = False
HISTOGRAM = True
COlOR_PROPAGATION = True
MAX_COLOR_SPEED = 200
MIN_COLOR_SPEED = 0
MIN_COLOR = np.array([90, 190, 255], dtype=float)
MAX_COLOR = np.array([255, 95, 80], dtype=float)
BIN_NUMBER = 130
HIST_V_CUTOFF = 600

WIDTH = 650
HEIGHT = WIDTH  # The simulation currently assumes a square container.
HIST_HEIGHT = 200
FPS = 40
TIME_SCALE = 0.5
STEPS_PER_FRAME = int(round(TIME_SCALE * 12))

PARTICLE_SIZE = 6
PARTICLE_MASS = 1.0
ATOM_NUMBER = round(math.sqrt(400)) ** 2
ATOM_PAIRS = ATOM_NUMBER * (ATOM_NUMBER - 1) / 2

INITIAL_MAX_SPEED = 80
AVG_ENERGY = (
    INITIAL_MAX_SPEED**2
    / 3
    * PARTICLE_MASS
)

CLAMP = False

TARGET_RATIO = 0.8  # Desired ratio of E to epsilon.

# Lennard-Jones parameters.
LJ_SIGMA = 2.2 * PARTICLE_SIZE
LJ_EPSILON = AVG_ENERGY / TARGET_RATIO
LJ_CUTOFF = 3 * LJ_SIGMA
LJ_MIN_DISTANCE = 0.7 * LJ_SIGMA

GRID_SIZE = max(1, int(WIDTH / LJ_CUTOFF))
CELL_SIZE = WIDTH / GRID_SIZE

CALCULATE_ENERGY = False


INSTRUCTIONS = INSTRUCTIONS = f"""
========================================
          BOLTZMANN'S BILLIARD
========================================

A 2D simulation of energy redistribution
through elastic particle collisions.

Initially, {ATOM_NUMBER - 1} particles are stationary
and one particle carries all kinetic energy.

As collisions occur, energy is transferred between
particles and a statistical speed distribution emerges.

Particle color indicates speed:
  BLUE  = slow
  RED   = fast

The histogram shows the instantaneous
distribution of particle speeds.

--- SIMULATION ---
Particles:           {ATOM_NUMBER}
Particle radius:     {PARTICLE_SIZE} px
Initial fast vx:     1200 px/s
Initial fast vy:     1200 px/s
Physics steps/frame: {STEPS_PER_FRAME}
Target frame rate:   {FPS} FPS

--- HISTOGRAM ---
Bins:                {BIN_NUMBER}
Displayed range:     0 - {HIST_V_CUTOFF} px/s
Blue/red color range:{MIN_COLOR_SPEED} - {MAX_COLOR_SPEED} px/s

--- CONTROLS ---
C    Toggle hard-disk collisions
F    Toggle Lennard-Jones forces
D    Toggle damping
E    Print system energy
ESC  Exit

========================================
"""


class Atom:
    def __init__(
        self,
        x: float,
        y: float,
        vel: list[float],
        color: str = "lightblue",
        size: int = PARTICLE_SIZE,
        mass: float = PARTICLE_MASS,
        type: int = 0,
    ) -> None:
        self.x = float(x)
        self.y = float(y)

        self.vel = [float(vel[0]), float(vel[1])]
        self.acc = [0.0, 0.0]

        self.size = size
        self.color = color
        self.mass = mass
        self.type = type

    def move(self, dt: float) -> None:
        """Move the particle and reflect it from the container walls."""

        new_x = self.x + dt * self.vel[0]
        new_y = self.y + dt * self.vel[1]

        left_wall = self.size
        right_wall = WIDTH - self.size
        top_wall = self.size
        bottom_wall = HEIGHT - self.size

        if new_x < left_wall:
            new_x = 2.0 * left_wall - new_x
            self.vel[0] = abs(self.vel[0])

        elif new_x > right_wall:
            new_x = 2.0 * right_wall - new_x
            self.vel[0] = -abs(self.vel[0])

        if new_y < top_wall:
            new_y = 2.0 * top_wall - new_y
            self.vel[1] = abs(self.vel[1])

        elif new_y > bottom_wall:
            new_y = 2.0 * bottom_wall - new_y
            self.vel[1] = -abs(self.vel[1])

        self.x = min(max(new_x, left_wall), right_wall)
        self.y = min(max(new_y, top_wall), bottom_wall)

    def kinetic_energy(self) -> float:
        speed_squared = self.vel[0] ** 2 + self.vel[1] ** 2
        return 0.5 * self.mass * speed_squared

def clear_accelerations(particles: list[Atom]) -> None:
    for particle in particles:
        particle.acc[0] = 0.0
        particle.acc[1] = 0.0

def grid_coordinates(particle: Atom) -> tuple[int, int]:
    """Return safe grid coordinates for a particle."""
    grid_x = int(particle.x // CELL_SIZE)
    grid_y = int(particle.y // CELL_SIZE)

    grid_x = min(max(grid_x, 0), GRID_SIZE - 1)
    grid_y = min(max(grid_y, 0), GRID_SIZE - 1)

    return grid_x, grid_y

Grid = list[list[list[int]]]

def build_grid(particles: list[Atom]) -> Grid:
    """Build a spatial grid containing each particle index exactly once."""

    grid: Grid = [
        [[] for _ in range(GRID_SIZE)]
        for _ in range(GRID_SIZE)
    ]

    for index, particle in enumerate(particles):
        grid_x, grid_y = grid_coordinates(particle)
        grid[grid_y][grid_x].append(index)

    return grid

def iter_candidate_pairs(
    grid: Grid,
    cell_radius: int,
) -> Iterator[tuple[int, int]]:
    """Yield each candidate particle pair exactly once.

    Pairs within one cell are emitted once. For distinct cells, only forward
    cell pairs are visited: cells to the right on the same row, and cells on
    later rows within ``cell_radius``. This eliminates duplicate pair searches
    without using temporary sets.
    """

    for grid_y in range(GRID_SIZE):
        for grid_x in range(GRID_SIZE):
            current_cell = grid[grid_y][grid_x]

            if not current_cell:
                continue

            # Pairs within the current cell.
            particle_count = len(current_cell)
            for a in range(particle_count - 1):
                i = current_cell[a]
                for b in range(a + 1, particle_count):
                    yield i, current_cell[b]

            # Pairs between this cell and forward neighboring cells.
            max_y = min(GRID_SIZE - 1, grid_y + cell_radius)
            min_x = max(0, grid_x - cell_radius)
            max_x = min(GRID_SIZE - 1, grid_x + cell_radius)

            for neighbor_y in range(grid_y, max_y + 1):
                for neighbor_x in range(min_x, max_x + 1):
                    # On the current row, only cells to the right are new.
                    # All cells in later rows have not yet been processed.
                    if (
                        neighbor_y == grid_y
                        and neighbor_x <= grid_x
                    ):
                        continue

                    neighbor_cell = grid[neighbor_y][neighbor_x]
                    if not neighbor_cell:
                        continue

                    for i in current_cell:
                        for j in neighbor_cell:
                            yield i, j


def candidate_pair_indices(
    grid: Grid,
    cell_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique candidate pairs as NumPy index arrays."""

    pair_i: list[int] = []
    pair_j: list[int] = []

    for i, j in iter_candidate_pairs(grid, cell_radius):
        pair_i.append(i)
        pair_j.append(j)

    return (
        np.asarray(pair_i, dtype=np.intp),
        np.asarray(pair_j, dtype=np.intp),
    )

def lennard_jones_force(p1: Atom, p2: Atom) -> tuple[float, float]:
    """Return the Lennard-Jones force acting on p1 due to p2."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    distance_squared = dx * dx + dy * dy
    cutoff_squared = LJ_CUTOFF * LJ_CUTOFF

    if distance_squared >= cutoff_squared:
        return 0.0, 0.0

    if CLAMP and distance_squared < 1e-12:
        angle = rnd.random() * 2.0 * math.pi
        dx = math.cos(angle)
        dy = math.sin(angle)
        force_distance = 1.0
        direction_x = dx / force_distance
        direction_y = dy / force_distance
    else:
        force_distance = math.sqrt(distance_squared)
        direction_x = dx / force_distance
        direction_y = dy / force_distance

    force_distance_squared = force_distance * force_distance

    sigma_squared_over_r_squared = (
        LJ_SIGMA * LJ_SIGMA / force_distance_squared
    )

    sigma_over_r_6 = sigma_squared_over_r_squared**3
    sigma_over_r_12 = sigma_over_r_6**2

    factor = (
        24.0
        * LJ_EPSILON
        / force_distance_squared
        * (2.0 * sigma_over_r_12 - sigma_over_r_6)
    )

    force_magnitude = factor * force_distance

    return (
        force_magnitude * direction_x,
        force_magnitude * direction_y,
    )

def lennard_jones_potential(p1: Atom, p2: Atom) -> float:
    """Return a cutoff-shifted Lennard-Jones potential energy."""
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    distance_squared = dx * dx + dy * dy

    if distance_squared >= LJ_CUTOFF * LJ_CUTOFF:
        return 0.0

    if distance_squared < 1e-12 and CLAMP:
        return 0.0

    distance = math.sqrt(distance_squared)

    sigma_over_r_6 = (LJ_SIGMA / distance) ** 6
    sigma_over_r_12 = sigma_over_r_6**2

    potential = 4.0 * LJ_EPSILON * (
        sigma_over_r_12 - sigma_over_r_6
    )

    sigma_over_cutoff_6 = (LJ_SIGMA / LJ_CUTOFF) ** 6
    sigma_over_cutoff_12 = sigma_over_cutoff_6**2

    cutoff_potential = 4.0 * LJ_EPSILON * (
        sigma_over_cutoff_12 - sigma_over_cutoff_6
    )

    return potential - cutoff_potential

def lennard_jones_forces_vectorized(
    particles: list[Atom],
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Lennard-Jones forces for candidate pairs in one NumPy batch.

    Returns the filtered pair indices and the x/y force components acting on
    particle i due to particle j.
    """

    if pair_i.size == 0:
        empty_indices = np.empty(0, dtype=np.intp)
        empty_values = np.empty(0, dtype=float)
        return (
            empty_indices,
            empty_indices.copy(),
            empty_values,
            empty_values.copy(),
        )

    positions = np.asarray(
        [(particle.x, particle.y) for particle in particles],
        dtype=float,
    )

    displacement = positions[pair_i] - positions[pair_j]
    distance_squared = np.einsum(
        "ij,ij->i",
        displacement,
        displacement,
    )

    within_cutoff = distance_squared < LJ_CUTOFF * LJ_CUTOFF

    pair_i = pair_i[within_cutoff]
    pair_j = pair_j[within_cutoff]
    displacement = displacement[within_cutoff]
    distance_squared = distance_squared[within_cutoff]

    if pair_i.size == 0:
        empty_values = np.empty(0, dtype=float)
        return pair_i, pair_j, empty_values, empty_values.copy()

    coincident = distance_squared < 1e-12

    if np.any(coincident) and not CLAMP:
        raise ZeroDivisionError(
            "Lennard-Jones force is undefined for coincident particles "
            "when CLAMP is False."
        )

    force_distance_squared = distance_squared.copy()

    if CLAMP and np.any(coincident):
        coincident_indices = np.flatnonzero(coincident)

        for index in coincident_indices:
            angle = rnd.random() * 2.0 * math.pi
            displacement[index, 0] = math.cos(angle)
            displacement[index, 1] = math.sin(angle)

        force_distance_squared[coincident] = 1.0

    sigma_squared_over_r_squared = (
        LJ_SIGMA * LJ_SIGMA / force_distance_squared
    )
    sigma_over_r_6 = sigma_squared_over_r_squared**3
    sigma_over_r_12 = sigma_over_r_6**2

    factor = (
        24.0
        * LJ_EPSILON
        / force_distance_squared
        * (2.0 * sigma_over_r_12 - sigma_over_r_6)
    )

    forces = factor[:, None] * displacement

    return (
        pair_i,
        pair_j,
        forces[:, 0],
        forces[:, 1],
    )

def calculate_accelerations(
    particles: list[Atom],
    grid: Grid,
) -> None:
    """Calculate accelerations, vectorizing only the LJ pair arithmetic."""

    particle_count = len(particles)
    accelerations = np.zeros((particle_count, 2), dtype=float)

    if FORCES:
        search_radius = math.ceil(LJ_CUTOFF / CELL_SIZE)

        pair_i, pair_j = candidate_pair_indices(
            grid,
            search_radius,
        )

        pair_i, pair_j, force_x, force_y = (
            lennard_jones_forces_vectorized(
                particles,
                pair_i,
                pair_j,
            )
        )

        masses = np.fromiter(
            (particle.mass for particle in particles),
            dtype=float,
            count=particle_count,
        )

        np.add.at(
            accelerations[:, 0],
            pair_i,
            force_x / masses[pair_i],
        )
        np.add.at(
            accelerations[:, 1],
            pair_i,
            force_y / masses[pair_i],
        )
        np.add.at(
            accelerations[:, 0],
            pair_j,
            -force_x / masses[pair_j],
        )
        np.add.at(
            accelerations[:, 1],
            pair_j,
            -force_y / masses[pair_j],
        )

    for particle, acceleration in zip(particles, accelerations):
        particle.acc[0] = float(acceleration[0])
        particle.acc[1] = float(acceleration[1])

def speed_to_color(
    speeds: np.ndarray,
) -> np.ndarray:
    ratios = np.clip(
        (speeds - MIN_COLOR_SPEED)
        / (MAX_COLOR_SPEED - MIN_COLOR_SPEED),
        0.0,
        1.0,
    )
    colors = (
        MIN_COLOR
        + ratios[:, None] * (MAX_COLOR - MIN_COLOR)
    )
    return colors.astype(np.uint8)

def velocity_colors(
    particles: list[Atom],
) -> np.ndarray:

    velocities = np.asarray(
        [particle.vel for particle in particles],
        dtype=float,
    )
    speeds = np.linalg.norm(velocities, axis=1)
    return speed_to_color(speeds)

def resolve_hard_disk_collisions(
    particles: list[Atom],
    grid: Grid,
) -> None:
    """Resolve overlaps and perfectly elastic hard-disk collisions."""

    # With equal particle sizes this is the largest possible collision
    # distance. Keeping the calculation explicit makes the required grid
    # neighborhood clear and remains correct if CELL_SIZE changes.
    collision_distance = 2.0 * PARTICLE_SIZE
    cell_radius = math.ceil(collision_distance / CELL_SIZE)

    for i1, i2 in iter_candidate_pairs(grid, cell_radius):
        p1 = particles[i1]
        p2 = particles[i2]

        dx = p1.x - p2.x
        dy = p1.y - p2.y
        distance_squared = dx * dx + dy * dy

        minimum_distance = p1.size + p2.size
        minimum_distance_squared = minimum_distance**2

        if distance_squared >= minimum_distance_squared:
            continue

        if distance_squared < 1e-12:
            angle = rnd.random() * 2.0 * math.pi
            normal_x = math.cos(angle)
            normal_y = math.sin(angle)
            distance = 0.0
        else:
            distance = math.sqrt(distance_squared)
            normal_x = dx / distance
            normal_y = dy / distance

        overlap = minimum_distance - distance

        inverse_mass_1 = 1.0 / p1.mass
        inverse_mass_2 = 1.0 / p2.mass
        inverse_mass_sum = inverse_mass_1 + inverse_mass_2

        correction_1 = overlap * inverse_mass_1 / inverse_mass_sum
        correction_2 = overlap * inverse_mass_2 / inverse_mass_sum

        p1.x += normal_x * correction_1
        p1.y += normal_y * correction_1

        p2.x -= normal_x * correction_2
        p2.y -= normal_y * correction_2

        clamp_particle_position(p1)
        clamp_particle_position(p2)

        relative_velocity_x = p1.vel[0] - p2.vel[0]
        relative_velocity_y = p1.vel[1] - p2.vel[1]

        relative_normal_speed = (
            relative_velocity_x * normal_x
            + relative_velocity_y * normal_y
        )

        if relative_normal_speed >= 0.0:
            continue

        restitution = 1.0

        impulse_magnitude = (
            -(1.0 + restitution)
            * relative_normal_speed
            / inverse_mass_sum
        )

        impulse_x = impulse_magnitude * normal_x
        impulse_y = impulse_magnitude * normal_y

        p1.vel[0] += impulse_x * inverse_mass_1
        p1.vel[1] += impulse_y * inverse_mass_1

        p2.vel[0] -= impulse_x * inverse_mass_2
        p2.vel[1] -= impulse_y * inverse_mass_2


def clamp_particle_position(particle: Atom) -> None:
    particle.x = min(
        max(particle.x, particle.size),
        WIDTH - particle.size,
    )

    particle.y = min(
        max(particle.y, particle.size),
        HEIGHT - particle.size,
    )


def update(
    particles: list[Atom],
    dt: float,
) -> None:
    """Advance the simulation by one physics timestep."""

    if FORCES:
        for particle in particles:
            particle.vel[0] += 0.5 * dt * particle.acc[0]
            particle.vel[1] += 0.5 * dt * particle.acc[1]

    for particle in particles:
        if DAMPING:
            particle.vel[0] *= 0.9999
            particle.vel[1] *= 0.9999

        particle.move(dt)

    moved_grid = None

    if COLLISIONS or FORCES:
        moved_grid = build_grid(particles)

    if COLLISIONS:
        resolve_hard_disk_collisions(
            particles,
            moved_grid,
        )

    if FORCES:
        calculate_accelerations(
            particles,
            moved_grid,
        )

        for particle in particles:
            particle.vel[0] += 0.5 * dt * particle.acc[0]
            particle.vel[1] += 0.5 * dt * particle.acc[1]

    else:
        clear_accelerations(particles)

def calculate_energy(
    particles: list[Atom],
) -> tuple[float, float, float, float]:
    kinetic_energy = sum(
        particle.kinetic_energy()
        for particle in particles
    )

    potential_energy = 0.0
    gravitational_potential_energy = 0.0

    if FORCES:
        grid = build_grid(particles)
        search_radius = math.ceil(LJ_CUTOFF / CELL_SIZE)

        for i1, i2 in iter_candidate_pairs(
            grid,
            search_radius,
        ):
            potential_energy += lennard_jones_potential(
                particles[i1],
                particles[i2],
            )

        # Preserve the original energy convention exactly.
        potential_energy += LJ_EPSILON * ATOM_PAIRS

    total_energy = (
        kinetic_energy
        + potential_energy
        + gravitational_potential_energy
    )

    return (
        kinetic_energy,
        potential_energy,
        gravitational_potential_energy,
        total_energy,
    )


def handle_events(particles: list[Atom]) -> bool:
    global BOUNDARY
    global COLORED_HALVES
    global COLLISIONS
    global FORCES
    global DAMPING
    global GRAVITY
    global OSMOTIC_BOUNDARY

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return False

        elif event.type != pygame.KEYDOWN:
            continue

        elif event.key == pygame.K_ESCAPE:
            return False

        elif event.key == pygame.K_c:
            COLLISIONS = not COLLISIONS
            print(f"COLLISIONS: {COLLISIONS}")

            if COLLISIONS:
                FORCES = False
                clear_accelerations(particles)

        elif event.key == pygame.K_f:
            FORCES = not FORCES
            print(f"FORCES: {FORCES}")

            if FORCES:
                COLLISIONS = False
                initial_grid = build_grid(particles)
                calculate_accelerations(particles, initial_grid)
            else:
                clear_accelerations(particles)

        elif event.key == pygame.K_d:
            DAMPING = not DAMPING
            print(f"DAMPING: {DAMPING}")

        elif event.key == pygame.K_e:
            kinetic, potential, gravitational, total = calculate_energy(
                particles
            )
            print(
                f"--Average Energy by Particle--\n"
                f"K = {kinetic / ATOM_NUMBER:,.0f} | "
                f"U = {potential / ATOM_NUMBER:,.0f} | "
                f"G = {gravitational / ATOM_NUMBER:,.0f} | "
                f"E = {total / ATOM_NUMBER:,.0f}"
            )

            if CLAMP:
                print(
                    "LJ-Forces at small distances are being clamped. "
                    "This violates conservation of energy!"
                )

    return True

def draw_histogram(
    screen: pygame.Surface,
    particles: list[Atom],
) -> None:

    velocities = np.asarray(
        [particle.vel for particle in particles],
        dtype=float,
    )

    speeds = np.linalg.norm(
        velocities,
        axis=1,
    )

    counts, edges = np.histogram(
        speeds,
        bins=BIN_NUMBER,
        range=(0, HIST_V_CUTOFF),
    )

    # Speed at the center of each histogram bin
    bin_speeds = (
        edges[:-1] + edges[1:]
    ) / 2.0

    # Color each bin using the exact same scale as the particles
    colors = speed_to_color(bin_speeds)

    hist_top = HEIGHT+10
    hist_height = HIST_HEIGHT - 20
    hist_bottom = hist_top + hist_height

    pygame.draw.line(
        screen,
        "white",
        (0, hist_bottom),
        (WIDTH, hist_bottom),
        2,
    )

    max_count = 32

    for i, (count, color) in enumerate(zip(counts, colors)):
        bar_height = (
                             min(1,count / max_count)
                     ) * (hist_height - 20)

        x0 = round(i * WIDTH / len(counts))
        x1 = round((i + 1) * WIDTH / len(counts))

        y = hist_bottom - bar_height

        pygame.draw.rect(
            screen,
            color,
            (
                x0,
                round(y),
                x1 - x0,
                round(bar_height),
            ),
        )

def draw(
    screen: pygame.Surface,
    particles: list[Atom],
) -> None:
    screen.fill("black")
    colors = velocity_colors(particles)
    for particle,color in zip(particles, colors):
        pygame.draw.circle(
            screen,
            color=color,
            center=(round(particle.x), round(particle.y)),
            radius=particle.size,
        )
    if HISTOGRAM:
        pygame.draw.line(
            screen,
            "white",
            (0,HEIGHT),
            (WIDTH, HEIGHT),
            width=2
        )
        draw_histogram(screen, particles)

    pygame.display.flip()


def create_particles() -> list[Atom]:
    particles: list[Atom] = []

    particle_grid_size = round(math.sqrt(ATOM_NUMBER))
    initial_cell_size = WIDTH / particle_grid_size

    for grid_x in range(particle_grid_size):
        for grid_y in range(particle_grid_size):
            center_x = (grid_x + 0.5) * initial_cell_size
            center_y = (grid_y + 0.5) * initial_cell_size

            jitter_limit = max(
                0.0,
                initial_cell_size / 2.0
                - PARTICLE_SIZE
                - LJ_SIGMA * 0.15,
            )

            x = center_x + rnd.uniform(-jitter_limit, jitter_limit)
            y = center_y + rnd.uniform(-jitter_limit, jitter_limit)

            velocity = [0,0]

            particles.append(
                Atom(
                    x=x,
                    y=y,
                    vel=velocity,
                    size=PARTICLE_SIZE,
                    type=rnd.randint(0, 1),
                )
            )

    particles[0].color="red"
    particles[0].vel = [1200,1200]

    return particles


def main() -> None:
    pygame.init()

    screen = pygame.display.set_mode((WIDTH, HEIGHT+HIST_HEIGHT if HISTOGRAM else HEIGHT))

    clock = pygame.time.Clock()
    particles = create_particles()

    if FORCES:
        initial_grid = build_grid(particles)
        calculate_accelerations(
            particles,
            initial_grid,
        )

    print(INSTRUCTIONS)

    running = True

    physics_dt = TIME_SCALE / (FPS * STEPS_PER_FRAME)

    #START
    for t in range(5, 0, -1):
        start_time = time.time()
        while time.time() - start_time < 1.0:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            pygame.display.set_caption(f"Start in {t} seconds")
            draw(
                screen,
                particles,
            )
            clock.tick(FPS)

    pygame.display.set_caption("Boltzman's Billiard")
    while running:
        clock.tick(FPS)

        running = handle_events(particles)

        if not running:
            break

        print(clock.get_fps() // 1)
        for _ in range(STEPS_PER_FRAME):
            update(particles,physics_dt,)
        draw(
            screen,
            particles,
        )

    pygame.quit()


if __name__ == "__main__":
    main()
