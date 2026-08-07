import math
import matplotlib

# Must be set before importing pyplot.
matplotlib.use("MacOSX")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.integrate import quad
from scipy.optimize import brentq

plt.style.use("dark_background")

#Isn't the formatting beautiful? Thanks ChatGPT
# ============================================================
# SETTINGS
# ============================================================
BALL = True
DISPLAY_FORCE = True

ANIMATION_INTERVAL_MS = 10

# Physical simulation time advanced during each displayed frame.
SIMULATED_TIME_PER_FRAME = 20e-15  # 20 fs

# Number of velocity-Verlet steps per displayed frame.
PHYSICS_STEPS_PER_FRAME = 80


# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

ATOMIC_MASS_UNIT = 1.660_539_066_60e-27  # kg
BOLTZMANN_CONSTANT = 1.380_649e-23       # J/K

# ============================================================
# ARGON PARAMETERS
# ============================================================

ARGON_ATOMIC_MASS_U = 39.948

ARGON_PARTICLE_MASS = (
    ARGON_ATOMIC_MASS_U * ATOMIC_MASS_UNIT
)

# One argon atom moves while the other is fixed.
# Therefore, the reduced moving mass is one argon mass.
r_PARTICLE_MASS = 1.0

LJ_SIGMA = 3.405e-10  # m

LJ_EPSILON_OVER_K = 119.8  # K

LJ_EPSILON = (
    LJ_EPSILON_OVER_K * BOLTZMANN_CONSTANT
)  # J

TIME_SCALE=2 #sets up speed
# Lennard-Jones time unit:
#
# tau = sigma * sqrt(m / epsilon)
#
# Since one atom is fixed, m is the full argon atomic mass.
LJ_TIME_UNIT = LJ_SIGMA * math.sqrt(
    ARGON_PARTICLE_MASS / LJ_EPSILON
)/TIME_SCALE


# ============================================================
# REDUCED PARAMETERS
# ============================================================

r_LJ_MIN_DISTANCE = 0.7

# Minimum of the 12-6 Lennard-Jones potential.
r_R_WELL = 2.0 ** (1.0 / 6.0)

INITIAL_TEMPERATURE = 90.0  # K

# Excitation energy divided by the reference LJ epsilon.
r_INITIAL_EXCITATION_ENERGY = (
    INITIAL_TEMPERATURE / LJ_EPSILON_OVER_K
)


# ============================================================
# UNIT CONVERSIONS
# ============================================================

def reduce_r(r, invert=False):
    """Convert between metres and reduced distance r/sigma."""
    if invert:
        return r * LJ_SIGMA

    return r / LJ_SIGMA


def reduce_E(energy, invert=False):
    """Convert between joules and reduced energy E/epsilon."""
    if invert:
        return energy * LJ_EPSILON

    return energy / LJ_EPSILON


def reduce_F(force, invert=False):
    """Convert between newtons and reduced force F*sigma/epsilon."""
    if invert:
        return force * LJ_EPSILON / LJ_SIGMA

    return force * LJ_SIGMA / LJ_EPSILON


def reduce_stiffness(stiffness, invert=False):
    """Convert between N/m and reduced stiffness k*sigma²/epsilon."""
    if invert:
        return stiffness * LJ_EPSILON / LJ_SIGMA**2

    return stiffness * LJ_SIGMA**2 / LJ_EPSILON


def reduce_time(time, invert=False):
    """Convert between seconds and reduced LJ time t/tau."""
    if invert:
        return time * LJ_TIME_UNIT

    return time / LJ_TIME_UNIT


# ============================================================
# LENNARD-JONES FUNCTIONS
# ============================================================

def r_lennard_jones_potential(r_r, r_epsilon):
    """
    Reduced Lennard-Jones potential.

    U* = 4 epsilon* (r*^-12 - r*^-6)
    """
    return 4.0 * r_epsilon * (
        r_r**-12 - r_r**-6
    )


def r_lennard_jones_force(r_r, r_epsilon):
    """
    Reduced radial force.

    F* = -dU*/dr*
       = 24 epsilon* (2 r*^-13 - r*^-7)
    """
    return 24.0 * r_epsilon * (
        2.0 * r_r**-13 - r_r**-7
    )


def r_lennard_jones_stiffness(r_r, r_epsilon):
    """
    Reduced local stiffness.

    k* = -dF*/dr*
       = d²U*/dr*²
    """
    return 24.0 * r_epsilon * (
        26.0 * r_r**-14
        - 7.0 * r_r**-8
    )


FUNCTIONS_TO_PLOT = (
    r_lennard_jones_potential,
    r_lennard_jones_force,
)

PLOT_NAMES = (
    "LJ potential",
    "LJ force",
)


# ============================================================
# PARTICLE
# ============================================================

class Ball:
    """One-dimensional particle moving in a Lennard-Jones potential."""

    def __init__(
        self,
        mass: float = r_PARTICLE_MASS,
        radius: float = 0.08,
    ) -> None:
        self.r_mass = mass
        self.radius = radius

        self.r_epsilon = 1.0
        self.r_r = r_R_WELL
        self.r_velocity = 0.0
        self.r_acceleration = 0.0

        self.r_excitation_energy = (
            r_INITIAL_EXCITATION_ENERGY
        )

        self.r_total_energy = (
            self.r_excitation_energy - self.r_epsilon
        )

        self.is_bound = True

    def reset(
        self,
        r_excitation_energy: float,
        r_epsilon: float,
    ) -> None:
        """
        Reset the particle at the inner turning point.

        Starting at a turning point with zero velocity makes the
        mechanical energy equal to the selected total energy.
        """
        self.r_excitation_energy = r_excitation_energy
        self.r_epsilon = r_epsilon

        self.r_total_energy = (
            r_excitation_energy - r_epsilon
        )

        self.is_bound = (
            0.0 < r_excitation_energy < r_epsilon
        )

        if not self.is_bound:
            self.r_r = r_R_WELL
            self.r_velocity = 0.0
            self.r_acceleration = 0.0
            return

        r_inner, _ = r_find_turning_points(
            r_excitation_energy,
            r_epsilon,
        )

        self.r_r = r_inner
        self.r_velocity = 0.0

        self.r_acceleration = (
            r_lennard_jones_force(
                self.r_r,
                self.r_epsilon,
            )
            / self.r_mass
        )

    def update(self, r_dt: float) -> None:
        """Advance the particle using velocity-Verlet integration."""
        if not self.is_bound:
            return

        # First half-kick.
        self.r_velocity += (
            0.5 * r_dt * self.r_acceleration
        )

        # Drift.
        self.r_r += r_dt * self.r_velocity

        # Guard against the singularity at r = 0.
        if self.r_r <= 0.0:
            self.r_r = np.finfo(float).eps
            self.r_velocity = abs(self.r_velocity)

        # Acceleration at the new position.
        new_r_acceleration = (
            r_lennard_jones_force(
                self.r_r,
                self.r_epsilon,
            )
            / self.r_mass
        )

        # Second half-kick.
        self.r_velocity += (
            0.5 * r_dt * new_r_acceleration
        )

        self.r_acceleration = new_r_acceleration

    def r_kinetic_energy(self) -> float:
        """Return the reduced kinetic energy."""
        return (
            0.5
            * self.r_mass
            * self.r_velocity**2
        )

    def r_measured_total_energy(self) -> float:
        """Return the measured reduced mechanical energy."""
        return (
            self.r_kinetic_energy()
            + r_lennard_jones_potential(
                self.r_r,
                self.r_epsilon,
            )
        )


# ============================================================
# TURNING POINTS AND TIME AVERAGES
# ============================================================

def r_find_turning_points(
    r_excitation_energy: float,
    r_epsilon: float,
) -> tuple[float, float]:
    """Return the inner and outer turning points of a bound trajectory."""
    r_total_energy = (
        r_excitation_energy - r_epsilon
    )

    if not (-r_epsilon < r_total_energy < 0.0):
        raise ValueError(
            "Bound motion requires "
            "0 < excitation energy < epsilon."
        )

    def turning_point_equation(r_r: float) -> float:
        return (
            r_total_energy
            - r_lennard_jones_potential(
                r_r,
                r_epsilon,
            )
        )

    inner_turning_point = brentq(
        turning_point_equation,
        r_LJ_MIN_DISTANCE,
        r_R_WELL,
    )

    r_upper_bound = 3.0

    while turning_point_equation(r_upper_bound) > 0.0:
        r_upper_bound *= 2.0

    outer_turning_point = brentq(
        turning_point_equation,
        r_R_WELL,
        r_upper_bound,
    )

    return inner_turning_point, outer_turning_point


def find_avg_r_pos(
    r_excitation_energy: float,
    r_epsilon: float,
) -> float | None:
    """Return the time-averaged reduced separation."""
    r_total_energy = (
        r_excitation_energy - r_epsilon
    )

    if not (-r_epsilon < r_total_energy < 0.0):
        return None

    r_inner, r_outer = r_find_turning_points(
        r_excitation_energy,
        r_epsilon,
    )

    def time_weight(r_r: float) -> float:
        r_kinetic_energy = (
            r_total_energy
            - r_lennard_jones_potential(
                r_r,
                r_epsilon,
            )
        )

        # Small negative values can occur from roundoff directly
        # beside a turning point.
        r_kinetic_energy = max(
            r_kinetic_energy,
            np.finfo(float).tiny,
        )

        return 1.0 / math.sqrt(r_kinetic_energy)

    numerator, _ = quad(
        lambda r_r: r_r * time_weight(r_r),
        r_inner,
        r_outer,
        points=[r_inner, r_outer],
        limit=200,
    )

    denominator, _ = quad(
        time_weight,
        r_inner,
        r_outer,
        points=[r_inner, r_outer],
        limit=200,
    )

    if denominator == 0.0:
        return None

    r_average_position = numerator / denominator

    if not (
        r_inner
        < r_average_position
        < r_outer
    ):
        return None

    return r_average_position


def find_avg_r_stiffness(
    r_excitation_energy: float,
    r_epsilon: float,
) -> float | None:
    """Return the time-averaged reduced local stiffness."""
    r_total_energy = (
        r_excitation_energy - r_epsilon
    )

    if not (-r_epsilon < r_total_energy < 0.0):
        return None

    r_inner, r_outer = r_find_turning_points(
        r_excitation_energy,
        r_epsilon,
    )

    def r_available_energy(r_r: float) -> float:
        return (
            r_total_energy
            - r_lennard_jones_potential(
                r_r,
                r_epsilon,
            )
        )

    def time_weight(r_r: float) -> float:
        r_kinetic_energy = r_available_energy(r_r)

        r_kinetic_energy = max(
            r_kinetic_energy,
            np.finfo(float).tiny,
        )

        return 1.0 / math.sqrt(r_kinetic_energy)

    numerator, _ = quad(
        lambda r_r: (
            r_lennard_jones_stiffness(
                r_r,
                r_epsilon,
            )
            * time_weight(r_r)
        ),
        r_inner,
        r_outer,
        points=[r_inner, r_outer],
        limit=200,
    )

    denominator, _ = quad(
        time_weight,
        r_inner,
        r_outer,
        points=[r_inner, r_outer],
        limit=200,
    )

    if denominator == 0.0:
        return None

    return numerator / denominator


# ============================================================
# PLOT STYLING
# ============================================================

def style_axes(axes) -> None:
    """Apply dark-background styling to all plot axes."""
    for ax in axes:
        ax.set_facecolor("black")

        ax.tick_params(
            axis="both",
            colors="white",
        )

        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

        for spine in ax.spines.values():
            spine.set_color("white")

        ax.grid(
            visible=True,
            linestyle=":",
            alpha=0.25,
        )


def style_slider(
    slider: Slider,
    slider_ax: plt.Axes,
) -> None:
    """Apply dark-background styling to a slider."""
    slider_ax.set_facecolor("black")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    if hasattr(slider, "track"):
        slider.track.set_facecolor("black")
        slider.track.set_edgecolor("white")

    if hasattr(slider, "poly"):
        slider.poly.set_edgecolor("white")

    if hasattr(slider, "vline"):
        slider.vline.set_alpha(0.5)
        slider.vline.set_color("darkred")

    for spine in slider_ax.spines.values():
        spine.set_color("white")


# ============================================================
# ANIMATION
# ============================================================

def animate_potential(
    r_start: float = 0.9,
    r_end: float = 3.0,
):
    r_r_values = np.linspace(
        r_start,
        r_end,
        800,
    )

    rows = 2 if DISPLAY_FORCE else 1

    fig, axes = plt.subplots(
        nrows=rows,
        ncols=1,
        figsize=(10, 8.4),
        facecolor="black",
        sharex=True,
    )

    # Ensure axes is always a one-dimensional array.
    axes = np.atleast_1d(axes)

    fig.canvas.manager.set_window_title(
        "Lennard-Jones Simulation"
    )

    fig.subplots_adjust(
        top=0.95,
        bottom=0.23,
    )

    axes[0].set_ylabel("Potential energy U (J)")
    axes[0].set_title(
        "Argon Lennard-Jones Potential"
    )

    if DISPLAY_FORCE:
        axes[1].set_ylabel("Radial force F (N)")
        axes[1].set_title(
            "Argon Lennard-Jones Force"
        )

    style_axes(axes)

    for ax in axes:
        ax.set_xlabel(
            "Reduced separation r / σ"
        )

        ax.set_xlim(
            r_r_values.min(),
            r_r_values.max(),
        )

    axes[0].set_ylim(
        -1.7 * LJ_EPSILON,
        0.5 * LJ_EPSILON,
    )

    force_scale = LJ_EPSILON / LJ_SIGMA

    if DISPLAY_FORCE:
        axes[1].set_ylim(
            -5.0 * force_scale,
            25.0 * force_scale,
        )

    # Keep artist lists local so repeated calls do not reuse
    # artists from previous figures.
    curves = []
    avg_pos_lines = []
    inner_tps = []
    outer_tps = []
    balls = []
    texts = []

    # --------------------------------------------------------
    # Sliders
    # --------------------------------------------------------

    temperature_slider_ax = fig.add_axes(
        (0.15, 0.025, 0.70, 0.055),
        facecolor="black",
    )

    temperature_slider = Slider(
        ax=temperature_slider_ax,
        label="Temperature (K)",
        valmin=0.1,
        valmax=1.5 * LJ_EPSILON_OVER_K,
        valinit=INITIAL_TEMPERATURE,
        color="salmon",
    )

    style_slider(
        temperature_slider,
        temperature_slider_ax,
    )

    epsilon_slider_ax = fig.add_axes(
        (0.15, 0.105, 0.70, 0.055),
        facecolor="black",
    )

    epsilon_slider = Slider(
        ax=epsilon_slider_ax,
        label="Epsilon (J)",
        valmin=0.4 * LJ_EPSILON,
        valmax=1.6 * LJ_EPSILON,
        valinit=LJ_EPSILON,
        color="lightblue",
    )

    style_slider(
        epsilon_slider,
        epsilon_slider_ax,
    )

    # --------------------------------------------------------
    # Static artists
    # --------------------------------------------------------

    energy_line = axes[0].axhline(
        y=0.0,
        linestyle="--",
        linewidth=2,
        color="salmon",
        label="Selected total energy",
    )

    if DISPLAY_FORCE:
        axes[1].axhline(
            y=0.0,
            linestyle="-",
            linewidth=1,
            color="white",
            alpha=0.75,
        )

        texts.append(
            axes[1].text(
                0.992,
                0.02,
                "",
                transform=axes[1].transAxes,
                ha="right",
                va="bottom",
                color="white",
                fontsize=11,
            )
        )

    functions_to_plot = (
        FUNCTIONS_TO_PLOT
        if DISPLAY_FORCE
        else FUNCTIONS_TO_PLOT[:1]
    )

    plot_names = (
        PLOT_NAMES
        if DISPLAY_FORCE
        else PLOT_NAMES[:1]
    )

    for i, ax in enumerate(axes):
        function = functions_to_plot[i]

        curve, = ax.plot(
            [],
            [],
            color="lightblue",
            linewidth=2,
            label=plot_names[i],
        )

        curves.append(curve)

        average_position_line = ax.axvline(
            x=r_R_WELL,
            linestyle=":",
            linewidth=2,
            color="red",
            label="Time-averaged position",
            alpha=0.45,
        )

        avg_pos_lines.append(
            average_position_line
        )

        inner_turning_line = ax.axvline(
            x=r_R_WELL,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
            label="Turning points",
        )

        inner_tps.append(inner_turning_line)

        outer_turning_line = ax.axvline(
            x=r_R_WELL,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
        )

        outer_tps.append(outer_turning_line)

        ball_marker, = ax.plot(
            [],
            [],
            marker="o",
            markersize=14,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.2,
            linestyle="None",
            label=(
                "Oscillating particle"
                if BALL
                else "_nolegend_"
            ),
            zorder=10,
            visible=BALL,
        )

        balls.append(ball_marker)

        legend = ax.legend(
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
            loc=(
                "lower right"
                if i == 0
                else "upper right"
            ),
            frameon=True,
        )

        legend.get_frame().set_alpha(0.8)

    parameters_changed = BALL
    ball = Ball() if BALL else None

    # Reduced integration step.
    physics_dt_reduced = reduce_time(
        SIMULATED_TIME_PER_FRAME
        / PHYSICS_STEPS_PER_FRAME
    )

    def current_parameters() -> tuple[float, float]:
        """Return reduced excitation energy and reduced epsilon."""
        r_excitation_energy = reduce_E(
            temperature_slider.val
            * BOLTZMANN_CONSTANT
        )

        r_epsilon = reduce_E(
            epsilon_slider.val
        )

        return r_excitation_energy, r_epsilon

    def reset_ball() -> None:
        nonlocal parameters_changed

        if ball is None:
            parameters_changed = False
            return

        r_excitation_energy, r_epsilon = (
            current_parameters()
        )

        ball.reset(
            r_excitation_energy=r_excitation_energy,
            r_epsilon=r_epsilon,
        )

        parameters_changed = False

    def mark_parameters_changed(_value: float) -> None:
        nonlocal parameters_changed

        if BALL:
            parameters_changed = True

    temperature_slider.on_changed(
        mark_parameters_changed
    )

    epsilon_slider.on_changed(
        mark_parameters_changed
    )

    def update_static_elements(
        r_excitation_energy: float,
        r_epsilon: float,
    ) -> None:
        r_total_energy = (
            r_excitation_energy - r_epsilon
        )

        total_energy_si = reduce_E(
            r_total_energy,
            invert=True,
        )

        energy_line.set_ydata(
            [total_energy_si, total_energy_si]
        )

        is_bound = (
            0.0
            < r_excitation_energy
            < r_epsilon
        )

        if is_bound:
            r_inner, r_outer = (
                r_find_turning_points(
                    r_excitation_energy,
                    r_epsilon,
                )
            )

            r_average_position = find_avg_r_pos(
                r_excitation_energy,
                r_epsilon,
            )

            r_avg_stiffness = (
                find_avg_r_stiffness(
                    r_excitation_energy,
                    r_epsilon,
                )
            )
        else:
            r_inner = None
            r_outer = None
            r_average_position = None
            r_avg_stiffness = None

        for text in texts:
            if r_avg_stiffness is None:
                text.set_text(
                    "Average stiffness: unbound"
                )
            else:
                stiffness_si = reduce_stiffness(
                    r_avg_stiffness,
                    invert=True,
                )

                text.set_text(
                    "Average stiffness: "
                    f"{stiffness_si:.3f} N/m"
                )

        for i, curve in enumerate(curves):
            r_y_values = functions_to_plot[i](
                r_r_values,
                r_epsilon,
            )

            if i == 0:
                y_values = reduce_E(
                    r_y_values,
                    invert=True,
                )
            else:
                y_values = reduce_F(
                    r_y_values,
                    invert=True,
                )

            curve.set_data(
                r_r_values,
                y_values,
            )

            if r_inner is None:
                inner_tps[i].set_xdata([])
                inner_tps[i].set_visible(False)
            else:
                inner_tps[i].set_xdata(
                    [r_inner, r_inner]
                )
                inner_tps[i].set_visible(True)

            if r_outer is None:
                outer_tps[i].set_xdata([])
                outer_tps[i].set_visible(False)
            else:
                outer_tps[i].set_xdata(
                    [r_outer, r_outer]
                )
                outer_tps[i].set_visible(True)

            if r_average_position is None:
                avg_pos_lines[i].set_xdata([])
                avg_pos_lines[i].set_visible(False)
            else:
                avg_pos_lines[i].set_xdata(
                    [
                        r_average_position,
                        r_average_position,
                    ]
                )
                avg_pos_lines[i].set_visible(True)

    def update_ball_markers() -> None:
        if ball is None or not ball.is_bound:
            for ball_marker in balls:
                ball_marker.set_data([], [])
                ball_marker.set_visible(False)

            return

        for i, ball_marker in enumerate(balls):
            r_ball_y = functions_to_plot[i](
                ball.r_r,
                ball.r_epsilon,
            )

            if i == 0:
                ball_y = reduce_E(
                    r_ball_y,
                    invert=True,
                )
            else:
                ball_y = reduce_F(
                    r_ball_y,
                    invert=True,
                )

            ball_marker.set_data(
                [ball.r_r],
                [ball_y],
            )

            ball_marker.set_visible(True)

    def all_artists():
        return (
            curves
            + [energy_line]
            + avg_pos_lines
            + inner_tps
            + outer_tps
            + balls
            + texts
        )

    def init():
        if ball is not None:
            reset_ball()

        r_excitation_energy, r_epsilon = (
            current_parameters()
        )

        update_static_elements(
            r_excitation_energy,
            r_epsilon,
        )

        update_ball_markers()

        return all_artists()

    def update(_frame):
        nonlocal parameters_changed

        r_excitation_energy, r_epsilon = (
            current_parameters()
        )

        if parameters_changed and ball is not None:
            reset_ball()

            r_avg_stiffness = (
                find_avg_r_stiffness(
                    r_excitation_energy,
                    r_epsilon,
                )
            )

            if r_avg_stiffness is None:
                print("Average stiffness: unbound")
            else:
                stiffness_si = reduce_stiffness(
                    r_avg_stiffness,
                    invert=True,
                )

                print(
                    "Average stiffness: "
                    f"{stiffness_si:.3f} N/m"
                )

        update_static_elements(
            r_excitation_energy,
            r_epsilon,
        )

        if ball is not None and ball.is_bound:
            for _ in range(
                PHYSICS_STEPS_PER_FRAME
            ):
                ball.update(
                    physics_dt_reduced
                )

        update_ball_markers()

        return all_artists()

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=ANIMATION_INTERVAL_MS,
        cache_frame_data=False,
        blit=False,
    )

    # Keep slider references alive.
    sliders = (
        epsilon_slider,
        temperature_slider,
    )

    return fig, axes, ani, sliders


if __name__ == "__main__":
    fig, axes, ani, sliders = animate_potential()

    print(
        f"LJ time unit: "
        f"{LJ_TIME_UNIT:.6e} s"
    )

    print(
        f"Reduced physics timestep: "
        f"{reduce_time(SIMULATED_TIME_PER_FRAME / PHYSICS_STEPS_PER_FRAME):.6e}"
    )

    plt.show(block=True)

    print(plt.get_backend())