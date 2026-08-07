import math

import matplotlib
matplotlib.use("MacOSX")#absolutely needed

import matplotlib.animation as animation
import matplotlib.pyplot as plt#
import numpy as np
from matplotlib.widgets import Slider
from scipy.integrate import quad#integrating stuff
from scipy.optimize import brentq#root detection


plt.style.use("dark_background")#doesn't change anything I think

## SETTINGS ## (as similar to the other simulation as possible
BALL = False#Bool to determine if Balll is displayed

PARTICLE_SIZE = 5 #Relevant for LJ_Sigma
PARTICLE_MASS = 1.0

INITIAL_MAX_SPEED = 150 #Relevent for starting excitation energy
AVG_ENERGY = INITIAL_MAX_SPEED**2 / 3 * PARTICLE_MASS

TARGET_RATIO = 0.1 #indirect way to set up epsilon

# Lennard-Jones parameters.
LJ_SIGMA = 2.0 * PARTICLE_SIZE #position of zero potential (not the well)
LJ_EPSILON = AVG_ENERGY / TARGET_RATIO #depth of well
LJ_CUTOFF = 3.0 * LJ_SIGMA #force cutoff. Should be equal to the max x value. unused
LJ_MIN_DISTANCE = 0.7 * LJ_SIGMA #minimum distance. unused

R_WELL = 2.0 ** (1.0 / 6.0) * LJ_SIGMA#position of well

# Animation controls.
ANIMATION_INTERVAL_MS = 10 #interval between frames in ms (10 is 100 fps)
PHYSICS_STEPS_PER_FRAME = 80 #computations per step. Higher is good the more the particle aproaches 0
TIME_SCALE = 0.04 #scaling factor for speed
DISPLAY_FORCE = True #automatically displays stiffness

#potential is zero at LJ_SIGMA
def lennard_jones_potential(
    r: float | np.ndarray,
    epsilon: float,
) -> float | np.ndarray:
    """Return the unshifted Lennard-Jones potential."""
    sigma_over_r_6 = (LJ_SIGMA / r) ** 6

    return 4.0 * epsilon * (
        sigma_over_r_6**2 - sigma_over_r_6
    )

#Force is zero at R_WELL
def lennard_jones_force(
    r: float,
    epsilon: float,
) -> float:
    """
    Return the signed radial Lennard-Jones force.

    Positive force increases r.
    Negative force decreases r.
    """
    sigma_over_r_6 = (LJ_SIGMA / r) ** 6

    return (
        24.0
        * epsilon
        / r
        * (
            2.0 * sigma_over_r_6**2
            - sigma_over_r_6
        )
    )

def lennard_jones_stiffness(
    r: float, epsilon: float ):
    #stiffness is the negative derivative of LJ_Force
    return 4*epsilon*(12*13*(LJ_SIGMA**12 / r**14)-6*7*(LJ_SIGMA**6/r**8))

#lists to access and process duplicate variables
curves = []
f_to_plot = (lennard_jones_potential, lennard_jones_force)
name_to_plot = ("LJ potential", "LJ force")
avg_pos_lines = []
inner_tps = []
outer_tps = []
balls = []
texts = [] #currently only displays avg stiffness

class Ball:
    """One-dimensional classical particle moving in the LJ potential."""

    def __init__(
        self,
        mass: float = PARTICLE_MASS,
        radius: float = 0.08,
    ) -> None:
        self.mass = mass
        self.radius = radius

        self.r = R_WELL
        self.velocity = 0.0
        self.acceleration = 0.0

        self.epsilon = LJ_EPSILON
        self.excitation_energy = 0.75 * LJ_EPSILON
        self.total_energy = self.excitation_energy - self.epsilon

        self.is_bound = True

    def reset(
        self,
        excitation_energy: float,
        epsilon: float,
    ) -> None:
        """
        Reset the ball at the inner turning point.

        Starting at a turning point with zero velocity ensures that its
        mechanical energy equals the selected total energy.
        """
        self.epsilon = epsilon
        self.excitation_energy = excitation_energy
        self.total_energy = excitation_energy - epsilon

        self.is_bound = 0.0 < excitation_energy < epsilon

        if not self.is_bound:
            self.r = R_WELL
            self.velocity = 0.0
            self.acceleration = 0.0
            return

        inner, _ = find_turning_points(
            excitation_energy,
            epsilon,
        )

        self.r = inner
        self.velocity = 0.0
        self.acceleration = (
            lennard_jones_force(self.r, epsilon)
            / self.mass
        )

    def update(
        self,
        dt: float,
    ) -> None:
        """Advance the ball using velocity Verlet integration."""
        if not self.is_bound:
            return

        # First half-kick.
        self.velocity += 0.5 * dt * self.acceleration

        # Drift.
        self.r += dt * self.velocity

        # Prevent an invalid or singular position.
        if self.r <= 0.0:
            self.r = np.finfo(float).eps
            self.velocity = abs(self.velocity)

        # Acceleration at the new position.
        new_acceleration = (
            lennard_jones_force(self.r, self.epsilon)
            / self.mass
        )

        # Second half-kick.
        self.velocity += 0.5 * dt * new_acceleration
        self.acceleration = new_acceleration

    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * self.velocity**2

    def measured_total_energy(self) -> float:
        return (
            self.kinetic_energy()
            + lennard_jones_potential(
                self.r,
                self.epsilon,
            )
        )


def find_turning_points(
    excitation_energy: float,
    epsilon: float,
) -> tuple[float, float]:
    """Return the inner and outer turning points of a bound trajectory."""
    total_energy = excitation_energy - epsilon

    if not (-epsilon < total_energy < 0.0):
        raise ValueError(
            "Bound motion requires 0 < excitation energy < epsilon."
        )

    def turning_point_equation(r: float) -> float:
        return (
            total_energy
            - lennard_jones_potential(r, epsilon)
        )

    inner_turning_point = brentq(
        turning_point_equation,
        LJ_MIN_DISTANCE,
        R_WELL,
    )

    upper_bound = 3.0 * LJ_SIGMA

    while turning_point_equation(upper_bound) > 0.0:
        upper_bound *= 2.0

    outer_turning_point = brentq(
        turning_point_equation,
        R_WELL,
        upper_bound,
    )

    return inner_turning_point, outer_turning_point


def find_avg_pos(
    excitation_energy: float,
    epsilon: float,
) -> float | None:
    """
    Return the classical time-averaged separation in particle-size units.
    """
    total_energy = excitation_energy - epsilon

    if not (-epsilon < total_energy < 0.0):
        return None

    inner, outer = find_turning_points(
        excitation_energy,
        epsilon,
    )

    def available_energy(r: float) -> float:
        return (
            total_energy
            - lennard_jones_potential(r, epsilon)
        )

    def time_weight(r: float) -> float:
        kinetic_energy = available_energy(r)

        return 1.0 / math.sqrt(
            max(
                kinetic_energy,
                np.finfo(float).tiny,
            )
        )

    numerator, _ = quad(
        lambda r: r * time_weight(r),
        inner,
        outer,
        points=[inner, outer],
        limit=200,
    )

    denominator, _ = quad(
        time_weight,
        inner,
        outer,
        points=[inner, outer],
        limit=200,
    )

    return numerator / denominator / PARTICLE_SIZE

#Finding the E-Module (Young's Modulus) for oscillating particles
def find_avg_stiffness(
    excitation_energy: float,
    epsilon: float,
) -> float | None:
    total_energy = excitation_energy - epsilon
    if not (-epsilon < total_energy < 0.0):
        return None

    inner, outer = find_turning_points(
        excitation_energy,
        epsilon,
    )
    #computes available kinetic energy
    def available_energy(r: float) -> float:
        return (
            total_energy
            - lennard_jones_potential(r, epsilon)
        )
    def time_weight(r: float) -> float:
        kinetic_energy = available_energy(r)
        return 1.0 / math.sqrt(
            max(
                kinetic_energy,
                np.finfo(float).tiny,
            )
        )

    numerator, _ = quad(
        lambda r: lennard_jones_stiffness(r, epsilon) * time_weight(r),
        inner,
        outer,
        points=[inner, outer],
        limit=200,
    )
    denominator, _ = quad(
        time_weight,
        inner,
        outer,
        points=[inner, outer],
        limit=200,
    )
    return numerator / denominator / PARTICLE_SIZE

def style_axes(axes) -> None:
    """Apply the black-background appearance to a plot axis."""
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
    """Apply the black-background appearance to a slider."""
    slider_ax.set_facecolor("black")
    slider.label.set_color("white")
    slider.valtext.set_color("white")

    # Slider internals vary between Matplotlib versions.
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


def animate_potential(
    start: float = 0.9 * LJ_SIGMA,
    end: float = 3.0 * LJ_SIGMA,
):
    r_values = np.linspace(start, end, 800)
    displayed_r_values = r_values / PARTICLE_SIZE

    rows = 2 if DISPLAY_FORCE else 1
    fig, axes = plt.subplots(
        nrows=rows, ncols=1,
        figsize=(10, 8.4),
        facecolor="black",
        sharex=True,
    )
    fig.canvas.manager.set_window_title("Lennard-Jones Simulation")
    fig.subplots_adjust(top=0.95,bottom=0.23)
    if not DISPLAY_FORCE:
        axes = np.array(axes)
    else: #name the second plot if existent
        axes[1].set_ylabel("Force")
        axes[1].set_title("Lennard-Jones Force")
    style_axes(axes)
    #name the first plot
    axes[0].set_ylabel("Energy")
    axes[0].set_title("Lennard-Jones Potential")

    for i,ax in enumerate(axes):
        ax.set_xlabel("Separation r / particle radius")
        ax.set_xlim(
            displayed_r_values.min(),
            displayed_r_values.max(),
        )
    axes[0].set_ylim(
            -1.7 * LJ_EPSILON,
            0.5 * LJ_EPSILON
        )
    force_values = lennard_jones_force(r_values, LJ_EPSILON)
    axes[1].set_ylim(-0.5 * LJ_EPSILON,1.1 * LJ_EPSILON)


    # Energy slider.
    energy_slider_ax = fig.add_axes(
        (0.15, 0.025, 0.70, 0.055),
        facecolor="black",
    )

    energy_slider = Slider(
        ax=energy_slider_ax,
        label="Temperature",
        valmin=0.001 * LJ_EPSILON,
        valmax=1.5 * LJ_EPSILON,
        valinit=0.75 * LJ_EPSILON,
        color="salmon",
    )

    style_slider(
        energy_slider,
        energy_slider_ax,
    )

    # Epsilon slider.
    epsilon_slider_ax = fig.add_axes(
        (0.15, 0.105, 0.70, 0.055),
        facecolor="black",
    )

    epsilon_slider = Slider(
        ax=epsilon_slider_ax,
        label="Epsilon",
        valmin=0.4 * LJ_EPSILON,
        valmax=1.6 * LJ_EPSILON,
        valinit=LJ_EPSILON,
        color="lightblue",
    )

    style_slider(
        epsilon_slider,
        epsilon_slider_ax,
    )
    # LEAVE THAT OUTSIDE THE LOOP
    energy_line = axes[0].axhline(
        y=0.0,
        linestyle="--",
        linewidth=2,
        color="salmon",
        label="Selected total energy",
    )
    #displays x-axis
    _ = axes[1].axhline(
        y=0,
        linestyle="-",
        linewidth=1,
        color="white",
        alpha=0.75,
    )
    #stiffness display
    if DISPLAY_FORCE:
        texts.append(axes[1].text(
            0.992, 0.02,
            "",
            transform=axes[1].transAxes,#prior coordinates are relative the the top left corner of axes
            ha="right",
            va="bottom",
            color="white",
            fontsize=11,
            #bbox=dict(facecolor="black",edgecolor="white",alpha=0.7,),
        ))

    for i,ax in enumerate(axes):
        (curve,) = ax.plot(
            [],
            [],
            color="lightblue",
            linewidth=2,
            label=name_to_plot[i],
        )
        curves.append(curve)

        average_position_line = ax.axvline(
            x=R_WELL / PARTICLE_SIZE,
            linestyle=":",
            linewidth=2,
            color="red",
            label="Time-averaged position",
            alpha=0.45,
        )
        avg_pos_lines.append(average_position_line)

        inner_turning_line = ax.axvline(
            x=R_WELL / PARTICLE_SIZE,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
            label="Turning points",
        )
        inner_tps.append(inner_turning_line)

        outer_turning_line = ax.axvline(
            x=R_WELL / PARTICLE_SIZE,
            linestyle="--",
            linewidth=1,
            color="gray",
            alpha=0.6,
        )
        outer_tps.append(outer_turning_line)

        # Always create the marker so animation callbacks can return it.
        # It remains hidden when BALL is False.
        (ball_marker,) = ax.plot(
            [],
            [],
            marker="o",
            markersize=14,
            markerfacecolor="red",
            markeredgecolor="white",
            markeredgewidth=1.2,
            linestyle="None",
            label="Oscillating particle" if BALL else "_nolegend_",
            zorder=10,
            visible=BALL,
        )
        balls.append(ball_marker)

        legend = ax.legend(
            facecolor="black",
            edgecolor="white",
            labelcolor="white",
            loc="lower right" if i == 0 else "upper right",
            frameon=True,
        )
        legend.get_frame().set_alpha(0.8)

    parameters_changed = BALL

    # Ball is None when particle animation is disabled.
    ball = Ball() if BALL else None

    def reset_ball() -> None:
        nonlocal parameters_changed

        if ball is None:
            parameters_changed = False
            return

        ball.reset(
            excitation_energy=energy_slider.val,
            epsilon=epsilon_slider.val,
        )

        parameters_changed = False

    def mark_parameters_changed(_value: float) -> None:
        nonlocal parameters_changed
        if BALL:
            parameters_changed = True

    energy_slider.on_changed(mark_parameters_changed)
    epsilon_slider.on_changed(mark_parameters_changed)

    def update_static_elements(
        excitation_energy: float,
        epsilon: float,
    ) -> None:

        total_energy = excitation_energy - epsilon
        energy_line.set_ydata(
            [total_energy, total_energy]
        )
        #compute values now to minimize breaks in double for loop
        if 0.0 < excitation_energy < epsilon:
            inner, outer = find_turning_points(
                excitation_energy,
                epsilon,
            )

            average_position = find_avg_pos(excitation_energy,epsilon)
        else:
            inner, outer, average_position = None, None, None
        for text in texts:
            avg_stiffness=find_avg_stiffness(excitation_energy,epsilon)
            text.set_text(f"average stiffness: {avg_stiffness: .1f}")
        for i, curve in enumerate(curves):
            y_values = f_to_plot[i](
                r_values,
                epsilon,
            )
            curve.set_data(
                displayed_r_values,
                y_values,
            )
            inner_data=[inner / PARTICLE_SIZE,inner / PARTICLE_SIZE] if inner is not None else []
            inner_tps[i].set_xdata(inner_data)
            outer_data = [outer / PARTICLE_SIZE, outer / PARTICLE_SIZE] if outer is not None else []
            outer_tps[i].set_xdata(outer_data)

            if average_position is not None:
                avg_pos_lines[i].set_xdata(
                    [average_position, average_position]
                )

                inner_tps[i].set_visible(True)
                outer_tps[i].set_visible(True)
                avg_pos_lines[i].set_visible(
                    average_position is not None
                )
            else:
                inner_tps[i].set_visible(False)
                outer_tps[i].set_visible(False)
                avg_pos_lines[i].set_visible(False)

    def init():
        if ball is not None:
            reset_ball()

        update_static_elements(
            energy_slider.val,
            epsilon_slider.val,
        )
        for i, ball_marker in enumerate(balls):
            if ball is not None and ball.is_bound:
                ball_marker.set_data(
                    [ball.r / PARTICLE_SIZE],
                    [
                        f_to_plot[i](
                            ball.r,
                            ball.epsilon,
                        )
                    ],
                )
                ball_marker.set_visible(True)
            else:
                ball_marker.set_data([], [])
                ball_marker.set_visible(False)

        return (
            curves,
            energy_line,
            avg_pos_lines,
            inner_tps,
            outer_tps,
            balls,
        )

    def update(_frame):
        nonlocal parameters_changed

        excitation_energy = energy_slider.val
        epsilon = epsilon_slider.val

        if parameters_changed and ball is not None:
            reset_ball()
            avg_stiffness = find_avg_stiffness(excitation_energy, epsilon)
            print(f"Average stiffness: {avg_stiffness}")

        update_static_elements(
            excitation_energy,
            epsilon,
        )

        if ball is not None and ball.is_bound:
            frame_dt = (
                ANIMATION_INTERVAL_MS
                / 1000.0
                * TIME_SCALE
            )
            physics_dt = (
                frame_dt
                / PHYSICS_STEPS_PER_FRAME
            )

            for _ in range(PHYSICS_STEPS_PER_FRAME):
                ball.update(
                    physics_dt,
                )
            for i,ball_marker in enumerate(balls):
                ball_y = f_to_plot[i](
                    ball.r,
                    epsilon,
                )

                ball_marker.set_data(
                    [ball.r / PARTICLE_SIZE],
                    [ball_y],
                )
                ball_marker.set_visible(True)

        else:
            for ball_marker in balls:
                ball_marker.set_data([], [])
                ball_marker.set_visible(False)

        return (
            curves,
            energy_line,
            avg_pos_lines,
            inner_tps,
            outer_tps,
            balls,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        interval=ANIMATION_INTERVAL_MS,
        cache_frame_data=False,
        blit=False,
    )

    # Keep references alive; otherwise widgets or animation may be
    # garbage-collected.
    sliders = (
        epsilon_slider,
        energy_slider,
    )

    return fig, axes, ani, sliders


if __name__ == "__main__":
    fig, ax, ani, sliders = animate_potential()
    plt.show(block=True)
    print(plt.get_backend())