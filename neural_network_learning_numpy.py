from __future__ import annotations

import numpy as np

from manim import (
    AnimationGroup,
    Arrow,
    Axes,
    BraceBetweenPoints,
    Circle,
    Create,
    Dot,
    DOWN,
    FadeIn,
    FadeOut,
    Flash,
    GrowArrow,
    GrowFromCenter,
    ImplicitFunction,
    LaggedStart,
    LEFT,
    Line,
    MathTex,
    MoveAlongPath,
    ORIGIN,
    Rectangle,
    ReplacementTransform,
    RIGHT,
    RoundedRectangle,
    Scene,
    ShowPassingFlash,
    Succession,
    Text,
    Transform,
    TransformFromCopy,
    UP,
    VGroup,
    VMobject,
    WHITE,
    Write,
    config,
    linear,
    smooth,
)

from neural_training_numpy import (
    NetworkSnapshot,
    effective_hidden_coefficients,
    forward,
    gradient_at_snapshot,
    hidden_state,
    predict,
    sample_binary_cross_entropy,
    train_network,
)


config.frame_width = 8
config.frame_height = 10
config.background_color = "#07111F"

BACKGROUND = "#07111F"
PANEL = "#0E2036"
MUTED = "#7C91AA"
INPUT_BLUE = "#38BDF8"
CLASS_RED = "#FF5C5C"
WEIGHT_YELLOW = "#FACC15"
ACTIVATION_GREEN = "#4ADE80"
ERROR_RED = "#FB7185"
BOUNDARY_WHITE = WHITE

PLOT_LEFT = -3.12
PLOT_BOTTOM = -0.62
PLOT_WIDTH = 6.24
PLOT_HEIGHT = 3.55
PLOT_CENTER = np.array([0.0, 1.30, 0.0])

BUNDLE = train_network()


def plot_point(x_value: float, y_value: float) -> np.ndarray:
    return np.array(
        [
            PLOT_LEFT + PLOT_WIDTH * x_value / 6,
            PLOT_BOTTOM + PLOT_HEIGHT * y_value / 4,
            0,
        ]
    )


def scene_to_data(scene_x: float, scene_y: float) -> tuple[float, float]:
    x_value = 6 * (scene_x - PLOT_LEFT) / PLOT_WIDTH
    y_value = 4 * (scene_y - PLOT_BOTTOM) / PLOT_HEIGHT
    return x_value, y_value


def tex_number(value: float, decimals: int = 2) -> str:
    rounded = f"{value:.{decimals}f}"
    if rounded.startswith("-0") and abs(value) < 0.5 * 10 ** (-decimals):
        rounded = rounded[1:]
    return rounded.replace(".", "{,}")


class NeuralNetworkLearningNumpy(Scene):
    """Anima resultados de uma rede realmente treinada com NumPy."""

    def construct(self):
        self.camera.background_color = BACKGROUND
        self.opening_scene()
        self.data_scene()
        self.weights_scene()
        self.activation_scene()
        self.layers_scene()
        self.loss_scene()
        self.backpropagation_scene()
        self.training_scene()
        self.closing_loop()

    # ------------------------------------------------------------------
    # Componentes compartilhados
    # ------------------------------------------------------------------

    def clear_scene(self, run_time: float = 0.35):
        if self.mobjects:
            self.play(
                *[FadeOut(mobject) for mobject in list(self.mobjects)],
                run_time=run_time,
            )

    def show_phrase(self, phrase: str, total_time: float = 3.0):
        self.clear_scene()
        text = self.centered_text(
            phrase,
            color=WHITE,
            font_size=43,
            weight="MEDIUM",
            line_buff=0.28,
        )
        if text.width > 6.7:
            text.scale_to_fit_width(6.7)
        self.play(FadeIn(text, shift=UP * 0.12), run_time=0.45)
        self.wait(max(0.1, total_time - 0.80))
        self.play(FadeOut(text, shift=UP * 0.08), run_time=0.35)

    def centered_text(
        self,
        phrase: str,
        color=WHITE,
        font_size: float = 43,
        weight: str = "NORMAL",
        line_buff: float = 0.22,
    ):
        lines = VGroup(
            *[
                Text(
                    line[:-1] if line.endswith(".") else line,
                    color=color,
                    font_size=font_size,
                    weight=weight,
                )
                for line in phrase.splitlines()
            ]
        )
        lines.arrange(DOWN, buff=line_buff)
        return lines

    def input_node(self, label: str, center: np.ndarray):
        circle = Circle(
            radius=0.42,
            color=INPUT_BLUE,
            stroke_width=4,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to(center)
        formula = MathTex(label, color=INPUT_BLUE, font_size=34).move_to(circle)
        return VGroup(circle, formula)

    def neuron_node(
        self,
        label: str,
        center: np.ndarray,
        color: str = ACTIVATION_GREEN,
        radius: float = 0.50,
    ):
        circle = Circle(
            radius=radius,
            color=color,
            stroke_width=4,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to(center)
        formula = MathTex(label, color=color, font_size=29).move_to(circle)
        return VGroup(circle, formula)

    def make_dataset(self):
        panel = RoundedRectangle(
            width=7.05,
            height=4.55,
            corner_radius=0.18,
            stroke_color="#29425E",
            stroke_width=2,
            fill_color=PANEL,
            fill_opacity=0.78,
        ).move_to(PLOT_CENTER)

        x_axis = Line(
            plot_point(0, 0),
            plot_point(6, 0),
            color=MUTED,
            stroke_width=2,
        )
        y_axis = Line(
            plot_point(0, 0),
            plot_point(0, 4),
            color=MUTED,
            stroke_width=2,
        )
        x_label = MathTex(r"x_1", color=INPUT_BLUE, font_size=24)
        x_label.next_to(x_axis.get_right(), DOWN, buff=0.10)
        y_label = MathTex(r"x_2", color=INPUT_BLUE, font_size=24)
        y_label.next_to(y_axis.get_top(), LEFT, buff=0.10)
        axes = VGroup(x_axis, y_axis, x_label, y_label)

        class_zero = VGroup(
            *[
                Dot(plot_point(x, y), radius=0.075, color=INPUT_BLUE)
                for (x, y), target in zip(BUNDLE.x, BUNDLE.y)
                if target == 0
            ]
        )
        class_one = VGroup(
            *[
                Dot(plot_point(x, y), radius=0.075, color=CLASS_RED)
                for (x, y), target in zip(BUNDLE.x, BUNDLE.y)
                if target == 1
            ]
        )

        legend_zero = VGroup(
            Dot(radius=0.060, color=INPUT_BLUE),
            Text("classe 0", font_size=18, color=INPUT_BLUE),
        ).arrange(RIGHT, buff=0.12)
        legend_one = VGroup(
            Dot(radius=0.060, color=CLASS_RED),
            Text("classe 1", font_size=18, color=CLASS_RED),
        ).arrange(RIGHT, buff=0.12)
        legend = VGroup(legend_zero, legend_one).arrange(RIGHT, buff=0.35)
        legend.move_to([0, 3.23, 0])
        return VGroup(panel, axes, class_zero, class_one, legend)

    def decision_boundary(
        self,
        snapshot: NetworkSnapshot,
        color: str = BOUNDARY_WHITE,
        stroke_width: float = 6,
        opacity: float = 1.0,
    ):
        def level_set(scene_x: float, scene_y: float) -> float:
            x_value, y_value = scene_to_data(scene_x, scene_y)
            probability = predict(
                BUNDLE,
                snapshot,
                np.array([[x_value, y_value]]),
            )[0]
            return float(probability - 0.5)

        return ImplicitFunction(
            level_set,
            x_range=[PLOT_LEFT, PLOT_LEFT + PLOT_WIDTH],
            y_range=[PLOT_BOTTOM, PLOT_BOTTOM + PLOT_HEIGHT],
            min_depth=6,
            max_quads=2200,
            use_smoothing=True,
            color=color,
            stroke_width=stroke_width,
            stroke_opacity=opacity,
        )

    def partial_decision_boundary(
        self,
        snapshot: NetworkSnapshot,
        hidden_units: list[int],
        opacity: float = 0.62,
    ):
        units = np.asarray(hidden_units, dtype=int)

        def partial_logit(scene_x: float, scene_y: float) -> float:
            x_value, y_value = scene_to_data(scene_x, scene_y)
            normalized = (
                np.array([x_value, y_value], dtype=float) - BUNDLE.mean
            ) / BUNDLE.scale
            pre_activations = normalized @ snapshot.w1[:, units] + snapshot.b1[units]
            activations = np.maximum(0.0, pre_activations)
            logit = activations @ snapshot.w2[units, 0] + float(snapshot.b2)
            return float(logit)

        return ImplicitFunction(
            partial_logit,
            x_range=[PLOT_LEFT, PLOT_LEFT + PLOT_WIDTH],
            y_range=[PLOT_BOTTOM, PLOT_BOTTOM + PLOT_HEIGHT],
            min_depth=6,
            max_quads=2200,
            use_smoothing=True,
            color=BOUNDARY_WHITE,
            stroke_width=4,
            stroke_opacity=opacity,
        )

    def hidden_line(self, snapshot: NetworkSnapshot, hidden_unit: int, **kwargs):
        coefficient_x, coefficient_y, bias = effective_hidden_coefficients(
            BUNDLE, snapshot, hidden_unit
        )
        candidates: list[tuple[float, float]] = []

        if abs(coefficient_y) > 1e-9:
            for x_value in (0.0, 6.0):
                y_value = -(coefficient_x * x_value + bias) / coefficient_y
                if 0 <= y_value <= 4:
                    candidates.append((x_value, y_value))
        if abs(coefficient_x) > 1e-9:
            for y_value in (0.0, 4.0):
                x_value = -(coefficient_y * y_value + bias) / coefficient_x
                if 0 <= x_value <= 6:
                    candidates.append((x_value, y_value))

        unique: list[tuple[float, float]] = []
        for candidate in candidates:
            if not any(np.linalg.norm(np.subtract(candidate, item)) < 1e-6 for item in unique):
                unique.append(candidate)
        if len(unique) < 2:
            return None

        first, second = max(
            (
                (a, b)
                for index, a in enumerate(unique)
                for b in unique[index + 1 :]
            ),
            key=lambda pair: np.linalg.norm(np.subtract(pair[0], pair[1])),
        )
        options = {
            "color": WEIGHT_YELLOW,
            "stroke_width": 4,
            **kwargs,
        }
        return Line(plot_point(*first), plot_point(*second), **options)

    def misclassification_rings(self, snapshot: NetworkSnapshot):
        predictions = (snapshot.probabilities >= 0.5).astype(float)
        rings = VGroup()
        for point, prediction, target in zip(BUNDLE.x, predictions, BUNDLE.y):
            if prediction != target:
                rings.add(
                    Circle(
                        radius=0.13,
                        color=ERROR_RED,
                        stroke_width=3,
                    ).move_to(plot_point(*point))
                )
        return rings

    def apply_plot_transform(
        self,
        mobject,
        scale: float,
        target_center: np.ndarray,
    ):
        shift = target_center - scale * PLOT_CENTER
        mobject.scale(scale, about_point=ORIGIN).shift(shift)
        return mobject

    def loss_curve(self, axes: Axes, maximum_epoch: int):
        points = [
            axes.c2p(epoch, BUNDLE.loss_history[epoch])
            for epoch in range(maximum_epoch + 1)
        ]
        curve = VMobject(color=ERROR_RED, stroke_width=5)
        if len(points) == 1:
            points.append(points[0] + RIGHT * 0.001)
        curve.set_points_smoothly(points)
        return curve

    # ------------------------------------------------------------------
    # Cena 1 — abertura e fronteira inicial real
    # ------------------------------------------------------------------

    def opening_scene(self):
        title = Text(
            "Como uma rede neural aprende?",
            color=WHITE,
            font_size=48,
            weight="BOLD",
        )
        title.scale_to_fit_width(6.8)
        # O primeiro e o último quadro usam este mesmo título para fechar o loop.
        self.add(title)
        self.wait(2.15)
        self.play(FadeOut(title, shift=UP * 0.08), run_time=0.35)

        snapshot = BUNDLE.snapshots[0]
        dataset = self.make_dataset()
        boundary = self.decision_boundary(snapshot, opacity=0.72)
        rings = self.misclassification_rings(snapshot)
        metric = MathTex(
            rf"\text{{época}}=0\qquad \bar L={tex_number(snapshot.loss, 3)}",
            font_size=31,
        ).move_to(DOWN * 1.65)
        metric.set_color_by_tex(r"\bar L", ERROR_RED)

        self.play(Create(VGroup(dataset[0], dataset[1])), run_time=0.75)
        self.play(
            LaggedStart(
                *[
                    GrowFromCenter(point)
                    for point in [*dataset[2], *dataset[3]]
                ],
                lag_ratio=0.025,
            ),
            FadeIn(dataset[4]),
            run_time=1.15,
        )
        self.play(Create(boundary), FadeIn(rings), Write(metric), run_time=0.90)
        self.wait(0.65)

    # ------------------------------------------------------------------
    # Cena 2 — valores reais da observação
    # ------------------------------------------------------------------

    def data_scene(self):
        self.show_phrase(
            "Cada observação entra na rede\ncomo um conjunto de números"
        )

        dataset = self.make_dataset()
        self.play(FadeIn(dataset, lag_ratio=0.02), run_time=0.70)

        sample_index = BUNDLE.selected_sample
        sample = BUNDLE.x[sample_index]
        sample_dot = Dot(plot_point(*sample), radius=0.075, color=CLASS_RED)
        focus = Circle(radius=0.19, color=WHITE, stroke_width=3).move_to(sample_dot)
        self.play(Create(focus), Flash(sample_dot, color=CLASS_RED), run_time=0.45)

        pair = MathTex(
            rf"(x_1,x_2)=({tex_number(sample[0])},{tex_number(sample[1])})",
            color=INPUT_BLUE,
            font_size=40,
        ).move_to(DOWN * 1.55)
        moving_point = sample_dot.copy()
        self.add(moving_point)
        self.play(
            ReplacementTransform(moving_point, pair),
            FadeOut(focus),
            run_time=0.75,
        )

        normalized = (sample - BUNDLE.mean) / BUNDLE.scale
        normalization_note = Text(
            "Antes do treino: centralizar e dividir pelo desvio padrão",
            color=MUTED,
            font_size=18,
        ).move_to(DOWN * 2.12)
        normalized_tex = MathTex(
            r"\tilde{\mathbf x}=\frac{\mathbf x-\boldsymbol\mu}{\boldsymbol\sigma}"
            rf"=({tex_number(normalized[0])},{tex_number(normalized[1])})",
            font_size=29,
        ).move_to(DOWN * 2.62)
        normalized_tex.set_color_by_tex(r"\tilde{\mathbf x}", INPUT_BLUE)
        self.play(FadeIn(normalization_note), Write(normalized_tex), run_time=0.65)

        input_one = self.input_node(r"\tilde x_1", np.array([-1.70, -3.52, 0]))
        input_two = self.input_node(r"\tilde x_2", np.array([-1.70, -4.38, 0]))
        neuron = self.neuron_node(r"z_j", np.array([2.05, -3.95, 0]))
        arrow_one = Arrow(input_one.get_right(), neuron.get_left() + UP * 0.16, buff=0.10, color=INPUT_BLUE)
        arrow_two = Arrow(input_two.get_right(), neuron.get_left() + DOWN * 0.16, buff=0.10, color=INPUT_BLUE)
        self.play(FadeIn(VGroup(input_one, input_two)), run_time=0.55)
        self.play(
            GrowArrow(arrow_one),
            GrowArrow(arrow_two),
            GrowFromCenter(neuron),
            run_time=0.75,
        )
        self.wait(0.65)

    # ------------------------------------------------------------------
    # Cena 3 — pesos e substituição numérica reais
    # ------------------------------------------------------------------

    def weights_scene(self):
        self.show_phrase(
            "Cada conexão possui um peso que controla\n"
            "a influência da entrada sobre o neurônio"
        )

        snapshot = BUNDLE.snapshots[0]
        unit = BUNDLE.selected_hidden_unit
        sample = BUNDLE.x[BUNDLE.selected_sample]
        normalized = (sample - BUNDLE.mean) / BUNDLE.scale
        weight_one, weight_two = snapshot.w1[:, unit]
        bias_value = snapshot.b1[unit]
        hidden_z, _ = hidden_state(BUNDLE, snapshot, sample.reshape(1, -1))
        z_value = hidden_z[0, unit]

        input_one = self.input_node(r"\tilde x_1", np.array([-2.65, 1.55, 0]))
        input_two = self.input_node(r"\tilde x_2", np.array([-2.65, -0.15, 0]))
        bias = self.input_node(r"1", np.array([-2.65, -1.85, 0]))
        bias[0].set_stroke(WEIGHT_YELLOW, width=4)
        bias[1].set_color(WEIGHT_YELLOW)
        neuron = self.neuron_node(rf"z_{{{unit + 1}}}", np.array([2.40, -0.10, 0]))

        width_one = 2.5 + 2.8 * abs(weight_one)
        width_two = 2.5 + 2.8 * abs(weight_two)
        line_one = Line(input_one.get_right(), neuron.get_left() + UP * 0.18, color=WEIGHT_YELLOW, stroke_width=width_one)
        line_two = Line(input_two.get_right(), neuron.get_left(), color=WEIGHT_YELLOW, stroke_width=width_two)
        line_bias = Line(bias.get_right(), neuron.get_left() + DOWN * 0.18, color=WEIGHT_YELLOW, stroke_width=2.5)

        label_one = MathTex(
            r"w_1=", tex_number(weight_one), color=WEIGHT_YELLOW, font_size=30
        ).move_to([-0.45, 1.47, 0])
        label_two = MathTex(
            r"w_2=", tex_number(weight_two), color=WEIGHT_YELLOW, font_size=30
        ).move_to([-0.05, 0.45, 0])
        bias_label = MathTex(
            r"b=", tex_number(bias_value), color=WEIGHT_YELLOW, font_size=30
        ).move_to([-0.05, -1.62, 0])

        input_value_one = MathTex(
            r"\tilde x_1=", tex_number(normalized[0]),
            color=INPUT_BLUE,
            font_size=23,
        ).next_to(input_one, UP, buff=0.10)
        input_value_two = MathTex(
            r"\tilde x_2=", tex_number(normalized[1]),
            color=INPUT_BLUE,
            font_size=23,
        ).next_to(input_two, DOWN, buff=0.10)
        z_result = MathTex(
            r"=", tex_number(z_value, 3),
            color=ACTIVATION_GREEN,
            font_size=27,
        ).next_to(neuron, RIGHT, buff=0.12)

        symbolic = MathTex(
            rf"z_{{{unit + 1}}}=w_1\tilde x_1+w_2\tilde x_2+b",
            font_size=37,
        ).move_to(DOWN * 3.55)
        symbolic.set_color_by_tex("w_1", WEIGHT_YELLOW)
        symbolic.set_color_by_tex("w_2", WEIGHT_YELLOW)
        symbolic.set_color_by_tex("b", WEIGHT_YELLOW)
        symbolic.set_color_by_tex(r"\tilde x_1", INPUT_BLUE)
        symbolic.set_color_by_tex(r"\tilde x_2", INPUT_BLUE)

        numeric = VGroup(
            MathTex(rf"z_{{{unit + 1}}}=", color=ACTIVATION_GREEN, font_size=30),
            MathTex(rf"({tex_number(weight_one)})", color=WEIGHT_YELLOW, font_size=30),
            MathTex(rf"({tex_number(normalized[0])})", color=INPUT_BLUE, font_size=30),
            MathTex(r"+", font_size=30),
            MathTex(rf"({tex_number(weight_two)})", color=WEIGHT_YELLOW, font_size=30),
            MathTex(rf"({tex_number(normalized[1])})", color=INPUT_BLUE, font_size=30),
            MathTex(r"+", font_size=30),
            MathTex(tex_number(bias_value), color=WEIGHT_YELLOW, font_size=30),
            MathTex(r"=", font_size=30),
            MathTex(tex_number(z_value, 3), color=ACTIVATION_GREEN, font_size=30),
        ).arrange(RIGHT, buff=0.045).move_to(DOWN * 3.55)
        if numeric.width > 6.90:
            numeric.scale_to_fit_width(6.90)

        self.play(
            FadeIn(VGroup(input_one, input_two, bias, neuron)),
            FadeIn(VGroup(input_value_one, input_value_two)),
            run_time=0.60,
        )
        self.play(Create(line_one), Write(label_one), run_time=0.65)
        self.play(Create(line_two), Write(label_two), run_time=0.65)
        self.play(Create(line_bias), Write(bias_label), run_time=0.55)
        self.play(Write(symbolic), run_time=0.75)
        self.play(Write(z_result), Flash(neuron, color=ACTIVATION_GREEN), run_time=0.45)
        self.play(
            FadeOut(symbolic),
            Write(VGroup(numeric[0], numeric[3], numeric[6], numeric[8])),
            run_time=0.35,
        )
        self.play(
            TransformFromCopy(label_one[1], numeric[1]),
            TransformFromCopy(input_value_one[1], numeric[2]),
            TransformFromCopy(label_two[1], numeric[4]),
            TransformFromCopy(input_value_two[1], numeric[5]),
            TransformFromCopy(bias_label[1], numeric[7]),
            TransformFromCopy(z_result[1], numeric[9]),
            run_time=1.00,
        )
        self.wait(0.80)

    # ------------------------------------------------------------------
    # Cena 4 — ReLU real, sigmoide de saída e fronteira reta
    # ------------------------------------------------------------------

    def activation_scene(self):
        self.show_phrase(
            "O neurônio transforma a soma\ncom uma função de ativação"
        )

        snapshot = BUNDLE.snapshots[0]
        unit = BUNDLE.selected_hidden_unit
        sample = BUNDLE.x[BUNDLE.selected_sample]
        z1, hidden, output_logit, probability = forward(
            ((sample - BUNDLE.mean) / BUNDLE.scale).reshape(1, -1),
            snapshot.w1,
            snapshot.b1,
            snapshot.w2,
            snapshot.b2,
        )
        z_value = z1[0, unit]
        activation_value = hidden[0, unit]

        z_equation = MathTex(
            rf"z_{{{unit + 1}}}={tex_number(z_value, 3)}",
            font_size=42,
        ).move_to(UP * 4.05)
        self.play(Write(z_equation), run_time=0.45)

        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 2, 1],
            x_length=4.7,
            y_length=2.20,
            tips=False,
            axis_config={"color": MUTED, "include_ticks": False},
        ).move_to(UP * 2.35)
        relu = axes.plot(lambda z: max(0, z), x_range=[-2, 2], color=ACTIVATION_GREEN, stroke_width=6)
        axis_z = MathTex(r"z", color=MUTED, font_size=25).next_to(axes.x_axis, RIGHT, buff=0.08)
        axis_h = MathTex(r"h", color=ACTIVATION_GREEN, font_size=25).next_to(axes.y_axis, UP, buff=0.08)
        self.play(Create(axes), Create(relu), FadeIn(VGroup(axis_z, axis_h)), run_time=0.65)

        relu_definition = MathTex(
            r"\operatorname{ReLU}(z)=\max(0,z)",
            font_size=36,
        ).move_to(UP * 0.62)
        relu_definition.set_color_by_tex(r"\operatorname{ReLU}", ACTIVATION_GREEN)
        relu_note = self.centered_text(
            "se z é negativo, a saída vira 0\n"
            "se z é positivo, o próprio valor é mantido",
            color=WHITE,
            font_size=18,
            line_buff=0.10,
        ).move_to(DOWN * 0.02)
        self.play(Write(relu_definition), FadeIn(relu_note), run_time=0.60)

        start_dot = Dot(axes.c2p(-1.5, 0), color=ACTIVATION_GREEN)
        destination = axes.c2p(z_value, activation_value)
        path = VMobject().set_points_as_corners([start_dot.get_center(), axes.c2p(0, 0), destination])
        self.add(start_dot)
        self.play(MoveAlongPath(start_dot, path), run_time=0.65, rate_func=linear)

        relu_equation = MathTex(
            rf"h_{{{unit + 1}}}=\operatorname{{ReLU}}(z_{{{unit + 1}}})={tex_number(activation_value, 3)}",
            font_size=36,
        ).move_to(UP * 0.62)
        relu_equation.set_color_by_tex(r"\operatorname{ReLU}", ACTIVATION_GREEN)
        self.play(ReplacementTransform(relu_definition, relu_equation), run_time=0.55)
        self.wait(0.25)

        combine_note = self.centered_text(
            "O neurônio de saída combina as ativações\n"
            "dos 8 neurônios ocultos",
            color=WHITE,
            font_size=22,
            line_buff=0.16,
        ).move_to(UP * 4.00)
        self.play(
            FadeOut(
                VGroup(
                    z_equation,
                    axes,
                    relu,
                    axis_z,
                    axis_h,
                    start_dot,
                    relu_equation,
                    relu_note,
                )
            ),
            FadeIn(combine_note),
            run_time=0.40,
        )

        hidden_nodes = VGroup()
        value_labels = VGroup()
        hidden_values = hidden[0]
        for index, x_position in enumerate(np.linspace(-3.05, 3.05, 8)):
            node = self.neuron_node(
                rf"h_{{{index + 1}}}",
                np.array([x_position, 2.25, 0]),
                radius=0.27,
            )
            node[1].scale(0.70)
            hidden_nodes.add(node)
            value_labels.add(
                MathTex(
                    tex_number(float(hidden_values[index]), 3),
                    color=ACTIVATION_GREEN,
                    font_size=18,
                ).next_to(node, UP, buff=0.10)
            )

        output_node = self.neuron_node(r"s", np.array([0.0, 0.38, 0]), radius=0.43)
        connections = VGroup(
            *[
                Line(
                    node.get_bottom(),
                    output_node.get_top(),
                    color=WEIGHT_YELLOW,
                    stroke_width=1.8,
                    stroke_opacity=0.58,
                )
                for node in hidden_nodes
            ]
        )
        connections.set_z_index(-1)
        self.play(
            LaggedStart(*[GrowFromCenter(node) for node in hidden_nodes], lag_ratio=0.05),
            FadeIn(value_labels),
            run_time=0.60,
        )
        self.play(
            LaggedStart(*[Create(line) for line in connections], lag_ratio=0.04),
            GrowFromCenter(output_node),
            run_time=0.70,
        )

        sum_equation = MathTex(
            rf"s=\sum_{{j=1}}^{{8}}v_jh_j+c={tex_number(output_logit[0], 3)}",
            font_size=36,
        ).move_to(DOWN * 1.25)
        sum_equation.set_color_by_tex("v_j", WEIGHT_YELLOW)
        output_equation = MathTex(
            rf"\hat y=\sigma(s)={tex_number(probability[0], 3)}",
            font_size=39,
        ).move_to(DOWN * 2.18)
        output_equation.set_color_by_tex(r"\sigma", ACTIVATION_GREEN)
        self.play(Write(sum_equation), run_time=0.55)
        self.play(Write(output_equation), run_time=0.50)
        self.wait(0.70)

    # ------------------------------------------------------------------
    # Cena 5 — retas ocultas e fronteira real combinada
    # ------------------------------------------------------------------

    def layers_scene(self):
        self.show_phrase(
            "Cada ReLU muda de regime sobre uma reta\n"
            "Juntos, os 8 neurônios formam a fronteira"
        )

        snapshot = BUNDLE.snapshots[100]
        dataset = self.make_dataset()
        self.play(FadeIn(dataset, lag_ratio=0.02), run_time=0.60)

        ranked_units = np.argsort(np.abs(snapshot.w2.reshape(-1)))[::-1]
        selected_units: list[int] = []
        hidden_lines = VGroup()
        for unit in ranked_units:
            line = self.hidden_line(snapshot, int(unit), color=WEIGHT_YELLOW, stroke_width=3.5)
            if line is not None:
                selected_units.append(int(unit))
                hidden_lines.add(line)
            if len(selected_units) == 4:
                break

        neuron_centers = [
            np.array([-2.45, -3.58, 0]),
            np.array([-0.82, -3.58, 0]),
            np.array([0.82, -3.58, 0]),
            np.array([2.45, -3.58, 0]),
        ]
        neurons = VGroup(
            *[
                self.neuron_node(rf"h_{{{unit + 1}}}", center, radius=0.38)
                for unit, center in zip(selected_units, neuron_centers)
            ]
        )
        shown_note = Text(
            "4 das 8 retas aprendidas são mostradas",
            color=WEIGHT_YELLOW,
            font_size=21,
        ).move_to(DOWN * 2.75)
        self.play(
            LaggedStart(*[GrowFromCenter(node) for node in neurons], lag_ratio=0.10),
            FadeIn(shown_note),
            run_time=0.65,
        )

        partial_boundary = None
        partial_note = None
        for index, (neuron, line) in enumerate(zip(neurons, hidden_lines), start=1):
            next_note = Text(
                (
                    "1 neurônio: ainda não há fronteira parcial"
                    if index == 1
                    else f"fronteira parcial com {index} neurônios"
                ),
                color=WHITE,
                font_size=17,
            ).move_to(DOWN * 4.42)
            animations = [
                Flash(neuron, color=ACTIVATION_GREEN, flash_radius=0.50),
                Create(line),
            ]

            if index >= 2:
                next_boundary = self.partial_decision_boundary(
                    snapshot,
                    selected_units[:index],
                )
                if partial_boundary is None:
                    animations.append(Create(next_boundary))
                    partial_boundary = next_boundary
                else:
                    animations.append(Transform(partial_boundary, next_boundary))

            if partial_note is None:
                animations.append(FadeIn(next_note))
            else:
                animations.append(ReplacementTransform(partial_note, next_note))

            self.play(*animations, run_time=0.68)
            partial_note = next_note

        final_boundary = self.decision_boundary(snapshot)
        hidden_lines.save_state()
        self.play(
            hidden_lines.animate.set_stroke(opacity=0.18, width=2),
            Transform(partial_boundary, final_boundary),
            run_time=0.90,
        )
        caption = MathTex(
            r"\hat y(x_1,x_2)=0{,}5",
            color=WHITE,
            font_size=30,
        ).move_to(DOWN * 2.72)
        all_units_note = Text(
            "fronteira final: combinação dos 8 neurônios",
            color=WHITE,
            font_size=19,
        ).move_to(DOWN * 4.42)
        self.play(
            ReplacementTransform(shown_note, caption),
            ReplacementTransform(partial_note, all_units_note),
            run_time=0.45,
        )
        self.wait(0.85)

    # ------------------------------------------------------------------
    # Cena 6 — previsão e BCE reais
    # ------------------------------------------------------------------

    def loss_scene(self):
        self.show_phrase(
            "A previsão do modelo é comparada\ncom a resposta correta"
        )

        snapshot = BUNDLE.snapshots[0]
        sample_index = BUNDLE.selected_sample
        probability = float(snapshot.probabilities[sample_index])
        target = float(BUNDLE.y[sample_index])
        sample_loss = sample_binary_cross_entropy(target, probability)

        prediction_box = RoundedRectangle(
            width=2.80,
            height=1.25,
            corner_radius=0.16,
            color=ACTIVATION_GREEN,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to([-1.70, 2.85, 0])
        target_box = RoundedRectangle(
            width=2.60,
            height=1.25,
            corner_radius=0.16,
            color=WHITE,
            fill_color=PANEL,
            fill_opacity=1,
        ).move_to([1.70, 2.85, 0])
        prediction = MathTex(
            rf"\hat y={tex_number(probability, 3)}",
            font_size=40,
        ).move_to(prediction_box)
        prediction.set_color_by_tex(r"\hat y", ACTIVATION_GREEN)
        target_tex = MathTex(r"y=1", font_size=42).move_to(target_box)
        self.play(FadeIn(prediction_box), Write(prediction), run_time=0.65)
        self.play(FadeIn(target_box), Write(target_tex), run_time=0.60)

        number_line = Line(LEFT * 2.8, RIGHT * 2.8, color=MUTED, stroke_width=4).move_to(UP * 0.70)
        zero_tick = Line(UP * 0.12, DOWN * 0.12, color=MUTED).move_to(number_line.get_left())
        one_tick = Line(UP * 0.12, DOWN * 0.12, color=MUTED).move_to(number_line.get_right())
        zero_label = MathTex(r"0", font_size=28).next_to(zero_tick, DOWN, buff=0.16)
        one_label = MathTex(r"1", font_size=28).next_to(one_tick, DOWN, buff=0.16)
        self.play(Create(number_line), FadeIn(VGroup(zero_tick, one_tick, zero_label, one_label)), run_time=0.60)

        prediction_point = number_line.get_left() + RIGHT * number_line.width * probability
        target_point = number_line.get_right()
        prediction_dot = Dot(prediction_point, color=ACTIVATION_GREEN, radius=0.11)
        target_dot = Dot(target_point, color=WHITE, radius=0.11)
        error_bar = Line(prediction_point, target_point, color=ERROR_RED, stroke_width=10)
        self.play(GrowFromCenter(prediction_dot), GrowFromCenter(target_dot), run_time=0.40)
        self.play(Create(error_bar), run_time=0.60)

        brace = BraceBetweenPoints(prediction_point, target_point, direction=DOWN, color=ERROR_RED)
        distance = MathTex(
            rf"|y-\hat y|={tex_number(abs(target - probability), 3)}",
            color=ERROR_RED,
            font_size=32,
        ).next_to(brace, DOWN, buff=0.15)
        self.play(Create(brace), Write(distance), run_time=0.60)

        absolute_note = Text(
            "Erro absoluto entre os valores\n"
            "(não é a perda usada no treino)",
            color=ERROR_RED,
            font_size=19,
            line_spacing=1.00,
        ).move_to(DOWN * 0.62)
        condition_note = MathTex(
            r"\text{Como a resposta correta é }y=1:",
            color=WHITE,
            font_size=27,
        ).move_to(DOWN * 1.58)
        bce_note = Text(
            "A entropia cruzada usada no treino se reduz a:",
            color=WHITE,
            font_size=18,
        ).move_to(DOWN * 2.00)
        bce = MathTex(
            rf"L_i=-\ln(\hat y)=-\ln({tex_number(probability, 3)})={tex_number(sample_loss, 3)}",
            color=ERROR_RED,
            font_size=34,
        ).move_to(DOWN * 2.62)
        mean_loss = MathTex(
            rf"\bar L_0={tex_number(snapshot.loss, 3)}\quad\text{{média dos 28 pontos}}",
            color=ERROR_RED,
            font_size=29,
        ).move_to(DOWN * 3.52)
        self.play(FadeIn(absolute_note), run_time=0.35)
        self.play(FadeIn(condition_note), FadeIn(bce_note), Write(bce), run_time=0.65)
        self.play(Write(mean_loss), run_time=0.45)
        self.wait(0.65)

    # ------------------------------------------------------------------
    # Cena 7 — atualização exata de um peso
    # ------------------------------------------------------------------

    def backpropagation_scene(self):
        self.show_phrase(
            "O gradiente indica como alterar cada peso\n"
            "para reduzir a perda média"
        )

        snapshot_0 = BUNDLE.snapshots[0]
        snapshot_1 = BUNDLE.snapshots[1]
        dw1, _, _, _ = gradient_at_snapshot(BUNDLE, snapshot_0)
        weight_index = np.unravel_index(np.argmax(np.abs(dw1)), dw1.shape)
        input_index, hidden_index = (int(weight_index[0]), int(weight_index[1]))
        old_weight = float(snapshot_0.w1[input_index, hidden_index])
        gradient = float(dw1[input_index, hidden_index])
        new_weight = float(snapshot_1.w1[input_index, hidden_index])

        input_nodes = VGroup(
            self.input_node(r"\tilde x_1", np.array([-2.75, 1.05, 0])),
            self.input_node(r"\tilde x_2", np.array([-2.75, -1.05, 0])),
        )
        hidden_nodes = VGroup(
            self.neuron_node(rf"h_{{{hidden_index + 1}}}", np.array([0.0, 1.05, 0]), radius=0.46),
            self.neuron_node(r"h_k", np.array([0.0, -1.05, 0]), radius=0.46),
        )
        output_node = self.neuron_node(r"\hat y", np.array([2.75, 0.0, 0]), radius=0.52)
        connections = VGroup(
            *[
                Line(source.get_right(), target.get_left(), color=WEIGHT_YELLOW, stroke_width=3)
                for source in input_nodes
                for target in hidden_nodes
            ],
            *[
                Line(source.get_right(), output_node.get_left(), color=WEIGHT_YELLOW, stroke_width=3)
                for source in hidden_nodes
            ],
        )
        target_connection_index = input_index * 2
        connections[target_connection_index].set_stroke(width=5.4)
        network = VGroup(connections, input_nodes, hidden_nodes, output_node)
        self.play(FadeIn(network, lag_ratio=0.03), run_time=0.75)

        error = Dot(output_node.get_center(), color=ERROR_RED, radius=0.13)
        reverse_paths = [
            Line(output_node.get_left(), hidden_nodes[0].get_right()),
            Line(hidden_nodes[0].get_left(), input_nodes[input_index].get_right()),
        ]
        self.add(error)
        self.play(
            Succession(
                MoveAlongPath(error, reverse_paths[0], rate_func=linear),
                MoveAlongPath(error, reverse_paths[1], rate_func=linear),
            ),
            *[
                ShowPassingFlash(
                    connection.copy().set_color(ERROR_RED).set_stroke(width=7),
                    time_width=0.55,
                )
                for connection in connections
            ],
            run_time=1.15,
        )

        rule = MathTex(
            r"w\leftarrow w-\eta\frac{\partial\bar L}{\partial w}",
            font_size=41,
        ).move_to(DOWN * 2.28)
        rule.set_color_by_tex("w", WEIGHT_YELLOW)
        rule.set_color_by_tex(r"\frac{\partial\bar L}{\partial w}", ERROR_RED)
        self.play(Write(rule), run_time=0.70)

        numeric = MathTex(
            rf"w_{{{input_index + 1},{hidden_index + 1}}}\leftarrow"
            rf"{tex_number(old_weight, 3)}-{tex_number(BUNDLE.learning_rate, 2)}"
            rf"({tex_number(gradient, 3)})={tex_number(new_weight, 3)}",
            font_size=32,
        ).move_to(DOWN * 3.25)
        numeric.set_color_by_tex(rf"w_{{{input_index + 1},{hidden_index + 1}}}", WEIGHT_YELLOW)
        self.play(Write(numeric), run_time=0.75)
        self.play(
            connections[target_connection_index].animate.set_stroke(width=4.7),
            Flash(connections[target_connection_index].get_center(), color=WEIGHT_YELLOW),
            run_time=0.45,
        )

        loss_change = MathTex(
            rf"\bar L:\ {tex_number(snapshot_0.loss, 3)}\rightarrow{tex_number(snapshot_1.loss, 3)}",
            color=ERROR_RED,
            font_size=34,
        ).move_to(DOWN * 4.05)
        self.play(Write(loss_change), run_time=0.50)

        self.play(FadeOut(VGroup(network, error)), run_time=0.40)
        plot_scale = 0.64
        plot_center = UP * 1.15
        dataset = self.apply_plot_transform(self.make_dataset(), plot_scale, plot_center)
        boundary_0 = self.apply_plot_transform(
            self.decision_boundary(snapshot_0, opacity=0.65),
            plot_scale,
            plot_center,
        )
        boundary_1 = self.apply_plot_transform(
            self.decision_boundary(snapshot_1),
            plot_scale,
            plot_center,
        )
        self.play(FadeIn(dataset), Create(boundary_0), run_time=0.55)
        self.play(Transform(boundary_0, boundary_1), run_time=0.70)
        self.wait(0.65)

    # ------------------------------------------------------------------
    # Cena 8 — snapshots e perda provenientes do treinamento
    # ------------------------------------------------------------------

    def training_scene(self):
        self.clear_scene(run_time=0.35)

        epochs = [0, 1, 10, 50, 100]
        scale = 0.80
        target_center = UP * 1.10
        dataset = self.apply_plot_transform(self.make_dataset(), scale, target_center)
        boundary = self.apply_plot_transform(
            self.decision_boundary(BUNDLE.snapshots[0], opacity=0.72),
            scale,
            target_center,
        )
        rings = self.apply_plot_transform(
            self.misclassification_rings(BUNDLE.snapshots[0]),
            scale,
            target_center,
        )

        epoch_label = MathTex(r"\text{época}=0", font_size=33).move_to(UP * 4.00)
        metric_label = MathTex(
            rf"\bar L={tex_number(BUNDLE.snapshots[0].loss, 3)}\quad"
            rf"\text{{acurácia de treinamento}}={100 * BUNDLE.snapshots[0].accuracy:.0f}\%",
            font_size=24,
        ).move_to(UP * 3.48)
        metric_label.set_color_by_tex(r"\bar L", ERROR_RED)

        loss_axes = Axes(
            x_range=[0, 100, 20],
            y_range=[0, 0.65, 0.20],
            x_length=5.10,
            y_length=1.48,
            tips=False,
            axis_config={"color": MUTED, "include_ticks": False},
        ).move_to(DOWN * 3.40)
        loss_label = MathTex(r"\bar L", color=ERROR_RED, font_size=27).next_to(loss_axes.y_axis, UP, buff=0.08)
        x_label = MathTex(r"\text{épocas}", color=MUTED, font_size=24).next_to(loss_axes.x_axis, DOWN, buff=0.10)
        visible_curve = self.loss_curve(loss_axes, 0)

        self.play(FadeIn(dataset), Create(boundary), FadeIn(rings), Write(epoch_label), run_time=0.75)
        self.play(Write(metric_label), Create(loss_axes), FadeIn(VGroup(loss_label, x_label)), run_time=0.65)
        self.add(visible_curve)

        for epoch in epochs[1:]:
            snapshot = BUNDLE.snapshots[epoch]
            next_boundary = self.apply_plot_transform(
                self.decision_boundary(snapshot), scale, target_center
            )
            next_rings = self.apply_plot_transform(
                self.misclassification_rings(snapshot), scale, target_center
            )
            next_epoch_label = MathTex(
                rf"\text{{época}}={epoch}", font_size=33
            ).move_to(epoch_label)
            next_metric = MathTex(
                rf"\bar L={tex_number(snapshot.loss, 3)}\quad"
                rf"\text{{acurácia de treinamento}}={100 * snapshot.accuracy:.0f}\%",
                font_size=24,
            ).move_to(metric_label)
            next_metric.set_color_by_tex(r"\bar L", ERROR_RED)
            next_curve = self.loss_curve(loss_axes, epoch)

            animations = [
                Succession(
                    FadeOut(epoch_label, shift=UP * 0.06),
                    FadeIn(next_epoch_label, shift=UP * 0.06),
                ),
                Succession(
                    FadeOut(metric_label, shift=UP * 0.04),
                    FadeIn(next_metric, shift=UP * 0.04),
                ),
                Transform(boundary, next_boundary),
                Transform(visible_curve, next_curve),
            ]
            if len(rings) and len(next_rings):
                animations.append(Transform(rings, next_rings))
            elif len(rings):
                animations.append(FadeOut(rings))
            elif len(next_rings):
                animations.append(FadeIn(next_rings))

            self.play(*animations, run_time=1.05, rate_func=smooth)
            epoch_label = next_epoch_label
            metric_label = next_metric
            if len(next_rings) == 0:
                rings = VGroup()
            elif not len(rings):
                rings = next_rings

        verified = Text(
            "100% dos 28 pontos de treinamento\nclassificados",
            color=ACTIVATION_GREEN,
            font_size=22,
            line_spacing=1.05,
        ).move_to(DOWN * 1.90)
        self.play(FadeIn(verified), Flash(boundary, color=WHITE), run_time=0.55)
        self.wait(1.00)

    # ------------------------------------------------------------------
    # Encerramento e retorno ao título inicial para um loop contínuo
    # ------------------------------------------------------------------

    def closing_loop(self):
        self.clear_scene(run_time=0.45)
        conclusion = self.centered_text(
            "Treinar significa atualizar pesos com o gradiente:\n"
            "a fronteira e a perda mudam juntas",
            color=WHITE,
            font_size=38,
            weight="MEDIUM",
            line_buff=0.26,
        )
        conclusion.scale_to_fit_width(6.75)
        self.play(FadeIn(conclusion, shift=UP * 0.12), run_time=0.55)
        self.wait(2.15)

        title = Text(
            "Como uma rede neural aprende?",
            color=WHITE,
            font_size=48,
            weight="BOLD",
        )
        title.scale_to_fit_width(6.8)
        self.play(ReplacementTransform(conclusion, title), run_time=0.60)
        self.wait(1.25)
