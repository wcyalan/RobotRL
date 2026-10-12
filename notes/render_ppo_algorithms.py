"""Typeset the algorithm source as paper-style SVG figures.

Run from any directory with the project's Python environment:
    python notes/render_ppo_algorithms.py

assets/ppo_algorithms/source.json is the source of truth. This is a document
renderer, not an implementation of the reinforcement learning algorithms.
Requires matplotlib. Fonts default to Windows Times/SimSun;
DejaVu Serif and Noto Serif CJK SC are fallback families on other systems.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text


HERE = Path(__file__).resolve().parent
NOTE = HERE / "03_ppo.md"
OUT = HERE / "assets" / "ppo_algorithms"
WIDTH = 660.0  # points
MARGIN = 20.0
BODY_X = 48.0
INDENT = 17.0
FONT_SIZE = 11.5
DPI = 180

matplotlib.rcParams.update({
    "font.family": ["Times New Roman", "SimSun", "DejaVu Serif", "Noto Serif CJK SC"],
    "mathtext.fontset": "cm",
    "svg.fonttype": "path",
    "svg.hashsalt": "robotrl-ppo-algorithms",
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
})
# Only use installed families, so fallback lookup does not emit warnings.
from matplotlib import font_manager

available = {f.name for f in font_manager.fontManager.ttflist}
families = [f for f in matplotlib.rcParams["font.family"] if f in available]
if not any(f in available for f in ("SimSun", "Noto Serif CJK SC")):
    raise RuntimeError("Install SimSun or Noto Serif CJK SC to render Chinese pseudocode.")
matplotlib.rcParams["font.family"] = families


@dataclass(frozen=True)
class Spec:
    section: str
    slug: str
    title: str
    output: str


SPECS = [
    Spec("1.3", "reinforce", "REINFORCE", r"策略参数 $\theta$"),
    Spec("1.4", "a2c", "A2C (n-step)", r"策略参数 $\theta$ 与价值参数 $\phi$"),
    Spec("3.1", "natural_gradient", "Natural Policy Gradient", r"策略参数 $\theta$ 与价值参数 $\phi$"),
    Spec("3.2", "trpo", "TRPO", r"策略参数 $\theta$ 与价值参数 $\phi$"),
    Spec("3.3", "ppo_penalty", "PPO-Penalty", r"策略参数 $\theta$、价值参数 $\phi$ 与惩罚系数 $\beta$"),
    Spec("6.1", "gae", "GAE / Return Targets", r"固定的优势 $\hat{A}$ 与回报标签 $\hat{R}^{\lambda}$"),
    Spec("8.2", "ppo_clip", "PPO-Clip", r"策略参数 $\theta$ 与价值参数 $\phi$"),
]

# These substitutions change notation only. All control flow, comments and
# surrounding explanations are taken directly from the Markdown source.
EXPRESSIONS = {
    "loss = -(1 / M) * sum_i sum_t [gamma^t * log_prob_theta(a_t | s_t) * G_t]":
        r"$\mathcal{L}=-\frac{1}{M}\sum_{i=1}^{M}\sum_t\gamma^t\log\pi_\theta(a_t\mid s_t)G_t$",
    "actor_loss = -mean(log_prob * A_raw)":
        r"$\mathcal{L}_{\mathrm{actor}}=-\mathrm{mean}(\ell_\theta\hat{A})$",
    "critic_loss = mean((V_phi(s) - R_target)^2)":
        r"$\mathcal{L}_{\mathrm{critic}}=\mathrm{mean}((V_\phi(s)-\hat{R}^{\lambda})^2)$",
    "定义 L(theta) = mean(exp(log_prob_theta - old_log_prob) * A_raw)":
        r"定义 $L(\theta)=\mathrm{mean}(\exp(\ell_\theta-\ell^{\mathrm{old}})\hat{A})$",
    "H = F + zeta * I": r"$H=F+\zeta I$",
    "q = x^T * H * x": r"$q=x^\top Hx$",
    "q = x^T * Hvp(x)": r"$q=x^\top\mathrm{Hvp}(x)$",
    "step = sqrt(2 * delta / q) * x": r"$d=\sqrt{\frac{2\delta}{q}}x$",
    "full_step = sqrt(2 * delta / q) * x": r"$d=\sqrt{\frac{2\delta}{q}}x$",
    "candidate = theta_old + backtrack^j * full_step":
        r"$\theta_{\mathrm{cand}}=\theta_{\mathrm{old}}+\kappa^j d$",
    "ratio = exp(log_prob_theta(a | s) - old_log_prob)":
        r"$\rho=\exp(\log\pi_\theta(a\mid s)-\ell^{\mathrm{old}})$",
    "actor_loss = -mean(ratio * A_raw) + beta * kl":
        r"$\mathcal{L}_{\mathrm{actor}}=-\mathrm{mean}(\rho\hat{A})+\beta\bar{D}_{\mathrm{KL}}$",
    "td = reward[t] + gamma * b[t] * next_value[t] - old_value[t]":
        r"$\delta_t=r_t+\gamma b_tV_{\mathrm{old}}(s_{t+1}^{*})-V_{\mathrm{old}}(s_t)$",
    "A_raw[t] = td + gamma * lambda * c[t] * gae_next":
        r"$\hat{A}_t=\delta_t+\gamma\lambda c_t\hat{A}_{\mathrm{next}}$",
    "gae_next = A_raw[t]": r"$\hat{A}_{\mathrm{next}}=\hat{A}_t$",
    "R_target = old_value + A_raw": r"$\hat{R}^{\lambda}=V_{\mathrm{old}}+\hat{A}$",
    "A_policy = (A_raw - mean(A_raw)) / (std(A_raw) + 1e-8)":
        r"$\tilde{A}=\frac{\hat{A}-\mathrm{mean}(\hat{A})}{\mathrm{std}(\hat{A})+10^{-8}}$",
    "log_ratio = current_log_prob - old_log_prob": r"$z=\ell_\theta-\ell^{\mathrm{old}}$",
    "ratio = exp(log_ratio)": r"$\rho=\exp(z)$",
    "unclipped = ratio * A_policy": r"$u=\rho\tilde{A}$",
    "clipped = clip(ratio, 1-epsilon, 1+epsilon) * A_policy":
        r"$v=\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)\tilde{A}$",
    "policy_loss = -mean(min(unclipped, clipped))":
        r"$\mathcal{L}_{\mathrm{policy}}=-\mathrm{mean}(\min(u,v))$",
    "value_loss = mean((current_value - R_target)^2)":
        r"$\mathcal{L}_{\mathrm{value}}=\mathrm{mean}((V_\phi(s)-\hat{R}^{\lambda})^2)$",
    "loss = policy_loss + c_v * value_loss - c_e * mean(entropy)":
        r"$\mathcal{L}=\mathcal{L}_{\mathrm{policy}}+c_v\mathcal{L}_{\mathrm{value}}-c_e\mathrm{mean}(\mathcal{H})$",
    "approx_kl = mean(exp(log_ratio) - 1 - log_ratio)":
        r"$\widehat{D}_{\mathrm{KL}}=\mathrm{mean}(\exp(z)-1-z)$",
}
TOKENS = {
    "log_prob_theta(a_t | s_t)": r"\log\pi_\theta(a_t\mid s_t)",
    "V_phi(s)": r"V_\phi(s)",
    "KL_old(theta)": r"\bar{D}_{\mathrm{KL}}(\theta)",
    "KL_old": r"\bar{D}_{\mathrm{KL}}",
    "theta_old": r"\theta_{\mathrm{old}}", "theta": r"\theta", "phi": r"\phi",
    "old_log_prob": r"\ell^{\mathrm{old}}", "current_log_prob": r"\ell_\theta",
    "log_prob": r"\ell_\theta", "old_value": r"V_{\mathrm{old}}",
    "next_value": r"V_{\mathrm{next}}", "current_value": r"V_\phi",
    "A_raw": r"\hat{A}", "A_policy": r"\tilde{A}", "R_target": r"\hat{R}^{\lambda}",
    "actor_loss": r"\mathcal{L}_{\mathrm{actor}}", "critic_loss": r"\mathcal{L}_{\mathrm{critic}}",
    "gae_next": r"\hat{A}_{\mathrm{next}}", "gamma": r"\gamma", "lambda": r"\lambda",
    "alpha": r"\alpha", "delta": r"\delta", "zeta": r"\zeta", "epsilon": r"\epsilon",
    "beta": r"\beta", "backtrack": r"\kappa", "full_step": r"d", "step": r"d",
    "candidate": r"\theta_{\mathrm{cand}}", "d_target": r"d_{\mathrm{target}}",
    "target_kl": r"d_{\mathrm{target}}", "approx_kl": r"\widehat{D}_{\mathrm{KL}}",
    "c_v": r"c_v", "c_e": r"c_e", "G_t": r"G_t", "s_t": r"s_t", "a_t": r"a_t",
    "r_t": r"r_t", "L_old": r"L_{\mathrm{old}}", "K_ls": r"K_{\mathrm{ls}}",
    "g^T": r"g^\top", "theta_{k+1}": r"\theta_{k+1}",
}
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(
    re.escape(s) for s in sorted(TOKENS, key=len, reverse=True)
) + r")(?![A-Za-z0-9_])")


def notation(text: str) -> str:
    if text in EXPRESSIONS:
        # No pure spacing macros: the note renderer has rejected these before.
        return EXPRESSIONS[text].replace(r"\,", "")
    return "optimizer step".join(
        TOKEN_RE.sub(lambda m: "$" + TOKENS[m.group()] + "$", part)
        for part in text.split("optimizer step")
    )



def properties(weight="normal", size=FONT_SIZE):
    return FontProperties(family=families, size=size, weight=weight)


def text_runs(text):
    # Keep CJK outside MathText: its default text font lacks Chinese glyphs.
    return [part for part in re.split(r"(\$[^$]+\$)", text) if part]


def render(spec: Spec, number: int, source: str) -> dict:
    fig = plt.figure(figsize=(WIDTH / 72, 2), dpi=DPI)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width_cache = {}

    def measure(text, weight="normal", size=FONT_SIZE):
        key = (text, weight, size)
        if key not in width_cache:
            widths, heights = [], []
            for part in text_runs(text):
                artist = Text(0, 0, part, fontproperties=properties(weight, size),
                              parse_math=part.startswith("$"))
                artist.set_figure(fig)
                bbox = artist.get_window_extent(renderer)
                widths.append(bbox.width * 72 / DPI)
                heights.append(bbox.height * 72 / DPI)
            width_cache[key] = (sum(widths), max(heights, default=0))
        return width_cache[key]

    def wrap(text, max_width, weight="normal"):
        # Never split a math expression or a Latin identifier across lines.
        atoms = re.findall(r"\$[^$]+\$|[A-Za-z0-9_]+|[^$A-Za-z0-9_]", text)
        lines, current = [], ""
        for atom in atoms:
            trial = current + atom
            if current and measure(trial, weight)[0] > max_width:
                lines.append(current.rstrip())
                current = atom.lstrip()
            else:
                current = trial
        if current:
            lines.append(current.rstrip())
        for line in lines:
            if measure(line, weight)[0] > max_width + 0.1:
                raise ValueError(f"Unbreakable line too wide in {spec.slug}: {line}")
        return lines

    # Row tuple: x, y, text, weight, color, optional number, indentation level.
    rows = []
    y = 57.0
    number_line = 0
    body_top = None
    metadata_mode = True

    def add(text, indent=0, kind="body", line_number=None):
        nonlocal y
        x = MARGIN if kind == "metadata" else BODY_X + indent * INDENT
        color = "#555555" if kind == "comment" else "#151515"
        wrapped = wrap(text, WIDTH - MARGIN - x)
        for i, line in enumerate(wrapped):
            height = max(21.0, measure(line)[1] + 7.0)
            rows.append((x, y + height / 2, line, "normal", color,
                         line_number if i == 0 else None, indent if kind != "metadata" else 0))
            y += height

    lines = source.splitlines()
    for raw in lines:
        stripped = raw.strip()
        if not stripped:
            if not metadata_mode:
                y += 4
            continue
        if stripped.startswith("子过程 GAE_"):
            continue  # The algorithm caption already identifies this procedure.
        if metadata_mode and (stripped.startswith("输入") or raw.startswith("      ")):
            add(notation(stripped), kind="metadata")
            continue
        if metadata_mode:
            add("输出：" + spec.output, kind="metadata")
            y += 10
            body_top = y - 4
            metadata_mode = False
        indent = (len(raw) - len(raw.lstrip())) // 4
        comment_parts = re.split(r"\s+#\s*", stripped, maxsplit=1)
        main = comment_parts[0]
        if stripped.startswith("#"):
            add(notation(stripped[1:].strip()), indent, "comment")
            continue
        number_line += 1
        add(notation(main), indent, line_number=number_line)
        if len(comment_parts) == 2:
            add(notation(comment_parts[1]), indent, "comment")

    height = y + 20
    fig.set_size_inches(WIDTH / 72, height / 72)

    def rule(y_pos, weight):
        fig.add_artist(plt.Line2D([MARGIN / WIDTH, 1 - MARGIN / WIDTH],
                                 [1 - y_pos / height] * 2, transform=fig.transFigure,
                                 color="#151515", linewidth=weight))

    rule(18, 1.35)
    fig.text(MARGIN / WIDTH, 1 - 34 / height, f"Algorithm {number}   {spec.title}",
             fontproperties=properties("bold", 14), va="center", color="#111111")
    rule(47, 0.65)
    if body_top is not None:
        rule(body_top, 0.45)
    rule(height - 12, 1.1)

    bounds = []
    for x, row_y, text, weight, color, ln, indent in rows:
        # Bold the control word independently, preserving italic math notation.
        keyword = re.match(r"^(repeat|for|if|else if|else|break)(?=\b|:)", text)
        segments = [(text, weight)]
        if keyword:
            segments = [(keyword.group(), "bold"), (text[keyword.end():], "normal")]
        cursor = x
        for segment, segment_weight in segments:
            if not segment:
                continue
            for part in text_runs(segment):
                fig.text(cursor / WIDTH, 1 - row_y / height, part,
                         fontproperties=properties(segment_weight), va="center", color=color,
                         parse_math=part.startswith("$"))
                cursor += measure(part, segment_weight)[0]
        if cursor > WIDTH - MARGIN + 0.5:
            raise ValueError(f"Overflow after control-word styling: {spec.slug}: {text}")
        if ln is not None:
            fig.text((BODY_X - 12) / WIDTH, 1 - row_y / height, str(ln),
                     fontproperties=properties(size=9.5), ha="right", va="center", color="#666666")
        if indent:
            for level in range(indent):
                guide_x = BODY_X + level * INDENT + 6
                fig.add_artist(plt.Line2D([guide_x / WIDTH] * 2,
                    [1 - (row_y - 8) / height, 1 - (row_y + 8) / height],
                    transform=fig.transFigure, color="#b9b9b9", linewidth=0.4))
        bounds.append({"line": ln, "x": round(x, 2), "y": round(row_y, 2), "right": round(cursor, 2)})

    fig.savefig(OUT / f"{spec.slug}.svg", metadata={"Date": None, "Creator": "RobotRL algorithm renderer"})
    plt.close(fig)
    return {"algorithm": number, "section": spec.section, "slug": spec.slug,
            "width_pt": WIDTH, "height_pt": round(height, 2), "numbered_lines": number_line,
            "rendered_rows": len(rows), "row_bounds": bounds}



def main():
    sources = json.loads((OUT / "source.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    reports = []
    for number, spec in enumerate(SPECS, 1):
        source = sources[spec.slug]
        reports.append(render(spec, number, source))
        print(f"Rendered Algorithm {number}: {spec.title}")
    (OUT / "layout.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
