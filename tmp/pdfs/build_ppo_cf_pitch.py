from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path("/Users/charithapalika/Desktop/Personal Projects/PPO_CF")
OUT = ROOT / "output" / "pdf" / "ppo_cf_pitch_current_results.pdf"

FIGURES = [
    (
        "RedBlueDoors-6x6",
        Path("/var/folders/9v/hs9qlpdx4791kpqgpt09stch0000gn/T/codex-clipboard-55cabc85-e647-4300-876e-66fb8ebd13e8.png"),
        "All-action CF reaches the strongest final return; queried perturbation is faster early, but is overtaken.",
    ),
    (
        "UnlockPickup",
        Path("/var/folders/9v/hs9qlpdx4791kpqgpt09stch0000gn/T/codex-clipboard-1c351b4d-92c0-4b1a-ba0e-e576f019718a.png"),
        "Queried perturbation is the most sample-efficient; all methods eventually cluster near the same final return.",
    ),
    (
        "Taxi-v4",
        Path("/var/folders/9v/hs9qlpdx4791kpqgpt09stch0000gn/T/codex-clipboard-08893f39-8453-476d-a070-f1e6eff56888.png"),
        "All-action CF has the best final return; queried perturbation helps; distillation is not reliable here.",
    ),
    (
        "DoorKey-6x6",
        Path("/var/folders/9v/hs9qlpdx4791kpqgpt09stch0000gn/T/codex-clipboard-647a5f08-006a-4d53-82ff-3bb8bee145ab.png"),
        "All-action CF is clearly faster and slightly higher final; distillation and queried perturbation beat or match GAE.",
    ),
]


def paragraph_styles():
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#18212f"),
            spaceAfter=12,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#455468"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=18,
            textColor=colors.HexColor("#172033"),
            spaceBefore=12,
            spaceAfter=7,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#24324a"),
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=colors.HexColor("#475569"),
            spaceAfter=4,
        ),
        "math": ParagraphStyle(
            "Math",
            parent=base["BodyText"],
            fontName="Courier",
            fontSize=8.7,
            leading=12,
            textColor=colors.HexColor("#111827"),
            backColor=colors.HexColor("#f5f7fb"),
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8.2,
            leading=10.5,
            textColor=colors.HexColor("#475569"),
            alignment=TA_LEFT,
            spaceBefore=5,
            spaceAfter=8,
        ),
    }


def p(text: str, style: str, styles):
    return Paragraph(text, styles[style])


def make_table(data, col_widths, header=True, font_size=8.2):
    cell = ParagraphStyle(
        "TableCell",
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 2.3,
        textColor=colors.HexColor("#111827"),
    )
    head = ParagraphStyle(
        "TableHead",
        parent=cell,
        fontName="Helvetica-Bold",
    )
    wrapped = []
    for r, row in enumerate(data):
        style = head if header and r == 0 else cell
        wrapped.append([Paragraph(str(value), style) for value in row])
    table = Table(wrapped, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef8") if header else colors.white),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold" if header else "Helvetica"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def scaled_image(path: Path, max_w: float, max_h: float):
    with PILImage.open(path) as im:
        w, h = im.size
    scale = min(max_w / w, max_h / h)
    return Image(str(path), width=w * scale, height=h * scale)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.drawString(0.72 * inch, 0.45 * inch, "PPO-CF pitch memo - current internal results")
    canvas.drawRightString(7.78 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = paragraph_styles()
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=letter,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.65 * inch,
        bottomMargin=0.7 * inch,
        title="Counterfactual Action Landscapes for PPO",
        author="Codex",
    )

    story = []
    story.append(p("Counterfactual Action Landscapes for PPO", "title", styles))
    story.append(
        p(
            "A critical pitch memo on why the injection interface matters, and what the current four-environment results support.",
            "subtitle",
            styles,
        )
    )

    story.append(p("One-line pitch", "h1", styles))
    story.append(
        p(
            "Counterfactual advantages are not plug-and-play in PPO: adding them before clipping and adding them as a separate loss coincide only in PPO's locally linear regime, and this distinction explains why all-action CF losses, sampled perturbations, and distilled perturbations produce different learning behavior.",
            "body",
            styles,
        )
    )

    story.append(p("Why this is important", "h1", styles))
    story.append(
        p(
            "Counterfactual credit assignment is already well established. The useful pitch is narrower: once a counterfactual action landscape is available, modern clipped policy optimization still has an unresolved interface problem. PPO is not a raw linear policy-gradient estimator; clipping, normalization, minibatching, and gradient-scale effects decide whether the same CF signal helps, washes out, or changes the effective trust region.",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "This reframes the contribution from 'we use counterfactuals' to 'we identify and test optimizer-facing interfaces for counterfactual credit in clipped PPO.' That is the part reviewers are less likely to dismiss as already known.",
            "body",
            styles,
        )
    )

    story.append(p("Core mathematical distinction", "h1", styles))
    story.append(
        p(
            "Let r_t(theta) = pi_theta(a_t|s_t) / pi_old(a_t|s_t). Define the policy-centered counterfactual advantage:",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "A_CF(s_t,a) = Q_CF(s_t,a) - sum_b pi_old(b|s_t) Q_CF(s_t,b)",
            "math",
            styles,
        )
    )
    story.append(
        p(
            "Advantage perturbation forms a single scalar advantage before PPO clipping:",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "A_tilde_t = A_GAE_t + beta A_CF(s_t,a_t)",
            "math",
            styles,
        )
    )
    story.append(
        p(
            "and optimizes:",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "L_perturb(theta) = -E_t[min(r_t A_tilde_t, clip(r_t,1-eps,1+eps) A_tilde_t)]",
            "math",
            styles,
        )
    )
    story.append(
        p(
            "If clipping is inactive, or if both terms remain on the same clipping branch, this behaves approximately like L_GAE + beta L_CF. But in real PPO the clipping map is nonlinear, so generally:",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "L_PPO(A_GAE + beta A_CF) != L_PPO(A_GAE) + beta L_PPO(A_CF)",
            "math",
            styles,
        )
    )
    story.append(
        p(
            "The perturbation can change the sign, magnitude, and active clipping branch of the PPO update. This is the central pitch: the counterfactual estimator and the PPO interface cannot be evaluated independently.",
            "body",
            styles,
        )
    )

    story.append(p("Current empirical readout", "h1", styles))
    result_table = [
        ["Environment", "Best from current curves", "Worst", "Interpretation"],
        [
            "RedBlueDoors-6x6",
            "cf_all_action",
            "gae",
            "Queried perturbation learns earlier, but all-action CF finishes clearly highest.",
        ],
        [
            "UnlockPickup",
            "cf_queried_perturb",
            "gae",
            "Perturbation is strongly sample-efficient; all methods converge near similar final return.",
        ],
        [
            "Taxi-v4",
            "cf_all_action",
            "landscape_distill",
            "All-action CF has best final return; distillation is weak here.",
        ],
        [
            "DoorKey-6x6",
            "cf_all_action",
            "gae",
            "All-action CF is faster and slightly higher final; both perturbation and distillation beat or match GAE.",
        ],
    ]
    story.append(make_table(result_table, [1.18 * inch, 1.35 * inch, 1.0 * inch, 3.55 * inch]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(
        p(
            "Conservative claim: across these four environments, counterfactual information usually helps over vanilla GAE, but the best injection mechanism changes by task and metric. All-action CF is strongest overall for final performance; queried perturbation can be better for early learning.",
            "body",
            styles,
        )
    )

    story.append(p("What this supports - and what it does not", "h1", styles))
    claim_table = [
        ["Claim", "Status", "Reason"],
        [
            "CF perturbation improves over vanilla GAE",
            "Supported cautiously",
            "It matches or beats GAE in the shown curves, especially UnlockPickup and RedBlueDoors.",
        ],
        [
            "All-action CF is best overall",
            "Mostly supported",
            "It wins final performance on RedBlueDoors, Taxi, and DoorKey; UnlockPickup favors perturbation for speed.",
        ],
        [
            "Advantage perturbation equals a separate CF loss",
            "Not supported",
            "They coincide only in PPO's locally linear regime; clipping breaks additivity.",
        ],
        [
            "Distillation solves the oracle cost",
            "Not yet",
            "Mixed results: useful on some MiniGrid tasks, weak on Taxi and RedBlueDoors.",
        ],
        [
            "PPO-CF is a universal new SOTA algorithm",
            "Do not claim",
            "The evidence is too small, too mixed, and not yet compute-normalized.",
        ],
    ]
    story.append(make_table(claim_table, [2.0 * inch, 1.15 * inch, 3.93 * inch]))

    story.append(PageBreak())
    story.append(p("ICLR-facing narrative", "h1", styles))
    narrative = [
        [
            "Problem",
            "Counterfactual action landscapes provide richer per-state credit than sampled trajectories, but PPO's clipped objective creates multiple non-equivalent ways to use them.",
        ],
        [
            "Hypothesis",
            "The optimizer-facing interface is as important as the counterfactual estimator.",
        ],
        [
            "Method",
            "Compare all-action CF surrogate, queried sampled advantage perturbation, and learned/distilled perturbation under matched environments.",
        ],
        [
            "Finding",
            "All-action CF gives the most consistent final gains; sampled perturbation can improve early learning; distillation is promising but unreliable.",
        ],
        [
            "Message",
            "Counterfactual credit is useful, but not plug-and-play. In PPO, where the signal enters the clipping operation changes the algorithm.",
        ],
    ]
    story.append(make_table(narrative, [1.25 * inch, 5.83 * inch], header=False))

    story.append(p("What must be added before submission", "h1", styles))
    must_table = [
        ["Needed evidence", "Why reviewers will ask for it"],
        ["Multiple seeds with confidence intervals", "The current curves alone are not statistically enough."],
        ["AUC, time-to-threshold, and final return", "Different algorithms win on different metrics."],
        ["Compute-normalized comparison", "All-action CF is expensive; sample efficiency alone is not the whole cost."],
        ["Matched KL or target-KL runs", "Rules out the possibility that CF simply changes effective step size."],
        ["Shuffled CF control", "Tests whether the gain is state-local counterfactual information rather than extra gradient structure."],
        ["Beta/alpha sweeps", "Avoids a hyperparameter cherry-picking criticism."],
        ["Clipping diagnostics", "This is central to the pitch, so measure branch changes and clip fraction changes directly."],
    ]
    story.append(make_table(must_table, [2.2 * inch, 4.88 * inch]))

    story.append(p("Recommended paper title options", "h1", styles))
    title_table = [
        ["Option", "Comment"],
        ["Counterfactual Action Landscapes for Clipped Policy Optimization", "Best technical framing."],
        ["Where Should Counterfactual Credit Enter PPO?", "Best question-driven framing."],
        ["Counterfactual Credit Is Not Plug-and-Play in PPO", "Best provocative framing, but maybe less formal."],
    ]
    story.append(make_table(title_table, [2.9 * inch, 4.18 * inch]))

    story.append(PageBreak())
    story.append(p("Attached current curves", "h1", styles))
    story.append(
        p(
            "These figures are the current visual evidence from the four environments. They should be treated as provisional until backed by seed-level confidence intervals and scalar summary metrics.",
            "body",
            styles,
        )
    )

    for i, (name, path, caption) in enumerate(FIGURES):
        if i and i % 2 == 0:
            story.append(PageBreak())
        story.append(p(name, "h2", styles))
        story.append(scaled_image(path, 6.95 * inch, 2.75 * inch))
        story.append(p(caption, "caption", styles))

    story.append(PageBreak())
    story.append(p("Bottom line", "h1", styles))
    story.append(
        p(
            "The strongest pitch is not that PPO-CF universally beats PPO. The strongest pitch is that counterfactual action information changes PPO in interface-dependent ways. In the locally linear regime, perturbing the advantage can look like adding a CF loss. Under clipping, it becomes a distinct algorithm. Your current results are confusing only under the old framing; under this framing, the confusion is the evidence.",
            "body",
            styles,
        )
    )
    story.append(
        p(
            "Recommended claim: counterfactual landscapes provide useful policy-gradient information, but in PPO the clipping interface determines whether that information appears as variance reduction, advantage correction, step-size change, or an unstable/distilled guide.",
            "body",
            styles,
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    build()
    print(OUT)
