"""Okabe–Ito colorblind-safe palette, with semantic bindings for the paper.

One palette, one meaning for every color, reused across every figure — so
the reader learns the encoding once. Grayscale safe (differentiable when
the PDF is photocopied in black-and-white) thanks to marker-shape
redundancy baked into the plot helpers.

Reference: Okabe & Ito 2002, *Color Universal Design*. This is the palette
the physicist community has largely converged to for colorblind-safe
publication figures.
"""
from __future__ import annotations

# Okabe-Ito base (hex).
BLACK      = "#000000"
ORANGE     = "#E69F00"
SKY_BLUE   = "#56B4E9"
GREEN      = "#009E73"
YELLOW     = "#F0E442"
BLUE       = "#0072B2"
VERMILLION = "#D55E00"
PURPLE     = "#CC79A7"
GREY       = "#7F7F7F"

# Semantic role -> hex. Binding is fixed across the entire paper so a
# reader who learns "blue = Nash" on Fig. 1 re-applies that mapping on
# every subsequent figure.
NASH        = BLUE
SA_DQAS     = ORANGE
HEA         = GREY
COLD_START  = GREY
REFERENCE   = BLACK        # HF / FCI / exact / K_n optimum lines

# Four-player color mapping.
PLAYER_ANTI_BP      = SKY_BLUE
PLAYER_ANTI_SIM     = GREEN
PLAYER_PERFORMANCE  = BLUE
PLAYER_HARDWARE     = VERMILLION

# Marker shapes that pair with the above (redundancy against colorblind OR
# grayscale printing).
MARKER_NASH        = "o"
MARKER_SA_DQAS     = "^"
MARKER_HEA         = "s"   # square, open face
MARKER_COLD        = "D"   # diamond, open face

PLAYER_COLORS = {
    "anti_bp":     PLAYER_ANTI_BP,
    "anti_sim":    PLAYER_ANTI_SIM,
    "performance": PLAYER_PERFORMANCE,
    "hardware":    PLAYER_HARDWARE,
}

PLAYER_LABELS = {
    "anti_bp":     r"$f_1$ anti-BP",
    "anti_sim":    r"$f_2$ anti-sim",
    "performance": r"$f_3$ performance",
    "hardware":    r"$f_4$ hardware",
}


__all__ = [
    "BLACK", "ORANGE", "SKY_BLUE", "GREEN", "YELLOW", "BLUE",
    "VERMILLION", "PURPLE", "GREY",
    "NASH", "SA_DQAS", "HEA", "COLD_START", "REFERENCE",
    "PLAYER_ANTI_BP", "PLAYER_ANTI_SIM",
    "PLAYER_PERFORMANCE", "PLAYER_HARDWARE",
    "PLAYER_COLORS", "PLAYER_LABELS",
    "MARKER_NASH", "MARKER_SA_DQAS", "MARKER_HEA", "MARKER_COLD",
]
