"""One figure style: SimSun text, real XeLaTeX math, vector PDF output."""
from __future__ import annotations

from pathlib import Path
import matplotlib

matplotlib.use("pgf")
from matplotlib import pyplot as plt

SIMSUN = Path("C:/Windows/Fonts/simsun.ttc")
if not SIMSUN.is_file():
    raise FileNotFoundError("SimSun / 宋体 font is required: " + str(SIMSUN))

matplotlib.rcParams.update({
    "pgf.texsystem": "xelatex",
    "pgf.rcfonts": False,
    "pgf.preamble": (
        r"\usepackage{xeCJK}"
        r"\setmainfont{SimSun}"
        r"\setCJKmainfont{SimSun}"
        r"\usepackage{amsmath}"
    ),
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["SimSun"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.unicode_minus": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

BLUE = "#245C85"
ORANGE = "#B86736"
TEAL = "#17816F"
GREY = "#65747B"


def save_pdf(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
