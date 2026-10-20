"""Draw docs/architecture.png.  Run:  python docs/make_architecture_diagram.py"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

matplotlib.use("Agg")

# (text, x, y, colour); x and y are positions between 0 and 1
BOXES = [
    ("UCI dataset\n(download_data.py)", 0.12, 0.80, "#e9c46a"),                         # 0
    ("Clean + EDA\n(data.py, eda.py)", 0.37, 0.80, "#e9c46a"),                          # 1
    ("Feature pipeline\n(features.py)", 0.62, 0.80, "#e9c46a"),                         # 2
    ("Train + tune 3 models\n(train.py)", 0.87, 0.80, "#e9c46a"),                       # 3
    ("MLflow tracking\nparams, metrics, plots", 0.87, 0.50, "#a8dadc"),                 # 4
    ("Saved model\nmodel.joblib + MLflow", 0.62, 0.50, "#a8dadc"),                      # 5
    ("GitHub Actions CI/CD\nlint > test > train > docker", 0.37, 0.50, "#f4a261"),      # 6
    ("Docker image\nFastAPI /predict", 0.12, 0.50, "#f4a261"),                          # 7
    ("Client\n(curl / Swagger)", 0.12, 0.20, "#cdb4db"),                                # 8
    ("Kubernetes (Minikube)\n2 pods + LoadBalancer + Ingress", 0.37, 0.20, "#90be6d"),  # 9
    ("Prometheus\n(reads /metrics)", 0.62, 0.20, "#90be6d"),                            # 10
    ("Grafana dashboard\n+ JSON logs", 0.87, 0.20, "#90be6d"),                          # 11
]
ARROWS = [(0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (5, 6), (6, 7), (7, 9), (8, 9), (9, 10), (10, 11)]
WIDTH, HEIGHT = 0.21, 0.17      # size of every box


def edge_point(cx, cy, tx, ty):
    """Where the line from the centre of a box towards (tx, ty) leaves the box."""
    dx, dy = tx - cx, ty - cy
    scale = min((WIDTH / 2) / abs(dx) if dx else 9e9, (HEIGHT / 2) / abs(dy) if dy else 9e9)
    return cx + dx * scale, cy + dy * scale


fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

for start, end in ARROWS:
    (x1, y1), (x2, y2) = BOXES[start][1:3], BOXES[end][1:3]
    ax.annotate("", xy=edge_point(x2, y2, x1, y1), xytext=edge_point(x1, y1, x2, y2),
                arrowprops={"arrowstyle": "-|>", "color": "#264653", "lw": 1.5})

for text, x, y, colour in BOXES:
    ax.add_patch(FancyBboxPatch((x - WIDTH / 2, y - HEIGHT / 2), WIDTH, HEIGHT, boxstyle="round,pad=0.01",
                                facecolor=colour, edgecolor="#264653", linewidth=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=9.5)

ax.set_title("Heart Disease MLOps - End-to-End Architecture", fontsize=14, weight="bold")
fig.tight_layout()
fig.savefig("docs/architecture.png", dpi=150)
print("saved docs/architecture.png")
