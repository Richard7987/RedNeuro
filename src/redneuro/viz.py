import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from redneuro.model import forward_prop

FEATURES = ["ip_mean", "ip_std", "ip_kurt", "ip_skew",
            "dm_mean", "dm_std", "dm_kurt", "dm_skew"]
POSITIVO, NEGATIVO = "#2a78d6", "#eb6834"   # empuja hacia púlsar / hacia no púlsar
CORRECTO, INCORRECTO = "#008300", "#c62828"


def layer_positions(n, x):
    ys = np.linspace(-1, 1, n) if n > 1 else np.array([0.0])
    return np.column_stack([np.full(n, x, dtype=float), ys])


def draw_edges(ax, p_from, p_to, W, a_from, activo):
    """Dibuja las conexiones entre dos capas.

    W[j, i] conecta el nodo i de p_from con el nodo j de p_to.
    Si activo=False, las líneas se dibujan tenues y su grosor refleja el peso |W|.
    Si activo=True, cada línea refleja la señal que lleva para ESTE ejemplo:
    contribución = W[j, i] * a_from[i]. Grosor y opacidad crecen con |contribución|,
    el color indica su signo. Una entrada en cero o una neurona apagada no envía señal,
    así que sus líneas quedan casi invisibles.
    """
    segs = [(p_from[i], p_to[j]) for j in range(W.shape[0]) for i in range(W.shape[1])]
    if activo:
        contrib = (W * a_from.reshape(1, -1)).ravel()
        fuerza = np.abs(contrib) / (np.abs(contrib).max() + 1e-12)
        base = np.where(contrib > 0, POSITIVO, NEGATIVO)
        widths = 0.2 + 4.0 * fuerza
        alphas = 0.05 + 0.95 * fuerza
    else:
        w = W.ravel()
        base = np.where(w > 0, POSITIVO, NEGATIVO)
        widths = 0.3 + 2.0 * np.abs(w) / (np.abs(W).max() + 1e-12)
        alphas = np.full(w.size, 0.08)
    colors = [to_rgba(c, alpha=float(a)) for c, a in zip(base, alphas)]
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=1))


def _draw_nodes(ax, pos, valores, size, cmap, vmin, vmax):
    """Círculos de una capa. valores=None significa capa todavía apagada (blanca)."""
    if valores is None:
        ax.scatter(pos[:, 0], pos[:, 1], s=size, color="white", edgecolors="black", zorder=2)
    else:
        ax.scatter(pos[:, 0], pos[:, 1], s=size, c=valores, cmap=cmap, vmin=vmin, vmax=vmax,
                   edgecolors="black", zorder=2)


def draw_network(ax, params, x, y_true, fase=4, threshold=0.5):
    """Dibuja la red procesando un solo ejemplo x de shape (n_x, 1).

    fase 1: entra el dato (se pintan las entradas).
    fase 2: capa oculta (se encienden las líneas entrada→oculta y las neuronas ocultas).
    fase 3: salida (se encienden las líneas oculta→salida y el nodo de salida, con A2).
    fase 4: veredicto (púlsar / no púlsar, y si coincide con la etiqueta real).
    """
    W1, b1, W2, b2 = params
    _, A1, _, A2 = forward_prop(W1, b1, W2, b2, x)
    p_in = layer_positions(W1.shape[1], 0)
    p_h = layer_positions(W1.shape[0], 1)
    p_out = layer_positions(1, 2)

    ax.clear()
    ax.set_axis_off()
    ax.set_xlim(-0.9, 2.7)
    ax.set_ylim(-1.45, 1.4)

    draw_edges(ax, p_in, p_h, W1, x.ravel(), activo=fase >= 2)
    draw_edges(ax, p_h, p_out, W2, A1.ravel(), activo=fase >= 3)

    _draw_nodes(ax, p_in, x.ravel() if fase >= 1 else None, 300, "RdBu_r", -3, 3)
    _draw_nodes(ax, p_h, A1.ravel() if fase >= 2 else None, 300, "Blues", 0, max(A1.max(), 1.0))
    _draw_nodes(ax, p_out, A2.ravel() if fase >= 3 else None, 700, "Blues", 0, 1)

    for (px, py), name, val in zip(p_in, FEATURES, x.ravel()):
        ax.text(px - 0.1, py, f"{name} {val:+.2f}", ha="right", va="center", fontsize=8)

    ax.text(0, 1.22, "entrada (8 features\nnormalizadas)", ha="center", fontsize=8, color="gray")
    ax.text(1, 1.22, f"capa oculta ({W1.shape[0]} ReLU)", ha="center", fontsize=8, color="gray")
    ax.text(2, 1.22, "salida\nsigmoid = P(púlsar)", ha="center", fontsize=8, color="gray")
    ax.text(1, -1.2, "líneas: azul empuja hacia púlsar, naranja hacia no púlsar; "
                     "grosor = cuánta señal lleva para este candidato",
            ha="center", fontsize=8, color="gray")

    if fase >= 3:
        ax.text(2.15, 0, f"A2 = {A2.item():.3f}", va="center", fontsize=11)
    if fase >= 4:
        pred = int(A2.item() > threshold)
        real = int(y_true)
        veredicto = "PÚLSAR" if pred == 1 else "NO PÚLSAR"
        etiqueta = "púlsar" if real == 1 else "no púlsar"
        acierto = pred == real
        ax.text(1, -1.38,
                f"veredicto: {veredicto}    real: {etiqueta}    "
                f"{'✓ correcto' if acierto else '✗ incorrecto'}",
                ha="center", va="center", fontsize=14, fontweight="bold",
                color=CORRECTO if acierto else INCORRECTO)
    return A2.item()


def draw_confusion_bars(ax, tp, fp, fn, tn):
    """Panel lateral para el modo inferencia: conteo acumulado de aciertos y errores."""
    ax.clear()
    etiquetas = ["púlsar\nbien (TP)", "falsa\nalarma (FP)", "púlsar\nperdido (FN)", "no púlsar\nbien (TN)"]
    valores = [tp, fp, fn, tn]
    colores = [CORRECTO, INCORRECTO, INCORRECTO, CORRECTO]
    barras = ax.bar(etiquetas, valores, color=colores, width=0.6)
    for b, v in zip(barras, valores):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, str(v),
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(valores + [1]) * 1.25)
    ax.set_title("candidatos procesados: acumulado")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=8)
