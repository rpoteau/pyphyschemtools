############################################################
#                  Molecular symmetry
############################################################
"""
Molecular symmetry for teaching purposes.

Symmetry operations are written as 3x3 matrices and applied to the atomic
coordinates of a molecule. Tracking where each atom goes after an operation
gives the characters of the atomic representation Γ_atoms, which can then be
reduced with the character table of the point group.
"""
import re
import numpy as np
import pandas as pd
from IPython.display import display

############################################################
#            Symmetry operations as 3x3 matrices
############################################################
_AXES = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}
_PLANES = {"xy": (0, 0, 1), "yx": (0, 0, 1), "xz": (0, 1, 0), "zx": (0, 1, 0), "yz": (1, 0, 0), "zy": (1, 0, 0)}

def _unit(axis):
    if isinstance(axis, str):
        if axis not in _AXES:
            raise ValueError(f"Unknown axis '{axis}': use 'x', 'y', 'z' or a vector, e.g. (1, 1, 0)")
        axis = _AXES[axis]
    v = np.asarray(axis, dtype=float)
    return v / np.linalg.norm(v)

def identity():
    """Identity E."""
    return np.eye(3)

def inversion():
    """Inversion i through the origin."""
    return -np.eye(3)

def Cn(n, axis="z", k=1):
    """
    Proper rotation C_n^k about an axis passing through the origin.

    Args:
        n (int): order of the axis (rotation angle 360°/n).
        axis (str or array-like): "x", "y", "z" (default) or a direction, e.g. (1, 1, 1).
        k (int): power of the operation (rotation angle k*360°/n).

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    u = _unit(axis)
    theta = 2 * np.pi * k / n
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K

def sigma(plane="xy"):
    """
    Reflection σ through a plane containing the origin.

    Args:
        plane (str or array-like): "xy", "xz" or "yz", or, for any other plane,
            the vector normal to the plane, e.g. (1, -1, 0) for the diagonal
            plane containing z and the bisector of x and y (σd of CH4, D4h...).

    Returns:
        np.ndarray: 3x3 reflection matrix.

    Examples:
        >>> sigma("xy")          # σh when the principal axis is z
        >>> sigma((1, -1, 0))    # σd containing z and the direction (1, 1, 0)
    """
    if isinstance(plane, str):
        if plane not in _PLANES:
            raise ValueError(f"Unknown plane '{plane}': use 'xy', 'xz', 'yz' or a normal vector, e.g. (1, -1, 0)")
        plane = _PLANES[plane]
    u = _unit(plane)
    return np.eye(3) - 2 * np.outer(u, u)

def Sn(n, axis="z", k=1):
    """
    Improper rotation S_n^k: rotation C_n^k followed by a reflection
    through the plane perpendicular to the axis.

    Args:
        n (int): order of the axis.
        axis (str or array-like): "x", "y", "z" (default) or a direction, e.g. (1, 1, 1).
        k (int): power of the operation.
    """
    return sigma(_unit(axis)) @ Cn(n, axis, k)

def inv(M):
    """
    Inverse of a symmetry operation. The matrix is orthogonal, so its inverse
    is its transpose: inv(Cn(3)) = C3⁻¹ = C3².
    """
    return np.asarray(M, dtype=float).T.copy()

_SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def _angle(phi):
    """Angle in degrees, in [0, 180[, written without useless decimals."""
    phi = float(np.degrees(phi)) % 180
    if abs(phi - 180) < 1e-6:
        phi = 0.0
    return f"{int(round(phi))}°" if abs(phi - round(phi)) < 1e-3 else f"{phi:.1f}°"

def _axis_name(u):
    """
    Name of an axis: 'x', 'y', 'z'; 'xy,60°' for an axis in the xy plane at 60° from x;
    otherwise the direction, e.g. '1,1,1'.
    """
    for name, e in zip("xyz", np.eye(3)):
        if abs(abs(u @ e) - 1) < 1e-6:
            return name
    if abs(u[2]) < 1e-6:
        return "xy," + _angle(np.arctan2(u[1], u[0]))
    v = u / min(abs(c) for c in u if abs(c) > 1e-6)
    if np.allclose(v, np.round(v), atol=1e-3):
        return ",".join(str(int(round(c))) for c in v)
    return ",".join(f"{c:.3f}" for c in np.round(u, 3) + 0.0)

def identify(M, tol=1e-6, with_inverse=False):
    """
    Name of a symmetry operation given as a 3x3 matrix: E, i, σ(xy), C3(z),
    C3²(z), S4³(z), C3(1,1,1)...

    Vertical planes and axes lying in the xy plane are located by the angle φ
    of their direction in the xy plane, measured from x:
    σ(z,60°) is the plane containing z and the direction at 60° from x
    (σ(z,0°) = σ(xz), σ(z,90°) = σ(yz)); C2(xy,30°) is the C2 axis in the xy
    plane at 30° from x. Other planes are given by their normal: σ⊥(1,1,1).

    The direction of an axis is oriented so that its first non-zero component
    is positive, and the rotation angle follows the right-hand rule.

    Args:
        M (np.ndarray): 3x3 matrix.
        with_inverse (bool): if True, also gives the inverse notation when the
            operation is the inverse of the generator: 'C3²(z) = C3⁻¹(z)'.

    Examples:
        >>> identify(Cn(3) @ Cn(3))
        'C3²(z)'
        >>> identify(sigma((0, 0, 1)) @ Cn(2))
        'i'
    """
    M = np.asarray(M, dtype=float)
    if not np.allclose(M @ M.T, np.eye(3), atol=1e-6):
        raise ValueError("not an orthogonal matrix")
    if np.allclose(M, np.eye(3), atol=tol):
        return "E"
    if np.allclose(M, -np.eye(3), atol=tol):
        return "i"
    det = np.linalg.det(M)
    w, v = np.linalg.eig(M)
    u = np.real(v[:, int(np.argmin(np.abs(w - np.sign(det))))])
    u /= np.linalg.norm(u)
    u *= np.sign(next(c for c in u if abs(c) > 1e-6))
    if det < 0 and np.isclose(np.trace(M), 1):
        planes = {"z": "xy", "y": "xz", "x": "yz"}
        name = _axis_name(u)
        if name in planes:
            return f"σ({planes[name]})"
        if abs(u[2]) < 1e-6:
            # vertical plane: contains z and the direction of the xy plane at angle φ from x
            return "σ(z," + _angle(np.arctan2(u[0], -u[1])) + ")"
        return f"σ⊥({name})"
    R = M if det > 0 else sigma(u) @ M          # proper part
    K = (R - R.T) / 2
    sin = np.array([K[2, 1], K[0, 2], K[1, 0]]) @ u
    cos = (np.trace(R) - 1) / 2
    theta = np.arctan2(sin, cos) % (2 * np.pi)
    for n in range(2, 25):
        ks = range(1, n) if det > 0 else range(1, 2 * n, 2)
        for k in ks:
            if np.gcd(k, n) == 1 and abs((theta - 2 * np.pi * k / n + np.pi) % (2 * np.pi) - np.pi) < 1e-6:
                sym = "C" if det > 0 else "S"
                power = "" if k == 1 else str(k).translate(_SUP)
                name = f"{sym}{n}{power}({_axis_name(u)})"
                order = n if (det > 0 or n % 2 == 0) else 2 * n
                if with_inverse and k > 1 and k == order - 1:
                    name += f" = {sym}{n}⁻¹({_axis_name(u)})"
                return name
    return "?"

def _latex_entry(x, exact=True):
    """Matrix entry in LaTeX: exact form (1/2, √3/2, (√5-1)/4...) when it is simple, else decimal."""
    x = float(x)
    if abs(x) < 1e-9:
        return "0"
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    if exact:
        try:
            import sympy as sp
            e = sp.nsimplify(x, [sp.sqrt(2), sp.sqrt(3), sp.sqrt(5)], tolerance=1e-9)
            tex = sp.latex(e)
            if abs(float(e) - x) < 1e-8 and len(tex) <= 32:
                return tex
        except Exception:
            pass
    return f"{x:.3f}"

def _latex_label(label):
    """Operation label in LaTeX: C3²(z) -> C_{3}^{2}(z), C3⁻¹ -> C_{3}^{-1}, σv'(yz) -> \\sigma_{v}'(yz)..."""
    label = label.replace("°", "^{\\circ}")
    if " = " in label:
        return " = ".join(_latex_label(part) for part in label.split(" = "))
    sup = {c: str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")}
    sup["⁻"] = "-"
    m = re.match(r"^([CS])(\d+)(['′]*)([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]*)(.*)$", label)
    if m:
        power = "".join(sup[c] for c in m.group(4))
        tex = f"{m.group(1)}_{{{m.group(2)}}}{m.group(3)}"
        if power:
            tex += f"^{{{power}}}"
        return tex + m.group(5)
    m = re.match(r"^σ([vhd]?)(['′]*)(.*)$", label)
    if m:
        tex = r"\sigma" + (f"_{{{m.group(1)}}}" if m.group(1) else "") + m.group(2)
        return tex + m.group(3).replace("⊥", r"_{\perp}")
    return label

def show_matrix(M, name=None, exact=True):
    """
    Displays a 3x3 symmetry operation as a LaTeX matrix, with its name.

    Args:
        M (np.ndarray): 3x3 matrix.
        name (str): name of the operation; if None, given by `identify(M)`.
        exact (bool): if True, exact forms of the entries (1/2, √3/2...) when possible.
    """
    from IPython.display import Math
    if name is None:
        name = identify(M, with_inverse=True)
    rows = r" \\ ".join(" & ".join(_latex_entry(x, exact) for x in row) for row in np.asarray(M))
    display(Math(_latex_label(name) + r" = \begin{pmatrix} " + rows + r" \end{pmatrix}"))

############################################################
#                    Character tables
############################################################
def _times_sigma_h(table, improper_labels):
    """
    Builds G x Cs (σh perpendicular to z) from a group G of rotations about z:
    the irreps become ' (same characters) and '' (opposite characters on the
    improper classes).
    """
    sh = sigma((0, 0, 1))
    classes = list(table["classes"])
    for (lab, g, M), new_lab in zip(table["classes"], improper_labels):
        classes.append((new_lab, g, sh @ M))
    irreps = {}
    for name, chi in table["irreps"].items():
        irreps[name + "'"] = list(chi) + list(chi)
    for name, chi in table["irreps"].items():
        irreps[name + "''"] = list(chi) + [-c for c in chi]
    cplx = set()
    for name in table.get("complex", []):
        cplx |= {name + "'", name + "''"}
    rows = {}
    for name, (plus, minus) in table.get("complex_rows", {}).items():
        rows[name + "'"] = (list(plus) + list(plus), list(minus) + list(minus))
        rows[name + "''"] = (list(plus) + [-c for c in plus], list(minus) + [-c for c in minus])
    return {"classes": classes, "irreps": irreps, "complex": cplx,
            "complex_rows": rows, "eps_n": table.get("eps_n")}

# Each class is given by (label, number of operations, representative matrix).
# Conventions: principal axis along z; σv(xz) for C2v; C2' along x for Dnh.
_c30, _s30 = np.cos(np.pi / 6), np.sin(np.pi / 6)

_r2 = np.sqrt(2)
_r3 = np.sqrt(3)
_a = 2 * np.cos(2 * np.pi / 5)   # 2cos72° = 0.618
_b = 2 * np.cos(4 * np.pi / 5)   # 2cos144° = -1.618
_phi = (1 + np.sqrt(5)) / 2

def _times_i(table, improper_labels):
    """
    Builds G x Ci from a group G without inversion: each proper class R gives an
    improper class i.R, the irreps become g (same characters) and u (opposite
    characters on the improper classes).
    """
    classes = list(table["classes"])
    for (lab, g, M), new_lab in zip(table["classes"], improper_labels):
        classes.append((new_lab, g, -M))
    irreps = {}
    for name, chi in table["irreps"].items():
        irreps[name + "g"] = list(chi) + list(chi)
    for name, chi in table["irreps"].items():
        irreps[name + "u"] = list(chi) + [-c for c in chi]
    cplx = set()
    for name in table.get("complex", []):
        cplx |= {name + "g", name + "u"}
    rows = {}
    for name, (plus, minus) in table.get("complex_rows", {}).items():
        rows[name + "g"] = (list(plus) + list(plus), list(minus) + list(minus))
        rows[name + "u"] = (list(plus) + [-c for c in plus], list(minus) + [-c for c in minus])
    return {"classes": classes, "irreps": irreps, "complex": cplx,
            "complex_rows": rows, "eps_n": table.get("eps_n")}

# Each class is given by (label, number of operations, representative matrix).
# "complex" lists the pairs of complex conjugate 1D irreps, given here as a single
# real 2D row (sum of the two characters), as in most textbooks.
# Conventions: principal axis along z; C2v: σv(xz); D groups: one C2' axis along x.
_D3 = {"classes": [("E", 1, identity()), ("C3", 2, Cn(3)), ("C2", 3, Cn(2, (1, 0, 0)))],
       "irreps": {"A1": [1, 1, 1], "A2": [1, 1, -1], "E": [2, -1, 0]}}
_D5 = {"classes": [("E", 1, identity()), ("C5", 2, Cn(5)), ("C5²", 2, Cn(5, k=2)),
                   ("C2", 5, Cn(2, (1, 0, 0)))],
       "irreps": {"A1": [1, 1, 1, 1], "A2": [1, 1, 1, -1],
                  "E1": [2, _a, _b, 0], "E2": [2, _b, _a, 0]}}
_D8 = {"classes": [("E", 1, identity()), ("C8", 2, Cn(8)), ("C8³", 2, Cn(8, k=3)),
                   ("C4", 2, Cn(4)), ("C2", 1, Cn(2)),
                   ("C2'", 4, Cn(2, (1, 0, 0))),
                   ("C2''", 4, Cn(2, (np.cos(np.pi / 8), np.sin(np.pi / 8), 0)))],
       "irreps": {"A1": [1, 1, 1, 1, 1, 1, 1], "A2": [1, 1, 1, 1, 1, -1, -1],
                  "B1": [1, -1, -1, 1, 1, 1, -1], "B2": [1, -1, -1, 1, 1, -1, 1],
                  "E1": [2, _r2, -_r2, 0, -2, 0, 0], "E2": [2, 0, 0, -2, 2, 0, 0],
                  "E3": [2, -_r2, _r2, 0, -2, 0, 0]}}
_C4 = {"classes": [("E", 1, identity()), ("C4", 1, Cn(4)), ("C2", 1, Cn(2)),
                   ("C4³", 1, Cn(4, k=3))],
       "irreps": {"A": [1, 1, 1, 1], "B": [1, -1, 1, -1], "E": [2, 0, -2, 0]},
       "complex": {"E"}}
_C3 = {"classes": [("E", 1, identity()), ("C3", 1, Cn(3)), ("C3²", 1, Cn(3, k=2))],
       "irreps": {"A": [1, 1, 1], "E": [2, -1, -1]}, "complex": {"E"}}
_C6 = {"classes": [("E", 1, identity()), ("C6", 1, Cn(6)), ("C3", 1, Cn(3)),
                   ("C2", 1, Cn(2)), ("C3²", 1, Cn(3, k=2)), ("C6⁵", 1, Cn(6, k=5))],
       "irreps": {"A": [1, 1, 1, 1, 1, 1], "B": [1, -1, 1, -1, 1, -1],
                  "E1": [2, 1, -1, -2, -1, 1], "E2": [2, -1, -1, 2, -1, -1]},
       "complex": {"E1", "E2"}}
_C5 = {"classes": [("E", 1, identity()), ("C5", 1, Cn(5)), ("C5²", 1, Cn(5, k=2)),
                   ("C5³", 1, Cn(5, k=3)), ("C5⁴", 1, Cn(5, k=4))],
       "irreps": {"A": [1, 1, 1, 1, 1], "E1": [2, _a, _b, _b, _a], "E2": [2, _b, _a, _a, _b]},
       "complex": {"E1", "E2"}}
_T = {"classes": [("E", 1, identity()), ("C3", 4, Cn(3, (1, 1, 1))),
                  ("C3²", 4, Cn(3, (1, 1, 1), k=2)), ("C2", 3, Cn(2))],
      "irreps": {"A": [1, 1, 1, 1], "E": [2, -1, -1, 2], "T": [3, 0, 0, -1]},
      "complex": {"E"},
      "complex_rows": {"E": ([1, np.exp(2j * np.pi / 3), np.exp(-2j * np.pi / 3), 1],
                             [1, np.exp(-2j * np.pi / 3), np.exp(2j * np.pi / 3), 1])},
      "eps_n": 3}
_O = {"classes": [("E", 1, identity()), ("C3", 8, Cn(3, (1, 1, 1))),
                  ("C2'", 6, Cn(2, (1, 1, 0))), ("C4", 6, Cn(4)),
                  ("C2", 3, Cn(2))],
      "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, -1, -1, 1], "E": [2, -1, 0, 0, 2],
                 "T1": [3, 0, -1, 1, -1], "T2": [3, 0, 1, -1, -1]}}
_I = {"classes": [("E", 1, identity()), ("C5", 12, Cn(5, (0, 1, _phi))),
                  ("C5²", 12, Cn(5, (0, 1, _phi), k=2)),
                  ("C3", 20, Cn(3, (1, 1, 1))), ("C2", 15, Cn(2))],
      "irreps": {"A": [1, 1, 1, 1, 1], "T1": [3, -_b, -_a, 0, -1],
                 "T2": [3, -_a, -_b, 0, -1], "G": [4, -1, -1, 1, 0], "H": [5, 0, 0, -1, 1]}}

_TABLES = {
    # ---------------- low symmetry ----------------
    "C1": {"classes": [("E", 1, identity())], "irreps": {"A": [1]}},
    "Cs": {"classes": [("E", 1, identity()), ("σh", 1, sigma((0, 0, 1)))],
           "irreps": {"A'": [1, 1], "A''": [1, -1]}},
    "Ci": {"classes": [("E", 1, identity()), ("i", 1, inversion())],
           "irreps": {"Ag": [1, 1], "Au": [1, -1]}},
    # ---------------- Cn ----------------
    "C2": {"classes": [("E", 1, identity()), ("C2", 1, Cn(2))],
           "irreps": {"A": [1, 1], "B": [1, -1]}},
    "C3": _C3,
    "C4": _C4,
    "C5": _C5,
    "C6": _C6,
    "S8": {"classes": [("E", 1, identity()), ("S8", 1, Sn(8)), ("C4", 1, Cn(4)),
                       ("S8³", 1, Sn(8, k=3)), ("C2", 1, Cn(2)),
                       ("S8⁵", 1, Sn(8, k=5)), ("C4³", 1, Cn(4, k=3)),
                       ("S8⁷", 1, Sn(8, k=7))],
           "irreps": {"A": [1, 1, 1, 1, 1, 1, 1, 1], "B": [1, -1, 1, -1, 1, -1, 1, -1],
                      "E1": [2, _r2, 0, -_r2, -2, -_r2, 0, _r2],
                      "E2": [2, 0, -2, 0, 2, 0, -2, 0],
                      "E3": [2, -_r2, 0, _r2, -2, _r2, 0, -_r2]},
           "complex": {"E1", "E2", "E3"}},
    "S4": {"classes": [("E", 1, identity()), ("S4", 1, Sn(4)), ("C2", 1, Cn(2)),
                       ("S4³", 1, Sn(4, k=3))],
           "irreps": {"A": [1, 1, 1, 1], "B": [1, -1, 1, -1], "E": [2, 0, -2, 0]},
           "complex": {"E"}},
    # ---------------- Cnv ----------------
    "C2v": {"classes": [("E", 1, identity()), ("C2", 1, Cn(2)),
                        ("σv(xz)", 1, sigma((0, 1, 0))), ("σv'(yz)", 1, sigma((1, 0, 0)))],
            "irreps": {"A1": [1, 1, 1, 1], "A2": [1, 1, -1, -1],
                       "B1": [1, -1, 1, -1], "B2": [1, -1, -1, 1]}},
    "C3v": {"classes": [("E", 1, identity()), ("C3", 2, Cn(3)), ("σv", 3, sigma((0, 1, 0)))],
            "irreps": {"A1": [1, 1, 1], "A2": [1, 1, -1], "E": [2, -1, 0]}},
    "C4v": {"classes": [("E", 1, identity()), ("C4", 2, Cn(4)), ("C2", 1, Cn(2)),
                        ("σv", 2, sigma((0, 1, 0))), ("σd", 2, sigma((1, -1, 0)))],
            "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1],
                       "B1": [1, -1, 1, 1, -1], "B2": [1, -1, 1, -1, 1], "E": [2, 0, -2, 0, 0]}},
    "C5v": {"classes": [("E", 1, identity()), ("C5", 2, Cn(5)), ("C5²", 2, Cn(5, k=2)),
                        ("σv", 5, sigma((0, 1, 0)))],
            "irreps": {"A1": [1, 1, 1, 1], "A2": [1, 1, 1, -1],
                       "E1": [2, _a, _b, 0], "E2": [2, _b, _a, 0]}},
    "C6v": {"classes": [("E", 1, identity()), ("C6", 2, Cn(6)), ("C3", 2, Cn(3)),
                        ("C2", 1, Cn(2)), ("σv", 3, sigma((0, 1, 0))),
                        ("σd", 3, sigma((-_s30, _c30, 0)))],
            "irreps": {"A1": [1, 1, 1, 1, 1, 1], "A2": [1, 1, 1, 1, -1, -1],
                       "B1": [1, -1, 1, -1, 1, -1], "B2": [1, -1, 1, -1, -1, 1],
                       "E1": [2, 1, -1, -2, 0, 0], "E2": [2, -1, -1, 2, 0, 0]}},
    # ---------------- Cnh ----------------
    "C2h": {"classes": [("E", 1, identity()), ("C2", 1, Cn(2)),
                        ("i", 1, inversion()), ("σh", 1, sigma((0, 0, 1)))],
            "irreps": {"Ag": [1, 1, 1, 1], "Bg": [1, -1, 1, -1],
                       "Au": [1, 1, -1, -1], "Bu": [1, -1, -1, 1]}},
    "C3h": {"classes": [("E", 1, identity()), ("C3", 1, Cn(3)), ("C3²", 1, Cn(3, k=2)),
                        ("σh", 1, sigma((0, 0, 1))), ("S3", 1, Sn(3)),
                        ("S3⁵", 1, Sn(3, k=5))],
            "irreps": {"A'": [1, 1, 1, 1, 1, 1], "E'": [2, -1, -1, 2, -1, -1],
                       "A''": [1, 1, 1, -1, -1, -1], "E''": [2, -1, -1, -2, 1, 1]},
            "complex": {"E'", "E''"}},
    "C4h": _times_i(_C4, ["i", "S4³", "σh", "S4"]),
    "C5h": _times_sigma_h(_C5, ["σh", "S5", "S5⁷", "S5³", "S5⁹"]),
    "C6h": _times_i(_C6, ["i", "S3⁵", "S6⁵", "σh", "S6", "S3"]),
    "S6": _times_i(_C3, ["i", "S6⁵", "S6"]),
    # ---------------- Dn ----------------
    "D2": {"classes": [("E", 1, identity()), ("C2(z)", 1, Cn(2, (0, 0, 1))),
                       ("C2(y)", 1, Cn(2, (0, 1, 0))), ("C2(x)", 1, Cn(2, (1, 0, 0)))],
           "irreps": {"A": [1, 1, 1, 1], "B1": [1, 1, -1, -1],
                      "B2": [1, -1, 1, -1], "B3": [1, -1, -1, 1]}},
    "D3": _D3,
    "D4": {"classes": [("E", 1, identity()), ("C4", 2, Cn(4)), ("C2", 1, Cn(2)),
                       ("C2'", 2, Cn(2, (1, 0, 0))), ("C2''", 2, Cn(2, (1, 1, 0)))],
           "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1],
                      "B1": [1, -1, 1, 1, -1], "B2": [1, -1, 1, -1, 1], "E": [2, 0, -2, 0, 0]}},
    "D5": _D5,
    "D6": {"classes": [("E", 1, identity()), ("C6", 2, Cn(6)), ("C3", 2, Cn(3)),
                       ("C2", 1, Cn(2)), ("C2'", 3, Cn(2, (1, 0, 0))),
                       ("C2''", 3, Cn(2, (_c30, _s30, 0)))],
           "irreps": {"A1": [1, 1, 1, 1, 1, 1], "A2": [1, 1, 1, 1, -1, -1],
                      "B1": [1, -1, 1, -1, 1, -1], "B2": [1, -1, 1, -1, -1, 1],
                      "E1": [2, 1, -1, -2, 0, 0], "E2": [2, -1, -1, 2, 0, 0]}},
    # ---------------- Dnh ----------------
    "D2h": {"classes": [("E", 1, identity()), ("C2(z)", 1, Cn(2, (0, 0, 1))),
                        ("C2(y)", 1, Cn(2, (0, 1, 0))), ("C2(x)", 1, Cn(2, (1, 0, 0))),
                        ("i", 1, inversion()), ("σ(xy)", 1, sigma((0, 0, 1))),
                        ("σ(xz)", 1, sigma((0, 1, 0))), ("σ(yz)", 1, sigma((1, 0, 0)))],
            "irreps": {"Ag":  [1, 1, 1, 1, 1, 1, 1, 1],
                       "B1g": [1, 1, -1, -1, 1, 1, -1, -1],
                       "B2g": [1, -1, 1, -1, 1, -1, 1, -1],
                       "B3g": [1, -1, -1, 1, 1, -1, -1, 1],
                       "Au":  [1, 1, 1, 1, -1, -1, -1, -1],
                       "B1u": [1, 1, -1, -1, -1, -1, 1, 1],
                       "B2u": [1, -1, 1, -1, -1, 1, -1, 1],
                       "B3u": [1, -1, -1, 1, -1, 1, 1, -1]}},
    "D3h": {"classes": [("E", 1, identity()), ("C3", 2, Cn(3)),
                        ("C2'", 3, Cn(2, (1, 0, 0))), ("σh", 1, sigma((0, 0, 1))),
                        ("S3", 2, Sn(3)), ("σv", 3, sigma((0, 1, 0)))],
            "irreps": {"A1'": [1, 1, 1, 1, 1, 1], "A2'": [1, 1, -1, 1, 1, -1],
                       "E'": [2, -1, 0, 2, -1, 0], "A1''": [1, 1, 1, -1, -1, -1],
                       "A2''": [1, 1, -1, -1, -1, 1], "E''": [2, -1, 0, -2, 1, 0]}},
    "D4h": {"classes": [("E", 1, identity()), ("C4", 2, Cn(4)), ("C2", 1, Cn(2)),
                        ("C2'", 2, Cn(2, (1, 0, 0))), ("C2''", 2, Cn(2, (1, 1, 0))),
                        ("i", 1, inversion()), ("S4", 2, Sn(4)),
                        ("σh", 1, sigma((0, 0, 1))), ("σv", 2, sigma((0, 1, 0))),
                        ("σd", 2, sigma((1, -1, 0)))],
            "irreps": {"A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "A2g": [1, 1, 1, -1, -1, 1, 1, 1, -1, -1],
                       "B1g": [1, -1, 1, 1, -1, 1, -1, 1, 1, -1],
                       "B2g": [1, -1, 1, -1, 1, 1, -1, 1, -1, 1],
                       "Eg":  [2, 0, -2, 0, 0, 2, 0, -2, 0, 0],
                       "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                       "A2u": [1, 1, 1, -1, -1, -1, -1, -1, 1, 1],
                       "B1u": [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
                       "B2u": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                       "Eu":  [2, 0, -2, 0, 0, -2, 0, 2, 0, 0]}},
    "D5h": {"classes": [("E", 1, identity()), ("C5", 2, Cn(5)), ("C5²", 2, Cn(5, k=2)),
                        ("C2'", 5, Cn(2, (1, 0, 0))), ("σh", 1, sigma((0, 0, 1))),
                        ("S5", 2, Sn(5)), ("S5³", 2, Sn(5, k=3)),
                        ("σv", 5, sigma((0, 1, 0)))],
            "irreps": {"A1'":  [1, 1, 1, 1, 1, 1, 1, 1],
                       "A2'":  [1, 1, 1, -1, 1, 1, 1, -1],
                       "E1'":  [2, _a, _b, 0, 2, _a, _b, 0],
                       "E2'":  [2, _b, _a, 0, 2, _b, _a, 0],
                       "A1''": [1, 1, 1, 1, -1, -1, -1, -1],
                       "A2''": [1, 1, 1, -1, -1, -1, -1, 1],
                       "E1''": [2, _a, _b, 0, -2, -_a, -_b, 0],
                       "E2''": [2, _b, _a, 0, -2, -_b, -_a, 0]}},
    "D6h": {"classes": [("E", 1, identity()), ("C6", 2, Cn(6)), ("C3", 2, Cn(3)),
                        ("C2", 1, Cn(2)), ("C2'", 3, Cn(2, (1, 0, 0))),
                        ("C2''", 3, Cn(2, (_c30, _s30, 0))), ("i", 1, inversion()),
                        ("S3", 2, Sn(3)), ("S6", 2, Sn(6)),
                        ("σh", 1, sigma((0, 0, 1))), ("σd", 3, sigma((-_s30, _c30, 0))),
                        ("σv", 3, sigma((0, 1, 0)))],
            "irreps": {"A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                       "A2g": [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1],
                       "B1g": [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                       "B2g": [1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1],
                       "E1g": [2, 1, -1, -2, 0, 0, 2, 1, -1, -2, 0, 0],
                       "E2g": [2, -1, -1, 2, 0, 0, 2, -1, -1, 2, 0, 0],
                       "A1u": [1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1],
                       "A2u": [1, 1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1],
                       "B1u": [1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, 1],
                       "B2u": [1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1],
                       "E1u": [2, 1, -1, -2, 0, 0, -2, -1, 1, 2, 0, 0],
                       "E2u": [2, -1, -1, 2, 0, 0, -2, 1, 1, -2, 0, 0]}},
    "D8h": _times_i(_D8, ["i", "S8³", "S8", "S4", "σh", "σv", "σd"]),
    # ---------------- Dnd ----------------
    "D2d": {"classes": [("E", 1, identity()), ("S4", 2, Sn(4)), ("C2", 1, Cn(2)),
                        ("C2'", 2, Cn(2, (1, 0, 0))), ("σd", 2, sigma((1, -1, 0)))],
            "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1],
                       "B1": [1, -1, 1, 1, -1], "B2": [1, -1, 1, -1, 1], "E": [2, 0, -2, 0, 0]}},
    "D3d": _times_i(_D3, ["i", "S6", "σd"]),
    "D4d": {"classes": [("E", 1, identity()), ("S8", 2, Sn(8)), ("C4", 2, Cn(4)),
                        ("S8³", 2, Sn(8, k=3)), ("C2", 1, Cn(2)),
                        ("C2'", 4, Cn(2, (1, 0, 0))),
                        ("σd", 4, sigma((np.cos(5 * np.pi / 8), np.sin(5 * np.pi / 8), 0)))],
            "irreps": {"A1": [1, 1, 1, 1, 1, 1, 1], "A2": [1, 1, 1, 1, 1, -1, -1],
                       "B1": [1, -1, 1, -1, 1, 1, -1], "B2": [1, -1, 1, -1, 1, -1, 1],
                       "E1": [2, _r2, 0, -_r2, -2, 0, 0], "E2": [2, 0, -2, 0, 2, 0, 0],
                       "E3": [2, -_r2, 0, _r2, -2, 0, 0]}},
    "D5d": _times_i(_D5, ["i", "S10³", "S10", "σd"]),
    "D6d": {"classes": [("E", 1, identity()), ("S12", 2, Sn(12)), ("C6", 2, Cn(6)),
                        ("S4", 2, Sn(4)), ("C3", 2, Cn(3)),
                        ("S12⁵", 2, Sn(12, k=5)), ("C2", 1, Cn(2)),
                        ("C2'", 6, Cn(2, (1, 0, 0))),
                        ("σd", 6, sigma((np.cos(7 * np.pi / 12), np.sin(7 * np.pi / 12), 0)))],
            "irreps": {"A1": [1, 1, 1, 1, 1, 1, 1, 1, 1], "A2": [1, 1, 1, 1, 1, 1, 1, -1, -1],
                       "B1": [1, -1, 1, -1, 1, -1, 1, 1, -1], "B2": [1, -1, 1, -1, 1, -1, 1, -1, 1],
                       "E1": [2, _r3, 1, 0, -1, -_r3, -2, 0, 0],
                       "E2": [2, 1, -1, -2, -1, 1, 2, 0, 0],
                       "E3": [2, 0, -2, 0, 2, 0, -2, 0, 0],
                       "E4": [2, -1, -1, 2, -1, -1, 2, 0, 0],
                       "E5": [2, -_r3, 1, 0, -1, _r3, -2, 0, 0]}},
    # ---------------- cubic and icosahedral ----------------
    "T": _T,
    "Th": _times_i(_T, ["i", "S6⁵", "S6", "σh"]),
    "Td": {"classes": [("E", 1, identity()), ("C3", 8, Cn(3, (1, 1, 1))),
                       ("C2", 3, Cn(2)), ("S4", 6, Sn(4)),
                       ("σd", 6, sigma((1, -1, 0)))],
           "irreps": {"A1": [1, 1, 1, 1, 1], "A2": [1, 1, 1, -1, -1], "E": [2, -1, 2, 0, 0],
                      "T1": [3, 0, -1, 1, -1], "T2": [3, 0, -1, -1, 1]}},
    "O": _O,
    "Oh": {"classes": [("E", 1, identity()), ("C3", 8, Cn(3, (1, 1, 1))),
                       ("C2'", 6, Cn(2, (1, 1, 0))), ("C4", 6, Cn(4)),
                       ("C2", 3, Cn(2)), ("i", 1, inversion()),
                       ("S4", 6, Sn(4)), ("S6", 8, Sn(6, (1, 1, 1))),
                       ("σh", 3, sigma((0, 0, 1))), ("σd", 6, sigma((1, -1, 0)))],
           "irreps": {"A1g": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                      "A2g": [1, 1, -1, -1, 1, 1, -1, 1, 1, -1],
                      "Eg":  [2, -1, 0, 0, 2, 2, 0, -1, 2, 0],
                      "T1g": [3, 0, -1, 1, -1, 3, 1, 0, -1, -1],
                      "T2g": [3, 0, 1, -1, -1, 3, -1, 0, -1, 1],
                      "A1u": [1, 1, 1, 1, 1, -1, -1, -1, -1, -1],
                      "A2u": [1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
                      "Eu":  [2, -1, 0, 0, 2, -2, 0, 1, -2, 0],
                      "T1u": [3, 0, -1, 1, -1, -3, -1, 0, 1, 1],
                      "T2u": [3, 0, 1, -1, -1, -3, 1, 0, 1, -1]}},
    "I": _I,
    "Ih": _times_i(_I, ["i", "S10³", "S10", "S6", "σ"]),
}


# Orientation conventions, recalled in the warnings
_ORIENTATION = {
    "C2v": "C2 along z, molecular plane = yz",
    "D2": "C2 axes along x, y, z", "D2h": "C2 axes along x, y, z",
    "D2d": "S4 along z, C2' along x and y, σd planes on the diagonals",
    "Td": "C2 (S4) axes along x, y, z, C3 along (1,1,1)",
    "T": "C2 axes along x, y, z, C3 along (1,1,1)",
    "Th": "C2 axes along x, y, z, C3 along (1,1,1)",
    "O": "C4 axes along x, y, z", "Oh": "C4 axes along x, y, z",
    "I": "C2 axes along x, y, z, C5 along (0, 1, φ)",
    "Ih": "C2 axes along x, y, z, C5 along (0, 1, φ)",
}

_ORIENTATION_FR = {
    "C2v": "C2 selon z, plan moléculaire = yz",
    "D2": "axes C2 selon x, y, z", "D2h": "axes C2 selon x, y, z, z passant par le plus d'atomes, x perpendiculaire au plan moléculaire",
    "D2d": "S4 selon z, C2' selon x et y, plans σd sur les diagonales",
    "Td": "axes C2 (S4) selon x, y, z, C3 selon (1,1,1)",
    "T": "axes C2 selon x, y, z, C3 selon (1,1,1)",
    "Th": "axes C2 selon x, y, z, C3 selon (1,1,1)",
    "O": "axes C4 selon x, y, z", "Oh": "axes C4 selon x, y, z",
    "I": "axes C2 selon x, y, z, C5 selon (0, 1, φ)",
    "Ih": "axes C2 selon x, y, z, C5 selon (0, 1, φ)",
}
_ORIENTATION["D2h"] = "C2 axes along x, y, z, z through the largest number of atoms, x perpendicular to the molecular plane"

# classes examined first to choose between equivalent orientations (Mulliken conventions):
# C2v: molecular plane = yz; D2h: z through the largest number of atoms, molecular plane = yz.
# Otherwise, the first classes (C2', σv...) contain the largest number of atoms.
_PRIORITY = {"C2v": ["σv'(yz)"], "D2h": ["C2(z)", "σ(yz)"]}

# other names of the same groups (e.g. pymatgen)
_ALIASES = {"C3i": "S6", "S1": "Cs", "S2": "Ci", "S3": "C3h", "S5": "C5h", "S7": "C7h",
            "C1h": "Cs", "Vh": "D2h", "Vd": "D2d", "V": "D2"}

_LINEAR_GROUPS = {"C*v", "D*h", "C∞v", "D∞h", "Cinfv", "Dinfh"}

def available_groups():
    """List of the point groups whose character table is available."""
    return list(_TABLES)

def _key(M):
    return tuple(np.round(M, 6).ravel() + 0.0)

def _group_elements(generators):
    """All the operations of the group generated by a list of 3x3 matrices."""
    elements = {_key(np.eye(3)): np.eye(3)}
    queue = [np.eye(3)]
    while queue:
        A = queue.pop()
        for G in generators:
            B = G @ A
            k = _key(B)
            if k not in elements:
                elements[k] = B
                queue.append(B)
    return list(elements.values())

_CACHE = {}

def _group_data(group):
    """
    Operations of the group, the class index of each operation, and the norm
    of each row of the character table (1 for an irrep, 2 for a pair of
    complex conjugate irreps given as a single real row).
    """
    if group in _CACHE:
        return _CACHE[group]
    t = _TABLES[group]
    reps = [M for _, _, M in t["classes"]]
    elements = _group_elements(reps)
    conj = [{_key(A @ M @ A.T) for A in elements} for M in reps]
    class_of = []
    for R in elements:
        k = _key(R)
        class_of.append(next(c for c, s in enumerate(conj) if k in s))
    g = np.array([g for _, g, _ in t["classes"]])
    h = len(elements)
    chars = np.array(list(t["irreps"].values()), dtype=float)
    norms = (chars ** 2) @ g / h
    data = {"elements": elements, "class_of": class_of, "g": g, "h": h,
            "chars": chars, "norms": norms, "names": list(t["irreps"])}
    _CACHE[group] = data
    return data

############################################################
#      Basis functions (linear, rotations, quadratic)
############################################################
def _sym(*pairs):
    """Symmetric 3x3 matrix Q of a quadratic function f = r^T Q r."""
    Q = np.zeros((3, 3))
    for (i, j), c in pairs:
        Q[i, j] += c / 2
        Q[j, i] += c / 2
    return Q

_QUAD_SINGLE = [
    ("x²", _sym(((0, 0), 1))), ("y²", _sym(((1, 1), 1))), ("z²", _sym(((2, 2), 1))),
    ("x²+y²", _sym(((0, 0), 1), ((1, 1), 1))),
    ("x²+y²+z²", _sym(((0, 0), 1), ((1, 1), 1), ((2, 2), 1))),
    ("x²-y²", _sym(((0, 0), 1), ((1, 1), -1))),
    ("xy", _sym(((0, 1), 1))), ("xz", _sym(((0, 2), 1))), ("yz", _sym(((1, 2), 1))),
]
_Q = dict(_QUAD_SINGLE)
_Q["2z²-x²-y²"] = _sym(((2, 2), 2), ((0, 0), -1), ((1, 1), -1))
_QUAD_TUPLES = [("x²-y²", "xy"), ("xz", "yz"), ("2z²-x²-y²", "x²-y²"),
                ("xy", "xz", "yz"), ("2z²-x²-y²", "x²-y²", "xy", "xz", "yz")]
_QUAD_DISPLAY = ["x²+y²+z²", "x²+y²", "x²", "y²", "z²", "x²-y²", "xy", "xz", "yz"]

_E3 = np.eye(3)
_LIN = {"x": _E3[0], "y": _E3[1], "z": _E3[2], "Rx": _E3[0], "Ry": _E3[1], "Rz": _E3[2]}

def _projector(data, row, kind):
    """
    Projection operator of an irrep on the space of the linear ('lin'),
    rotation ('rot') or quadratic ('quad') functions.
    """
    chi = data["chars"][row]
    dim, norm = chi[0], data["norms"][row]
    if kind == "quad":
        P = np.zeros((6, 6))
        idx = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        def vec(Q):
            return np.array([Q[i, j] for i, j in idx])
        def mat(v):
            Q = np.zeros((3, 3))
            for (i, j), c in zip(idx, v):
                Q[i, j] = Q[j, i] = c
            return Q
        for R, c in zip(data["elements"], data["class_of"]):
            P += chi[c] * np.column_stack([vec(R.T @ mat(e) @ R) for e in np.eye(6)])
        return P * dim / norm / data["h"], vec
    P = np.zeros((3, 3))
    for R, c in zip(data["elements"], data["class_of"]):
        s = np.linalg.det(R) if kind == "rot" else 1.0
        P += chi[c] * s * R.T
    return P * dim / norm / data["h"], (lambda v: v)

def _basis_functions(data, row, kind):
    """Names of the basis functions of an irrep, e.g. ['(x, y)'] or ['x²+y²', 'z²']."""
    P, vec = _projector(data, row, kind)
    dim = int(round(data["chars"][row][0]))
    if kind == "quad":
        singles = [(n, vec(Q)) for n, Q in _QUAD_SINGLE]
        tuples = [(t, [vec(_Q[n]) for n in t]) for t in _QUAD_TUPLES]
    else:
        names = ["x", "y", "z"] if kind == "lin" else ["Rx", "Ry", "Rz"]
        singles = [(n, _LIN[n]) for n in names]
        tuples = [(tuple(names[:2]), [_LIN[n] for n in names[:2]]), (tuple(names), [_LIN[n] for n in names])]
    target = np.linalg.matrix_rank(P, tol=1e-6)
    chosen, basis = [], []
    candidates = singles if dim == 1 else [(t, v) for t, v in tuples if len(t) == dim]
    for name, v in candidates:
        vs = [v] if dim == 1 else v
        if all(np.allclose(P @ w, w, atol=1e-6) for w in vs):
            new = basis + list(vs)
            if np.linalg.matrix_rank(np.array(new), tol=1e-6) == len(new):
                basis = new
                chosen.append(name)
        if len(basis) == target:
            break
    if dim == 1:
        if kind == "quad":
            chosen.sort(key=_QUAD_DISPLAY.index)
        return chosen
    return ["(" + ", ".join(t) + ")" for t in chosen]

def _fmt(c):
    """Formats a character: integers as such, irrational values with 3 decimals."""
    if abs(c - round(c)) < 1e-9:
        return str(int(round(c)))
    return f"{c:.3f}"

def _resolve_group(group, lang="En"):
    """Standard name of a point group; tells the user when another name was given (S3 -> C3h...)."""
    std = _ALIASES.get(group, group)
    if std != group:
        m = re.match(r"^S(\d+)$", group)
        fr = lang.lower().capitalize() == "Fr"
        if m and int(m.group(1)) % 2 == 1 and int(m.group(1)) > 1:
            n = m.group(1)
            if fr:
                print(f"{group} = {std} : pour n impair, S{n} contient σh et C{n}, c'est le groupe {std}.")
            else:
                print(f"{group} = {std}: for odd n, S{n} contains σh and C{n}, this is the group {std}.")
        else:
            if fr:
                print(f"{group} = {std} (autre notation du même groupe).")
            else:
                print(f"{group} = {std} (another name of the same group).")
    return std

def _fmt_complex(c, N):
    """Complex character written with ε = exp(2πi/N): 1, -1, i, -i, ε, ε², ε*, -ε²*..."""
    c = complex(c)
    for v, txt in ((1, "1"), (-1, "-1"), (1j, "i"), (-1j, "-i")):
        if abs(c - v) < 1e-9:
            return txt
    for sign, prefix in ((1, ""), (-1, "-")):
        for p in range(1, N):
            if abs(sign * c - np.exp(2j * np.pi * p / N)) < 1e-9:
                q, star = (p, "") if p <= N / 2 else (N - p, "*")
                power = "" if q == 1 else str(q).translate(_SUP)
                return f"{prefix}ε{power}{star}"
    return f"{c.real:.3f}{c.imag:+.3f}i"

def character_table(group, functions=True, lang="En", complex=False):
    """
    Character table of a point group, as a pandas DataFrame.

    Pairs of complex conjugate irreducible representations (groups C3...C8,
    S4, S6, S8, C3h...C8h, T, Th) are given by default as a single real row
    (sum of the two characters), marked with \*. With complex=True, the two
    components are given separately (¹E, ²E), with ε = exp(2πi/n).

    Args:
        group (str): Schoenflies symbol of the point group (see `available_groups()`).
        functions (bool): if True, adds the columns of the basis functions
                          (linear functions and rotations, quadratic functions).
        lang (str): language of the messages ("En" (default) or "Fr").
        complex (bool): if True, complex form of the pairs of complex conjugate irreps.
    """
    group = _resolve_group(group, lang)
    if group in _LINEAR_GROUPS:
        raise ValueError("Linear molecules (C∞v, D∞h) are not handled: use the subgroups C2v and D2h.")
    if group not in _TABLES:
        raise ValueError(f"Unknown point group '{group}'. Available: {', '.join(_TABLES)}")
    t = _TABLES[group]
    data = _group_data(group)
    columns = [f"{g}{lab}" if g > 1 else lab for lab, g, _ in t["classes"]]
    cplx = t.get("complex", set())
    rows_c = t.get("complex_rows", {})
    use_complex = complex and bool(rows_c)
    index, values, funcs = [], [], []
    for row, (name, chars) in enumerate(zip(data["names"], data["chars"])):
        if functions:
            f = (", ".join(_basis_functions(data, row, "lin") + _basis_functions(data, row, "rot")),
                 ", ".join(_basis_functions(data, row, "quad")))
        else:
            f = ("", "")
        if use_complex and name in rows_c:
            plus, minus = rows_c[name]
            index += ["¹" + name, "²" + name]
            values += [[_fmt_complex(c, t["eps_n"]) for c in plus],
                       [_fmt_complex(c, t["eps_n"]) for c in minus]]
            funcs += [f, ("", "")]
        else:
            index.append(name + "*" if name in cplx else name)
            values.append([_fmt(c) for c in chars])
            funcs.append(f)
    df = pd.DataFrame(values, index=index, columns=columns)
    if functions:
        df["linear, rotations"] = [f[0] for f in funcs]
        df["quadratic"] = [f[1] for f in funcs]
    df.columns.name = group
    if use_complex:
        n = t["eps_n"]
        df.columns.name = f"{group} (ε = exp(2πi/{n}))"
    return df

def reduce_representation(group, chi, show=True, lang="En"):
    """
    Reduction of a representation of a point group into irreducible
    representations, without any molecule: n_i = (1/h) Σ_classes g_c χ(c) χ_i(c)*.
    A pair of complex conjugate irreps (row marked * in the character table) is
    counted as a whole.

    Args:
        group (str): Schoenflies symbol of the point group.
        chi (array-like): characters of the representation, one per class, in the
            order of the columns of `character_table(group)`.
        show (bool): if True, prints the decomposition (e.g. Γ = 2A1 + B2) and returns
            nothing; if False, returns the result without printing.
        lang (str): language of the messages ("En" (default) or "Fr").

    Returns:
        dict: number of times each irreducible representation appears (show=False).

    Example:
        >>> reduce_representation("C3v", [4, 1, 2])
        Γ = 2A1 + E
    """
    group = _resolve_group(group, lang)
    if group not in _TABLES:
        raise ValueError(f"Unknown point group '{group}'. Available: {', '.join(_TABLES)}")
    data = _group_data(group)
    chi = np.asarray(chi, dtype=float)
    if len(chi) != len(data["g"]):
        raise ValueError(f"{len(chi)} characters given, {len(data['g'])} classes in {group}")
    n = {}
    for irrep, chi_i, norm in zip(data["names"], data["chars"], data["norms"]):
        n[irrep] = float(np.sum(data["g"] * chi * chi_i) / data["h"] / norm)
    if any(abs(v - round(v)) > 1e-6 for v in n.values()):
        if lang.lower().capitalize() == "Fr":
            print("Attention : coefficients non entiers, vérifiez les caractères")
        else:
            print("Warning: non-integer coefficients, check the characters")
        if show:
            terms = [f"{v:.4g} {k}" for k, v in n.items() if abs(v) > 1e-9]
            print("Γ = " + " + ".join(terms).replace("+ -", "- "))
            return None
        return n
    n = {k: int(round(v)) for k, v in n.items() if round(v) != 0}
    if show:
        print("Γ = " + " + ".join(f"{v}{k}" if v > 1 else k for k, v in n.items()))
        return None
    return n

def ir_raman_activity(group, chi, show=True, lang="En"):
    """
    IR and Raman activity of the irreducible representations contained in a
    representation of a point group, without any molecule: IR active if the
    irrep transforms as x, y or z, Raman active if it transforms as a quadratic
    function (x², xy...).

    Args:
        group (str): Schoenflies symbol of the point group.
        chi (array-like): characters of the representation, one per class.
        show (bool): if True, prints the active irreps and returns nothing;
            if False, returns the result without printing.
        lang (str): language of the messages ("En" (default) or "Fr").

    Returns:
        dict: {"IR": [...], "Raman": [...]}, irreps with their multiplicities (show=False).
    """
    group = _resolve_group(group, lang)
    data = _group_data(group)
    n = reduce_representation(group, chi, show=False, lang=lang)
    rows = {name: k for k, name in enumerate(data["names"])}
    ir, raman = [], []
    for irrep, mult in n.items():
        k = rows[irrep]
        term = f"{mult}{irrep}" if mult > 1 else irrep
        if _basis_functions(data, k, "lin"):
            ir.append(term)
        if _basis_functions(data, k, "quad"):
            raman.append(term)
    if show:
        fr = lang.lower().capitalize() == "Fr"
        print(("actifs en IR : " if fr else "IR active: ") + (" + ".join(ir) if ir else "-")
              + ("   |   actifs en Raman : " if fr else "   |   Raman active: ") + (" + ".join(raman) if raman else "-"))
        return None
    return {"IR": ir, "Raman": raman}

############################################################
#              Flowchart of the point groups
############################################################
_FLOW_TXT = {
    "Fr": {"lin": "Molécule linéaire ?", "i": "Centre d'inversion i ?",
           "cub": "Plusieurs axes Cn\navec n ≥ 3 ?", "cn": "Un axe Cn ?\n(n le plus grand)",
           "nc2": "n axes C2 ⊥ Cn ?", "sh": "Plan σh ⊥ Cn ?", "nsd": "n plans σd ?",
           "nsv": "n plans σv ?\n(contenant Cn)", "s2n": "Axe S2n\nconfondu avec Cn ?",
           "sig": "Plan σ ?", "yes": "oui", "no": "non",
           "cubic": "T, Td, Th\nO, Oh, I, Ih", "title": "Recherche du groupe ponctuel"},
    "En": {"lin": "Linear molecule?", "i": "Inversion center i?",
           "cub": "Several Cn axes\nwith n ≥ 3?", "cn": "A Cn axis?\n(highest n)",
           "nc2": "n C2 axes ⊥ Cn?", "sh": "σh plane ⊥ Cn?", "nsd": "n σd planes?",
           "nsv": "n σv planes?\n(containing Cn)", "s2n": "S2n axis\ncollinear with Cn?",
           "sig": "σ plane?", "yes": "yes", "no": "no",
           "cubic": "T, Td, Th\nO, Oh, I, Ih", "title": "Finding the point group"},
}

def _flow_path(group):
    """Nodes and terminal reached in the flowchart for a point group (None: no path)."""
    if group is None:
        return set(), None
    g = _ALIASES.get(group, group)
    if g in _LINEAR_GROUPS:
        return {"lin", "lin_i"}, ("Dinfh" if g in ("D*h", "D∞h", "Dinfh") else "Cinfv")
    path = {"lin"}
    if g in ("T", "Th", "Td", "O", "Oh", "I", "Ih"):
        return path | {"cub"}, "cubic"
    path |= {"cub", "cn"}
    if g in ("C1", "Cs", "Ci"):
        path |= {"sig"}
        if g == "Cs":
            return path, "Cs"
        return path | {"sig_i"}, g
    path |= {"nc2"}
    if g.startswith("D"):
        path |= {"d_sh"}
        if g.endswith("h"):
            return path, "Dnh"
        path |= {"nsd"}
        return path, ("Dnd" if g.endswith("d") else "Dn")
    path |= {"sh"}
    if g.endswith("h"):
        return path, "Cnh"
    path |= {"nsv"}
    if g.endswith("v"):
        return path, "Cnv"
    path |= {"s2n"}
    return path, ("S2n" if g.startswith("S") else "Cn")

def point_group_flowchart(group=None, lang="En", ax=None):
    """
    Flowchart (decision tree) to find the point group of a molecule. If a group
    is given, the path leading to it is highlighted.

    Args:
        group (str): Schoenflies symbol of the point group to highlight (optional).
        lang (str): "En" (default) or "Fr".
        ax (matplotlib.axes.Axes): axes to draw in (optional).

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    T = _FLOW_TXT["Fr" if lang.lower().capitalize() == "Fr" else "En"]
    path, end = _flow_path(group)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 12))
    else:
        fig = ax.figure
    on, off = "#d62728", "#555555"
    # questions: id -> (x, y, text)
    Q = {"lin": (0, 0, T["lin"]), "lin_i": (3.4, 0, T["i"]),
         "cub": (0, -1.6, T["cub"]), "cn": (0, -3.2, T["cn"]),
         "sig": (3.4, -3.2, T["sig"]), "sig_i": (6.6, -4.0, T["i"]),
         "nc2": (0, -5.6, T["nc2"]), "d_sh": (3.4, -5.6, T["sh"]), "nsd": (6.6, -6.4, T["nsd"]),
         "sh": (0, -7.6, T["sh"]), "nsv": (0, -9.0, T["nsv"]), "s2n": (0, -10.4, T["s2n"])}
    # terminals: id -> (x, y, text)
    R = {"Dinfh": (6.6, 0.45, "D∞h"), "Cinfv": (6.6, -0.45, "C∞v"), "cubic": (3.4, -1.6, T["cubic"]),
         "Cs": (6.6, -2.7, "Cs"), "Ci": (9.4, -3.55, "Ci"), "C1": (9.4, -4.45, "C1"),
         "Dnh": (6.6, -5.1, "Dnh"), "Dnd": (9.4, -5.95, "Dnd"), "Dn": (9.4, -6.85, "Dn"),
         "Cnh": (3.4, -7.6, "Cnh"), "Cnv": (3.4, -9.0, "Cnv"), "S2n": (3.4, -10.4, "S2n"),
         "Cn": (0, -11.8, "Cn")}
    # half sizes of the boxes
    half = {k: (1.2, 0.4) for k in Q}
    half.update({k: (0.55, 0.25) for k in R})
    half["cubic"] = (1.2, 0.4)
    # edges: (from, to, label)
    E = [("lin", "lin_i", "yes"), ("lin_i", "Dinfh", "yes"), ("lin_i", "Cinfv", "no"),
         ("lin", "cub", "no"), ("cub", "cubic", "yes"), ("cub", "cn", "no"),
         ("cn", "sig", "no"), ("sig", "Cs", "yes"), ("sig", "sig_i", "no"),
         ("sig_i", "Ci", "yes"), ("sig_i", "C1", "no"),
         ("cn", "nc2", "yes"), ("nc2", "d_sh", "yes"), ("d_sh", "Dnh", "yes"), ("d_sh", "nsd", "no"),
         ("nsd", "Dnd", "yes"), ("nsd", "Dn", "no"),
         ("nc2", "sh", "no"), ("sh", "Cnh", "yes"), ("sh", "nsv", "no"),
         ("nsv", "Cnv", "yes"), ("nsv", "s2n", "no"), ("s2n", "S2n", "yes"), ("s2n", "Cn", "no")]
    pos = {k: (x, y) for k, (x, y, _) in {**Q, **R}.items()}
    reached = set(path) | ({end} if end else set())
    for a, b, lab in E:
        (x0, y0), (x1, y1) = pos[a], pos[b]
        if abs(x1 - x0) < 1e-9:        # vertical arrow
            start, stop = (x0, y0 - half[a][1]), (x1, y1 + half[b][1])
        else:                           # arrow to the right
            start, stop = (x0 + half[a][0], y0), (x1 - half[b][0], y1)
        active = a in path and b in reached
        col = on if active else off
        ax.annotate("", xy=stop, xytext=start,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2 if active else 1.0,
                                    shrinkA=0, shrinkB=0))
        mx, my = (start[0] + stop[0]) / 2, (start[1] + stop[1]) / 2
        if abs(x1 - x0) < 1e-9:
            ax.text(mx + 0.1, my, T[lab], color=col, fontsize=9, ha="left", va="center", style="italic")
        else:
            ax.text(mx, my + 0.08, T[lab], color=col, fontsize=9, ha="center", va="bottom", style="italic")
    for k, (x, y, txt) in Q.items():
        active = k in path
        ax.add_patch(FancyBboxPatch((x - 1.2, y - 0.4), 2.4, 0.8, boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#fde0dd" if active else "#f2f2f2", ec=on if active else off,
                                    lw=2 if active else 1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=9.5)
    for k, (x, y, txt) in R.items():
        active = k == end
        w, h = 2 * half[k][0], 2 * half[k][1]
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1",
                                    fc=on if active else "#dbe9f6", ec=on if active else "#1f77b4",
                                    lw=2 if active else 1))
        ax.text(x, y, txt, ha="center", va="center", fontsize=11, fontweight="bold",
                color="white" if active else "#1f4e79")
    title = T["title"] + (f" : {group}" if group and T is _FLOW_TXT["Fr"] else (f": {group}" if group else ""))
    ax.set_title(title, fontsize=13)
    ax.set_xlim(-1.4, 10.2)
    ax.set_ylim(-12.3, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.show()
    return fig

############################################################
#      Standard orientation from the symmetry operations
############################################################
def _direction(M):
    """Direction of the symmetry element of M (None for E and i)."""
    if np.allclose(M, np.eye(3), atol=1e-2) or np.allclose(M, -np.eye(3), atol=1e-2):
        return None
    w, v = np.linalg.eig(M)
    target = 1 if np.linalg.det(M) > 0 else -1
    u = np.real(v[:, int(np.argmin(np.abs(w - target)))])
    return u / np.linalg.norm(u)

def _signature(M):
    return (int(round(np.linalg.det(M))), round(float(np.trace(M)), 1))

def _frame(u, w):
    """Orthonormal frame built from a first direction u and a second one w."""
    e1 = u / np.linalg.norm(u)
    if w is None:
        w = np.array([1.0, 0, 0]) if abs(e1[0]) < 0.9 else np.array([0, 1.0, 0])
    e2 = w - (w @ e1) * e1
    e2 /= np.linalg.norm(e2)
    return np.array([e1, e2, np.cross(e1, e2)])

def _symmetrize(symbols, X, elements, tol):
    """Average of the images of each atom over the group: exact symmetry."""
    sy = np.array(symbols)
    same = sy[:, None] == sy[None, :]
    Xs = np.zeros_like(X)
    for R in elements:
        Y = X @ R.T
        D = np.linalg.norm(Y[:, None, :] - X[None, :, :], axis=2)
        D[~same] = np.inf
        j = D.argmin(axis=1)
        if D[np.arange(len(X)), j].max() > tol or len(set(j)) != len(X):
            return None
        Xs[j] += Y
    return Xs / len(elements)

def _orient(symbols, X, group, mol_ops, tol=0.3):
    """
    Rotates the molecule into the standard orientation of its point group.

    Args:
        symbols (list): chemical symbols.
        X (np.ndarray): coordinates centered on the center of mass.
        group (str): Schoenflies symbol.
        mol_ops (list of np.ndarray): the symmetry operations of the molecule
            in its own frame (e.g. given by pymatgen).
        tol (float): tolerance (Å) used to match atoms during the symmetrization.

    Returns:
        np.ndarray: symmetrized coordinates in the standard orientation, or None.
    """
    data = _group_data(group)
    ref = np.array(data["elements"])
    mol_ops = [np.asarray(M, dtype=float) for M in mol_ops]
    if len(mol_ops) != data["h"]:
        return None
    # two reference elements with non parallel directions define the frame
    reps = [M for _, _, M in _TABLES[group]["classes"]]
    A = next((M for M in reps if _direction(M) is not None), None)
    if A is None:                               # C1, Ci: no orientation needed
        return _symmetrize(symbols, X, ref, tol)
    uA = _direction(A)
    B = next((M for M in ref if _direction(M) is not None
              and abs(_direction(M) @ uA) < 0.99), None)
    uB = None if B is None else _direction(B)
    F_ref = _frame(uA, uB)
    sig = [_signature(M) for M in mol_ops]
    dirs = [_direction(M) for M in mol_ops]
    sA, sB = _signature(A), (None if B is None else _signature(B))
    cos_AB = None if B is None else abs(uB @ uA)
    best, best_score, seen = None, None, set()
    for ia, a in enumerate(mol_ops):
        if sig[ia] != sA or dirs[ia] is None:
            continue
        ua = dirs[ia]
        cands = [None] if B is None else [dirs[ib] for ib in range(len(mol_ops)) if sig[ib] == sB
                                          and dirs[ib] is not None
                                          and abs(abs(dirs[ib] @ ua) - cos_AB) < 0.05]
        for ub0 in cands:
            for sa in (1, -1):
                for sb in ((1,) if ub0 is None else (1, -1)):
                    ub = None if ub0 is None else sb * ub0
                    Q = F_ref.T @ _frame(sa * ua, ub)
                    k = tuple(np.round(Q, 2).ravel())
                    if k in seen:
                        continue
                    seen.add(k)
                    T = np.array([Q @ M @ Q.T for M in mol_ops])
                    dist = np.abs(T[:, None, :, :] - ref[None, :, :, :]).max(axis=(2, 3)).min(axis=1)
                    if dist.max() > 0.1:
                        continue
                    Xs = _symmetrize(symbols, X @ Q.T, ref, tol)
                    if Xs is None:
                        continue
                    # conventions: see _PRIORITY
                    fixed = []
                    for _, _, M in _TABLES[group]["classes"]:
                        Y = Xs @ M.T
                        fixed.append(sum(np.linalg.norm(x - y) < 1e-3 for x, y in zip(Xs, Y)))
                    labs = [lab for lab, _, _ in _TABLES[group]["classes"]]
                    first = [fixed[labs.index(l)] for l in _PRIORITY.get(group, [])]
                    score = tuple(first) + tuple(fixed)
                    if best_score is None or score > best_score:
                        best, best_score = Xs, score
    return best

############################################################
#   Automatic construction of the axial groups (n up to 8)
############################################################
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹"

def _power_of(name):
    """Power written in a label such as 'S10³' (1 if none)."""
    m = re.match(r"^[CS]\d+([⁰¹²³⁴⁵⁶⁷⁸⁹]*)", name)
    if not m or not m.group(1):
        return 1
    return int("".join(str(_SUPERSCRIPTS.index(c)) for c in m.group(1)))

def _class_label(mats):
    """Label of a class of rotations / improper rotations about z: the element of lowest power."""
    names = [re.sub(r"\(z\)$", "", identify(M)) for M in mats]
    return min(names, key=lambda n: (_power_of(n), n))

def _e_names(n_e):
    return ["E"] if n_e == 1 else [f"E{j}" for j in range(1, n_e + 1)]

def _cyclic(N, gen):
    """
    Cyclic group generated by gen (order N, about z): Cn, or S2n for even n.
    Pairs of complex conjugate irreps are given as real rows ("complex").
    """
    powers = [np.linalg.matrix_power(gen, k) for k in range(N)]
    classes = [("E", 1, identity())] + [(_class_label([powers[k]]), 1, powers[k]) for k in range(1, N)]
    irreps = {"A": [1] * N}
    if N % 2 == 0:
        irreps["B"] = [(-1) ** k for k in range(N)]
    n_e = (N - 1) // 2
    names = _e_names(n_e)
    rows = {}
    for j, name in zip(range(1, n_e + 1), names):
        irreps[name] = [2 * np.cos(2 * np.pi * j * k / N) for k in range(N)]
        rows[name] = ([np.exp(2j * np.pi * j * k / N) for k in range(N)],
                      [np.exp(-2j * np.pi * j * k / N) for k in range(N)])
    return {"classes": classes, "irreps": irreps, "complex": set(names),
            "complex_rows": rows, "eps_n": N}

def _dihedral(N, gen, flip1, flip2, labels):
    """
    Groups with the structure of C_Nv: Cnv, Dn, and Dnd for even n (N = 2n, gen = S2n).

    Args:
        N (int): order of the cyclic subgroup.
        gen (np.ndarray): generator of the cyclic subgroup (about z).
        flip1, flip2 (np.ndarray): representatives of the two classes of "flips"
            (σv/σd or C2'/C2''); flip2 is used only for even N.
        labels (tuple): labels of the flip classes, e.g. ("σv", "σd").
    """
    powers = [np.linalg.matrix_power(gen, k) for k in range(N)]
    classes = [("E", 1, identity())]
    ks = list(range(1, (N - 1) // 2 + 1))
    for k in ks:
        classes.append((_class_label([powers[k], powers[N - k]]), 2, powers[k]))
    if N % 2 == 0:
        classes.append((_class_label([powers[N // 2]]), 1, powers[N // 2]))
        classes.append((labels[0], N // 2, flip1))
        classes.append((labels[1], N // 2, flip2))
    else:
        classes.append((labels[0], N, flip1))
    rot_k = [0] + ks + ([N // 2] if N % 2 == 0 else [])
    n_f = 2 if N % 2 == 0 else 1
    irreps = {"A1": [1] * len(rot_k) + [1] * n_f,
              "A2": [1] * len(rot_k) + [-1] * n_f}
    if N % 2 == 0:
        irreps["B1"] = [(-1) ** k for k in rot_k] + [1, -1]
        irreps["B2"] = [(-1) ** k for k in rot_k] + [-1, 1]
    n_e = (N - 1) // 2
    for j, name in zip(range(1, n_e + 1), _e_names(n_e)):
        irreps[name] = [2 * np.cos(2 * np.pi * j * k / N) for k in rot_k] + [0] * n_f
    return {"classes": classes, "irreps": irreps}

def _class_members(table, M):
    """All the elements of the class of M in the group generated by the representatives of table."""
    reps = [R for _, _, R in table["classes"]]
    el = _group_elements(reps)
    keys = {_key(A @ M @ A.T) for A in el}
    return [R for R in el if _key(R) in keys]

def _vertical_planes_labels(table, op, n):
    """
    Labels σv / σd of the images op·C2' and op·C2'' (Dnh, Dnd for odd n):
    σv contains the C2' axes (the class of the C2 axis along x), σd does not.
    """
    c2p = [R for R in _class_members(table, Cn(2, "x"))]
    c2p_dirs = [_direction(R) for R in c2p]
    out = {}
    for lab, g, M in table["classes"]:
        if lab.startswith("C2'"):
            u = _direction(op @ M)
            contains = any(abs(u @ d) < 1e-6 for d in c2p_dirs)
            out[lab] = "σv" if contains else "σd"
    return out

def _product_labels(table, op):
    """Labels of the improper classes op·R, R running over the classes of table."""
    planes = _vertical_planes_labels(table, op, None)
    labels = []
    for lab, g, M in table["classes"]:
        P = op @ M
        if np.allclose(P, -np.eye(3)):
            labels.append("i")
        elif np.allclose(P, sigma("xy")):
            labels.append("σh")
        elif lab in planes:
            labels.append(planes[lab])
        else:
            labels.append(_class_label([op @ R for R in _class_members(table, M)]))
    return labels

def _axial_tables(nmax=8):
    """Cn, S2n, Cnv, Cnh, Dn, Dnh, Dnd for n up to nmax."""
    T = {}
    for n in range(2, nmax + 1):
        a = np.pi / n
        Cn_ = _cyclic(n, Cn(n))
        T[f"C{n}"] = Cn_
        T[f"C{n}v"] = _dihedral(n, Cn(n), sigma("xz"),
                                sigma((np.cos(a + np.pi / 2), np.sin(a + np.pi / 2), 0)), ("σv", "σd"))
        D = _dihedral(n, Cn(n), Cn(2, "x"), Cn(2, (np.cos(a), np.sin(a), 0)), ("C2'", "C2''"))
        T[f"D{n}"] = D
        if n % 2 == 0:
            T[f"C{n}h"] = _times_i(Cn_, _product_labels(Cn_, inversion()))
            T[f"D{n}h"] = _times_i(D, _product_labels(D, inversion()))
            b = np.pi / (2 * n)
            T[f"D{n}d"] = _dihedral(2 * n, Sn(2 * n), Cn(2, "x"),
                                    sigma((np.cos(b + np.pi / 2), np.sin(b + np.pi / 2), 0)), ("C2'", "σd"))
        else:
            T[f"C{n}h"] = _times_sigma_h(Cn_, _product_labels(Cn_, sigma("xy")))
            T[f"D{n}h"] = _times_sigma_h(D, _product_labels(D, sigma("xy")))
            T[f"D{n}d"] = _times_i(D, _product_labels(D, inversion()))
        if 2 * n <= nmax:
            if n % 2 == 0:
                T[f"S{2 * n}"] = _cyclic(2 * n, Sn(2 * n))
            else:
                T[f"S{2 * n}"] = _times_i(Cn_, _product_labels(Cn_, inversion()))
    return T


# The hand-written tables above (checked against the textbooks) are kept; the
# automatic construction, which reproduces them, completes the axial groups up to n = 8.
for _name, _table in _axial_tables(8).items():
    if _table.get("complex"):
        _TABLES[_name] = _table        # same table, plus the complex form of the pairs
    else:
        _TABLES.setdefault(_name, _table)

# usual order of the character tables in textbooks
_ORDER = (["C1", "Cs", "Ci"] + [f"C{n}" for n in range(2, 9)] + ["S4", "S6", "S8"]
          + [f"C{n}v" for n in range(2, 9)] + [f"C{n}h" for n in range(2, 9)]
          + [f"D{n}" for n in range(2, 9)] + [f"D{n}h" for n in range(2, 9)]
          + [f"D{n}d" for n in range(2, 9)] + ["T", "Th", "Td", "O", "Oh", "I", "Ih"])
_TABLES = {k: _TABLES[k] for k in _ORDER}

def _pymatgen_analysis(symbols, X, tolerance=0.3):
    """
    Point group and symmetry operations of a molecule, given by pymatgen.

    Returns:
        (str, np.ndarray, list): Schoenflies symbol, coordinates centered on the
        center of mass (same orientation), symmetry operations (3x3 matrices).
    """
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")        # numerical warnings of pymatgen, irrelevant here
        pga = PointGroupAnalyzer(Molecule(list(symbols), X), tolerance=tolerance)
        Xc = np.array(pga.centered_mol.cart_coords)
        ops = [np.real(op.rotation_matrix) for op in pga.get_symmetry_operations()]
        symbol = pga.sch_symbol
    return symbol, Xc, ops

def _chi_l(M, l):
    """
    Character of the 2l+1 spherical harmonics of angular momentum l
    (s: l=0, p: 1, d: 2, f: 3) for the operation M:
    χ_l(C_θ) = sin((2l+1)θ/2) / sin(θ/2), and χ_l(M) = (-1)^l χ_l(-M) for an improper M.
    """
    M = np.asarray(M, dtype=float)
    sign = 1
    if np.linalg.det(M) < 0:
        M, sign = -M, (-1) ** l
    theta = np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))
    if abs(np.sin(theta / 2)) < 1e-9:
        chi = 2 * l + 1
    else:
        chi = np.sin((2 * l + 1) * theta / 2) / np.sin(theta / 2)
    return sign * chi

_L = {"s": 0, "p": 1, "d": 2, "f": 3}

def _valence_shells(symbol):
    """Valence shells of an element: 's' (H, He), 'sp' (main groups), 'spd' (d block), 'spdf' (f block)."""
    try:
        from rdkit.Chem import GetPeriodicTable
        Z = GetPeriodicTable().GetAtomicNumber(symbol)
    except Exception:
        Z = {"H": 1, "He": 2}.get(symbol, 6)
    if Z <= 2:
        return "s"
    if 57 <= Z <= 71 or 89 <= Z <= 103:
        return "spdf"
    if 21 <= Z <= 30 or 39 <= Z <= 48 or 72 <= Z <= 80 or 104 <= Z <= 112:
        return "spd"
    return "sp"

# real f orbitals (cubic set), as functions of r = (x, y, z)
_F_ORBITALS = {
    "fz³": lambda r: r[2] * (2 * r[2] ** 2 - 3 * r[0] ** 2 - 3 * r[1] ** 2),
    "fxz²": lambda r: r[0] * (4 * r[2] ** 2 - r[0] ** 2 - r[1] ** 2),
    "fyz²": lambda r: r[1] * (4 * r[2] ** 2 - r[0] ** 2 - r[1] ** 2),
    "fz(x²-y²)": lambda r: r[2] * (r[0] ** 2 - r[1] ** 2),
    "fxyz": lambda r: r[0] * r[1] * r[2],
    "fx(x²-3y²)": lambda r: r[0] * (r[0] ** 2 - 3 * r[1] ** 2),
    "fy(3x²-y²)": lambda r: r[1] * (3 * r[0] ** 2 - r[1] ** 2),
}

def _parse_shells(shells, merge=True):
    """
    Shells of a basis, from a string ("s", "sp", "s px py", "spd", "s pxy") or a
    list (["s", "px", "py"]). The partial p shells of an atom are merged:
    "px py" -> "pxy", "px py pz" -> "p".
    """
    if isinstance(shells, str):
        tokens = re.findall(r"p[xyz]{1,3}|[a-z]", shells)
    else:
        tokens = [t for item in shells for t in re.findall(r"p[xyz]{1,3}|[a-z]", item)]
    if not merge:
        out = []
        for t in tokens:
            if t not in out:
                out.append(t)
        return out
    out, p_parts, p_pos = [], set(), None
    for t in tokens:
        if t == "p" or re.fullmatch(r"p[xyz]{1,3}", t):
            p_parts |= set("xyz") if t == "p" else set(t[1:])
            if p_pos is None:
                p_pos = len(out)
                out.append(None)
        elif t not in out:
            out.append(t)
    if p_pos is not None:
        out[p_pos] = "p" if p_parts == set("xyz") else "p" + "".join(sorted(p_parts))
    return out

############################################################
#                 Molecule and Γ_atoms
############################################################
class MolSym():
    """
    Symmetry analysis of a molecule: atom tracking under the symmetry
    operations of its point group, atomic representation Γ_atoms and its
    reduction into irreducible representations.

    By default (standard_orientation=True), the molecule is placed in the
    standard orientation of its point group, as the "Standard orientation" of
    Gaussian: principal axis along z; for C2v, molecular plane = yz; for Dnh,
    one C2' axis along x, through the largest number of atoms. If the molecule
    is not already in a standard frame, its symmetry operations are given by
    pymatgen. The coordinates are translated to the center of mass and
    symmetrized; the initial ones are kept in `coords_input`.

    Args:
        symbols (list of str): chemical symbols of the atoms.
        coords (array-like): Cartesian coordinates, shape (n_atoms, 3), in Å.
        group (str): Schoenflies symbol of the point group (see `available_groups()`);
                     if None, detected with pymatgen.
        tol (float): tolerance (Å) used to decide that two atoms coincide.
        lang (str): language of the messages and tables ("En" (default) or "Fr").
        standard_orientation (bool): if True (default), places the molecule in the
                     standard orientation of its point group.
        sym_tolerance (float): tolerance (Å) of the symmetry detection (pymatgen)
                     and of the symmetrization.
        verbose (bool): if True (default), prints the point group and the orientation
                     of the molecule at its creation.

    Attributes:
        group (str): Schoenflies symbol of the point group, e.g. "C2v".
        symbols (list): chemical symbols of the atoms.
        atom_labels (list): labels of the atoms, e.g. ["O0", "H1", "H2"].
        coords (np.ndarray): coordinates (Å) used by the module (standard orientation).
        coords_input (np.ndarray): initial coordinates, centred on the center of mass.
        labels (list): labels of the classes of operations (columns of the character table).
        order (int): order h of the group (number of operations).
    """
    def __init__(self, symbols, coords, group=None, tol=0.05, lang="En",
                 standard_orientation=True, sym_tolerance=0.3, verbose=True):
        self.lang = lang.lower().capitalize()
        self.verbose = verbose
        detected = group is None
        self.symbols = list(symbols)
        X = np.asarray(coords, dtype=float)
        if X.shape != (len(self.symbols), 3):
            raise ValueError("coords must have the shape (n_atoms, 3)")
        X = X - self._center_of_mass(X)
        self.coords_input = X.copy()
        ops = None
        if group is None:
            group, X, ops = _pymatgen_analysis(self.symbols, X, sym_tolerance)
        group = _resolve_group(group, self.lang)
        if group in _LINEAR_GROUPS:
            raise ValueError("Linear molecules (C∞v, D∞h) are not handled: use the subgroups C2v and D2h.")
        if group not in _TABLES:
            raise ValueError(f"Unknown point group '{group}'. Available: {', '.join(_TABLES)}")
        self.group = group
        self.tol = tol
        self._orientation_status = None
        if standard_orientation:
            X = self._standard_orientation(X, ops, sym_tolerance)
        self.coords = X
        if verbose:
            self._creation_message(detected)
        self._table = _TABLES[group]
        self.order = sum(g for _, g, _ in self._table["classes"])
        self.labels = [f"{g}{lab}" if g > 1 else lab for lab, g, _ in self._table["classes"]]
        self.atom_labels = [f"{s}{k}" for k, s in enumerate(self.symbols)]
        self._check_symmetry()

    def _standard_orientation(self, X, ops, sym_tolerance):
        """
        Rotates the molecule into the standard orientation of its point group
        (as the "Standard orientation" of Gaussian) and symmetrizes it.
        The symmetry operations of the molecule are given by pymatgen, unless
        the molecule is already in a standard frame.
        """
        ref = _group_data(self.group)["elements"]
        already = _symmetrize(self.symbols, X, ref, self.tol) is not None
        if already:
            if self.group in ("T", "Th", "Td", "O", "Oh", "I", "Ih"):
                self._orientation_status = "already"
                return X
            ops = ref
        elif ops is None:
            try:
                _, X, ops = _pymatgen_analysis(self.symbols, X, sym_tolerance)
            except ImportError:
                if self.lang == "Fr":
                    print("pymatgen n'est pas installé : la molécule n'a pas pu être réorientée.")
                else:
                    print("pymatgen is not installed: the molecule could not be reoriented.")
                self._orientation_status = "failed"
                return X
        Xs = _orient(self.symbols, X, self.group, ops, tol=sym_tolerance)
        if Xs is None:
            if self.lang == "Fr":
                print(f"La molécule n'a pas pu être placée dans l'orientation standard de {self.group}.")
            else:
                print(f"The molecule could not be placed in the standard orientation of {self.group}.")
            self._orientation_status = "failed"
            return X
        self._orientation_status = "placed" if np.abs(Xs - X).max() > 1e-3 else "already"
        return Xs

    def _creation_message(self, detected):
        """Point group (stored in the attribute group) and orientation of the molecule."""
        g = self.group
        fr = self.lang == "Fr"
        if fr:
            txt = f"Groupe ponctuel {g}" + (" (détecté par pymatgen)" if detected else "") \
                  + ", enregistré dans l'attribut group."
            if self._orientation_status == "already":
                txt += f" Molécule déjà dans l'orientation standard de {g}."
            elif self._orientation_status == "placed":
                txt += f" Molécule placée dans l'orientation standard de {g} et symétrisée."
        else:
            txt = f"Point group {g}" + (" (detected by pymatgen)" if detected else "") \
                  + ", stored in the attribute group."
            if self._orientation_status == "already":
                txt += f" Molecule already in the standard orientation of {g}."
            elif self._orientation_status == "placed":
                txt += f" Molecule placed in the standard orientation of {g} and symmetrized."
        print(txt)

    @classmethod
    def from_molView(cls, mv, group=None, display_now=False, **kwargs):
        """
        Builds the molecule from a `molView` object (Chem3D module). `MolSym` does
        not read structures itself: molView reads a SMILES (3D structure from
        PubChem or generated by RDKit), a file (xyz, pdb, mol...), a PubChem CID...

            MolSym.from_molView(molView("O", source="smiles", display_now=False))
            MolSym.from_molView(molView("SF6_model.xyz", source="file", display_now=False))
            MolSym.from_molView(molView(962, source="cid", display_now=False))

        The point group is detected with pymatgen (unless given), then the
        molecule is placed in the standard orientation of its point group
        and symmetrized (3D structures from PubChem or RDKit are not exactly symmetric).

        Args:
            mv (molView): molecule loaded with molView.
            group (str): Schoenflies symbol; if None, detected with pymatgen.
            display_now (bool): if True, displays the molecule in 3D, in its standard
                      orientation (see `show`). False by default, since molView already
                      displays the molecule (its own `display_now` argument).
            **kwargs: passed to `MolSym` (tol, lang, standard_orientation, sym_tolerance, verbose).
        """
        symbols = [str(s) for s in mv.data.symbols]
        mol = cls(symbols, mv.data.positions, group, **kwargs)
        if display_now:
            mol.show()
        return mol

    def _center_of_mass(self, X):
        try:
            from rdkit.Chem import GetPeriodicTable
            pt = GetPeriodicTable()
            m = np.array([pt.GetAtomicWeight(s) for s in self.symbols])
        except Exception:
            m = np.ones(len(self.symbols))
        return m @ X / m.sum()

    def _matrix(self, op):
        """Returns the matrix of an operation given by its class label or as a 3x3 matrix."""
        if isinstance(op, str):
            for lab, _, M in self._table["classes"]:
                if op == lab:
                    return M
            names = [lab for lab, _, _ in self._table["classes"]]
            raise ValueError(f"Unknown operation '{op}' for {self.group}. Available: {', '.join(names)}")
        M = np.asarray(op, dtype=float)
        if M.shape != (3, 3):
            raise ValueError("An operation must be a class label or a 3x3 matrix")
        return M

    def apply(self, op):
        """
        Coordinates of the atoms after a symmetry operation.

        Args:
            op (str or np.ndarray): class label (e.g. "C2", "σv") or 3x3 matrix.

        Returns:
            np.ndarray: transformed coordinates, shape (n_atoms, 3).
        """
        return self.coords @ self._matrix(op).T

    def track(self, op, verbose=True):
        """
        Atom tracking: for each atom, the index of the atom on which it lands
        after the symmetry operation.

        Args:
            op (str or np.ndarray): class label (e.g. "C2", "σv") or 3x3 matrix.
            verbose (bool): if True, prints the tracking (e.g. H1 → H2).

        Returns:
            list: destination index of each atom, or None for an atom that lands
                  on no atom of the same element (the operation is then not a
                  symmetry operation of the molecule).
        """
        Y = self.apply(op)
        dest = []
        for k, y in enumerate(Y):
            found = None
            for j, x in enumerate(self.coords):
                if self.symbols[j] == self.symbols[k] and np.linalg.norm(x - y) < self.tol:
                    found = j
                    break
            dest.append(found)
        if verbose:
            name = op if isinstance(op, str) else "R"
            arrows = [f"{self.atom_labels[k]} → {self.atom_labels[j] if j is not None else '?'}"
                      for k, j in enumerate(dest)]
            print(f"{name} : " + ", ".join(arrows))
            if None in dest:
                if self.lang == "Fr":
                    print(f"{name} n'est pas une opération de symétrie de la molécule")
                else:
                    print(f"{name} is not a symmetry operation of the molecule")
        return dest

    def matrices(self, op=None, all_operations=False, exact=True):
        """
        Displays the matrices of the symmetry operations of the group (LaTeX).

        Args:
            op (str or list of str): class label(s) (e.g. "C3", ["C2", "σv"]); if None, all the classes.
            all_operations (bool): if False (default), one representative per class;
                                   if True, all the operations of the class(es), named by `identify`.
            exact (bool): if True, exact forms of the entries (1/2, √3/2...) when possible.
        """
        data = _group_data(self.group)
        labels = [lab for lab, _, _ in self._table["classes"]]
        selected = labels if op is None else ([op] if isinstance(op, str) else list(op))
        for o in selected:
            if o not in labels:
                raise ValueError(f"Unknown operation '{o}' for {self.group}. Available: {', '.join(labels)}")
            c = labels.index(o)
            if not all_operations:
                show_matrix(self._table["classes"][c][2], o, exact)
                continue
            for M, cl in zip(data["elements"], data["class_of"]):
                if cl == c:
                    show_matrix(M, None, exact)

    def is_symmetry(self, op):
        """True if the operation leaves the molecule unchanged."""
        return None not in self.track(op, verbose=False)

    def fixed_atoms(self, op):
        """Labels of the atoms that stay in place under the operation."""
        dest = self.track(op, verbose=False)
        return [self.atom_labels[k] for k, j in enumerate(dest) if j == k]

    def _check_symmetry(self):
        """Checks that all the operations of the group leave the molecule unchanged."""
        elements = _group_data(self.group)["elements"]
        bad = [M for M in elements if not self.is_symmetry(M)]
        if bad:
            orient = _ORIENTATION.get(self.group, "principal axis along z, one C2' axis or σv(xz) plane along x")
            if self.lang == "Fr":
                orient = _ORIENTATION_FR.get(self.group, "axe principal selon z, un axe C2' ou un plan σv(xz) selon x")
                print(f"Attention : {len(bad)} opération(s) de {self.group} ne laisse(nt) pas la molécule "
                      f"inchangée. La molécule n'appartient pas à ce groupe, ou elle n'est pas dans "
                      f"l'orientation standard ({orient}).")
            else:
                print(f"Warning: {len(bad)} operation(s) of {self.group} do not leave the molecule "
                      f"unchanged. The molecule does not belong to this group, or it is not in the "
                      f"standard orientation ({orient}).")

    def gamma_atoms(self, show=True):
        """
        Characters of the atomic representation Γ_atoms: for each class, the
        number of atoms that stay in place under the operation.

        Args:
            show (bool): if True, displays the table with the fixed atoms.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            np.ndarray: characters of Γ_atoms, one per class.
        """
        fixed = [self.fixed_atoms(lab) for lab, _, _ in self._table["classes"]]
        chi = np.array([len(f) for f in fixed])
        if show:
            row_fixed = "atomes immobiles" if self.lang == "Fr" else "fixed atoms"
            everything = "tous" if self.lang == "Fr" else "all"
            names = [everything if len(f) == len(self.symbols) else (", ".join(f) if f else "-")
                     for f in fixed]
            df = pd.DataFrame([names, chi],
                              index=[row_fixed, "Γ_atoms"], columns=self.labels)
            display(df)
        return None if show else chi

    def equivalent_atoms(self):
        """
        Sets of symmetry-equivalent atoms (orbits of the group).

        Returns:
            list of lists of atom indices.
        """
        data = _group_data(self.group)
        dests = [self.track(M, verbose=False) for M in data["elements"]]
        sets, seen = [], set()
        for k in range(len(self.symbols)):
            if k in seen:
                continue
            orbit = sorted({d[k] for d in dests if d[k] is not None})
            seen |= set(orbit)
            sets.append(orbit)
        return sets

    def _set_name(self, orbit):
        labels = [self.atom_labels[k] for k in orbit]
        if len(labels) <= 3:
            return ",".join(labels)
        if orbit == list(range(orbit[0], orbit[-1] + 1)):
            return f"{labels[0]}–{labels[-1]}"
        return ",".join(labels[:2]) + ",...," + labels[-1]

    def _partial_p_ok(self, shell):
        """True if the shell is s, p, d, f, or a set of p components that spans a representation."""
        if not re.fullmatch(r"p[xyz]{1,2}", shell):
            return True
        idx = ["xyz".index(c) for c in shell[1:]]
        keep = idx if len(idx) == 1 else [i for i in range(3) if i not in idx]
        return all(abs(abs(M[keep[0], keep[0]]) - 1) < 1e-6 for M in _group_data(self.group)["elements"])

    def gamma_orbitals(self, basis="valence", show=True):
        """
        Representation spanned by a basis of atomic orbitals, and its reduction
        for each set of equivalent atoms and each shell (s, p, d, f): the
        symmetries of the symmetry-adapted linear combinations (SALCs), i.e. the
        building blocks of a molecular orbital diagram.

        For an operation R, an atom that stays in place contributes χ_l(R):
        1 for s, the trace of R for p (p orbitals transform as x, y, z), etc.;
        an atom that moves contributes 0.

        Args:
            basis (str or dict): "valence" (default): s for H and He, s and p for
                the main group elements, s, p, d for the d block, s, p, d, f for the
                f block; or the same shells on all the atoms, e.g. "s", "pz", "sp",
                ["s", "px", "py"];
                or a dict giving the shells of each element, e.g.
                {"C": "sp", "H": "s"} or {"Fe": "spd", "C": "p", "H": ""}.
                "px", "py" or "pz" alone: the p orbitals perpendicular to the molecular
                plane, i.e. a π basis, e.g. {"C": "pz"} for benzene (molecular plane xy
                in the standard orientation of D6h), {"C": "px"} for a planar C2v
                molecule (molecular plane yz); "pxy", "pxz", "pyz": the p orbitals lying
                in the molecular plane (σ basis), e.g. {"C": "s pxy", "H": "s"} for benzene.
            show (bool): if True, displays the table of the characters and of the reductions.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            np.ndarray: characters of the total representation, one per class.
        """
        classes = self._table["classes"]
        rows, names, decomp = [], [], []
        total = np.zeros(len(classes))
        for orbit in self.equivalent_atoms():
            el = self.symbols[orbit[0]]
            if basis == "valence":
                shells = _valence_shells(el)
            elif isinstance(basis, str):
                shells = basis                      # same shells on all the atoms, e.g. "s", "pz"
            elif isinstance(basis, dict):
                shells = basis.get(el, "")
            else:
                shells = list(basis)                # same shells on all the atoms, e.g. ["s", "px", "py"]
            # partial p shells (px, py...) are kept separate when each of them spans a
            # representation (e.g. px and py in C2v), merged otherwise ("px py" -> "pxy" in D6h)
            tokens = _parse_shells(shells, merge=False)
            if not all(self._partial_p_ok(t) for t in tokens):
                tokens = _parse_shells(shells, merge=True)
            for shell in tokens:
                partial = shell not in _L
                if partial:
                    if not re.fullmatch(r"p(x|y|z|xy|xz|yz)", shell):
                        raise ValueError(f"Unknown shell '{shell}': use s, p, d, f, or px, py, pz, pxy, pxz, pyz")
                    idx = ["xyz".index(c) for c in shell[1:]]
                    other = [i for i in range(3) if i not in idx]
                    # the subspace is invariant if every operation keeps the complementary axis (or plane)
                    keep = idx if len(idx) == 1 else other
                    if any(abs(abs(M[keep[0], keep[0]]) - 1) > 1e-6 for M in _group_data(self.group)["elements"]):
                        raise ValueError(f"{shell} orbitals alone do not span a representation of {self.group}: "
                                         f"use pz for the p orbitals perpendicular to a molecular plane xy "
                                         f"(px for a plane yz...), pxy (or \"px py\") for the p orbitals in this plane")
                chi = []
                for _, _, M in classes:
                    n_fixed = sum(1 for k, j in enumerate(self.track(M, verbose=False)) if k in orbit and j == k)
                    if not partial:
                        chi.append(n_fixed * _chi_l(M, _L[shell]))
                    else:
                        chi.append(n_fixed * sum(M[i, i] for i in idx))
                chi = np.array(chi)
                total += chi
                rows.append(chi)
                names.append(f"{self._set_name(orbit)} ({shell})")
                decomp.append(self._decomposition(chi))
        if show:
            df = pd.DataFrame([[_fmt(c) for c in r] for r in rows] + [[_fmt(c) for c in total]],
                              index=names + ["Γ"], columns=self.labels)
            df["Γ ="] = decomp + [self._decomposition(total)]
            display(df)
        return None if show else total

    def _decomposition(self, chi):
        n = self.reduce(chi, show=False)
        if not isinstance(n, dict) or any(isinstance(v, float) for v in n.values()):
            return "?"
        return " + ".join(f"{v}{k}" if v > 1 else k for k, v in n.items())

    def bonds(self, scale=1.25):
        """
        Bonds detected from the distances: d(A-B) < scale x (r_cov(A) + r_cov(B)),
        with the covalent radii of RDKit.

        Returns:
            list of tuples (i, j), i < j.
        """
        try:
            from rdkit.Chem import GetPeriodicTable
            pt = GetPeriodicTable()
            r = [pt.GetRcovalent(pt.GetAtomicNumber(e)) for e in self.symbols]
        except Exception:
            r = [0.75] * len(self.symbols)
        out = []
        for i in range(len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                if np.linalg.norm(self.coords[i] - self.coords[j]) < scale * (r[i] + r[j]):
                    out.append((i, j))
        return out

    def gamma_bonds(self, bonds=None, show=True):
        """
        Representation spanned by bond stretchings (internal coordinates): for an
        operation R, a bond that is sent onto itself (possibly with its two ends
        exchanged) contributes 1, a bond that moves contributes 0. The reduction,
        given for each set of equivalent bonds, gives the symmetries of the
        stretching vibrational modes (e.g. the number of IR / Raman active
        ν(CO) bands of a metal carbonyl).

        Args:
            bonds (list): bonds to consider, as pairs of atom indices [(0, 1), ...],
                or as element pairs ["C-H", "C-O"]; if None, all the bonds detected
                from the distances (see `bonds()`).
            show (bool): if True, displays the table of the characters and of the reductions.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            np.ndarray: characters of the total representation, one per class.
        """
        detected = self.bonds()
        if bonds is None:
            selected = detected
        elif all(isinstance(b, str) for b in bonds):
            pairs = [tuple(sorted(b.split("-"))) for b in bonds]
            selected = [(i, j) for i, j in detected
                        if tuple(sorted((self.symbols[i], self.symbols[j]))) in pairs]
        else:
            selected = [tuple(sorted(b)) for b in bonds]
        if not selected:
            raise ValueError("no bond selected")
        data = _group_data(self.group)
        dests = [self.track(M, verbose=False) for M in data["elements"]]
        # sets of equivalent bonds
        sets, seen = [], set()
        for b in selected:
            if b in seen:
                continue
            orbit = sorted({tuple(sorted((d[b[0]], d[b[1]]))) for d in dests})
            missing = [o for o in orbit if o not in selected]
            if missing:
                raise ValueError(f"the bonds {missing} are equivalent to {b} by symmetry: include them")
            seen |= set(orbit)
            sets.append(orbit)
        classes = self._table["classes"]
        class_dests = [self.track(M, verbose=False) for _, _, M in classes]
        rows, names, decomp = [], [], []
        total = np.zeros(len(classes))
        for orbit in sets:
            chi = np.array([sum(1 for (i, j) in orbit if {d[i], d[j]} == {i, j}) for d in class_dests], dtype=float)
            total += chi
            rows.append(chi)
            i, j = orbit[0]
            names.append(f"{self.symbols[i]}–{self.symbols[j]} ({len(orbit)})")
            decomp.append(self._decomposition(chi))
        if show:
            df = pd.DataFrame([[_fmt(c) for c in r] for r in rows] + [[_fmt(c) for c in total]],
                              index=names + ["Γ"], columns=self.labels)
            df["Γ ="] = decomp + [self._decomposition(total)]
            display(df)
            self.activity(total)
        return None if show else total

    def _rep_table(self, rows, show_activity=None):
        """Displays rows (name, characters) with their decompositions."""
        df = pd.DataFrame([[_fmt(c) for c in chi] for _, chi in rows],
                          index=[name for name, _ in rows], columns=self.labels)
        df["Γ ="] = [self._decomposition(chi) for _, chi in rows]
        display(df)
        if show_activity is not None:
            self.activity(show_activity)

    def gamma_3N(self, show=True):
        """
        Representation spanned by the 3N cartesian displacements of the atoms:
        χ_3N(R) = (number of atoms that stay in place) x χ_xyz(R), where
        χ_xyz(R) = trace of the matrix of R.

        Args:
            show (bool): if True, displays Γ_atoms, Γ_xyz and Γ_3N with their reductions.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the characters.

        Returns:
            np.ndarray: characters of Γ_3N, one per class (show=False).
        """
        atoms = self.gamma_atoms(show=False).astype(float)
        xyz = np.array([np.trace(M) for _, _, M in self._table["classes"]])
        chi = atoms * xyz
        if show:
            self._rep_table([("Γ_atoms", atoms), ("Γ_xyz", xyz), ("Γ_3N", chi)])
            return None
        return chi

    def gamma_vib(self, show=True):
        """
        Representation of the 3N-6 vibrational modes (non-linear molecule):
        Γ_vib = Γ_3N - Γ_trans - Γ_rot, with χ_trans(R) = trace(R) (x, y, z) and
        χ_rot(R) = det(R) x trace(R) (Rx, Ry, Rz: axial vectors), followed by the
        IR / Raman activity of the modes.

        Args:
            show (bool): if True, displays Γ_3N, Γ_trans, Γ_rot and Γ_vib with their
                         reductions, and the IR / Raman activity.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the characters.

        Returns:
            np.ndarray: characters of Γ_vib, one per class (show=False).
        """
        g3n = self.gamma_3N(show=False)
        trans = np.array([np.trace(M) for _, _, M in self._table["classes"]])
        rot = np.array([np.linalg.det(M) * np.trace(M) for _, _, M in self._table["classes"]])
        vib = g3n - trans - rot
        if show:
            self._rep_table([("Γ_3N", g3n), ("Γ_trans", trans), ("Γ_rot", rot), ("Γ_vib", vib)],
                            show_activity=vib)
            return None
        return vib

    def _basis_rep(self, M, atoms, shell):
        """Matrix of the operation M in a basis of one orbital (s, px, py or pz) per atom of `atoms`."""
        dest = self.track(M, verbose=False)
        f = 1.0 if shell == "s" else M["xyz".index(shell[1]), "xyz".index(shell[1])]
        pos = {a: k for k, a in enumerate(atoms)}
        D = np.zeros((len(atoms), len(atoms)))
        for k, a in enumerate(atoms):
            if dest[a] not in pos:
                raise ValueError(f"atom {self.atom_labels[a]} is sent onto {self.atom_labels[dest[a]]}, "
                                 f"which is not in the basis: include all the equivalent atoms")
            D[pos[dest[a]], k] = f
        return D

    def _basis_functions_list(self, basis, atoms=None):
        """
        List of the basis functions (atom index, shell, component) for a basis given
        as in `gamma_orbitals` ("valence", "s", "sp", "spd", "pz", ["s", "px"], or a
        dict per element). Supported shells: s, p, d and the p components px, py, pz.
        """
        comps = {"s": ["s"], "p": ["px", "py", "pz"],
                 "d": ["dz²", "dxz", "dyz", "dx²-y²", "dxy"],
                 "f": list(_F_ORBITALS),
                 "px": ["px"], "py": ["py"], "pz": ["pz"]}
        if atoms is None:
            atoms = list(range(len(self.symbols)))
        funcs = []
        for a in atoms:
            el = self.symbols[a]
            if basis == "valence":
                shells = _valence_shells(el)
            elif isinstance(basis, str):
                shells = basis
            elif isinstance(basis, dict):
                shells = basis.get(el, "")
            else:
                shells = list(basis)
            for sh in _parse_shells(shells):
                if sh in comps:
                    funcs += [(a, c) for c in comps[sh]]
                elif re.fullmatch(r"p[xyz]{1,2}", sh):
                    funcs += [(a, "p" + c) for c in sh[1:]]
                else:
                    raise ValueError(f"Shell '{sh}' not supported here: use s, p, d, f, or px, py, pz, pxy...")
        return funcs

    @staticmethod
    def _component_matrix(M, comps):
        """
        Matrix of the operation M acting on the orbital components of one atom
        (active transformation): p orbitals transform as x, y, z (matrix M);
        d orbitals as the quadratic functions r^T Q r, Q -> M Q M^T; f orbitals
        as cubic functions (decomposition computed numerically).
        """
        r6, r2 = np.sqrt(6), np.sqrt(2)
        Q = {"dz²": np.diag([-1, -1, 2]) / r6, "dx²-y²": np.diag([1, -1, 0]) / r2}
        for name, (i, j) in (("dxy", (0, 1)), ("dxz", (0, 2)), ("dyz", (1, 2))):
            q = np.zeros((3, 3)); q[i, j] = q[j, i] = 1 / r2
            Q[name] = q
        idx = {"px": 0, "py": 1, "pz": 2}
        fi = [k for k, c in enumerate(comps) if c in _F_ORBITALS]
        if fi:
            # f orbitals: (O_M f)(r) = f(M^T r), decomposed on the f basis by least squares;
            # the other shells are treated separately (block-diagonal matrix)
            fc = [comps[k] for k in fi]
            if len(fc) != len(_F_ORBITALS):
                raise ValueError("f orbitals must be given as a complete shell")
            pts = np.random.default_rng(0).normal(size=(60, 3))
            F = np.array([[_F_ORBITALS[c](r) for c in fc] for r in pts])
            G = np.array([[_F_ORBITALS[c](M.T @ r) for c in fc] for r in pts])
            D = np.zeros((len(comps), len(comps)))
            D[np.ix_(fi, fi)] = np.linalg.lstsq(F, G, rcond=None)[0]
            oi = [k for k in range(len(comps)) if k not in fi]
            if oi:
                D[np.ix_(oi, oi)] = MolSym._component_matrix(M, [comps[k] for k in oi])
            return D
        def image(c):
            """Coefficients of the image of component c on all the components."""
            if c == "s":
                return {"s": 1.0}
            if c in idx:
                return {k: M[idx[k], idx[c]] for k in idx}
            Qp = M @ Q[c] @ M.T
            return {k: float(np.sum(Qp * Q[k])) for k in Q}
        D = np.zeros((len(comps), len(comps)))
        for col, c in enumerate(comps):
            img = image(c)
            for k, v in img.items():
                if abs(v) > 1e-9:
                    if k not in comps:
                        raise ValueError(f"the orbital {c} is transformed into {k}, which is not in "
                                         f"the basis: complete the shell (e.g. 'p' instead of '{c}')")
                    D[comps.index(k), col] += v
        return D

    def representation_matrix(self, op, basis="s", atoms=None, show=True):
        """
        Matrix of a symmetry operation in a basis of atomic orbitals. For a basis
        of s orbitals, it is the permutation matrix of the atoms; for p orbitals,
        each block is the 3x3 matrix of the operation itself. The trace of the
        matrix is the character of the operation.

        Args:
            op (str or np.ndarray): class label (e.g. "C2") or 3x3 matrix.
            basis (str, list or dict): orbitals on each atom, as in `gamma_orbitals`:
                "s" (default), "p", "sp", "d", "f", "valence", "pz", ["s", "px"],
                or a dict per element, e.g. {"O": "sp", "H": "s"}.
            atoms (list): atoms carrying the basis orbitals (default: all the atoms).
            show (bool): if True, displays the matrix (LaTeX), the basis and the trace;
                         if False, returns the matrix without display.

        Returns:
            np.ndarray: the matrix (show=False).
        """
        M = self._matrix(op)
        funcs = self._basis_functions_list(basis, atoms)
        if not funcs:
            raise ValueError("empty basis")
        dest = self.track(M, verbose=False)
        by_atom = {}
        for k, (a, c) in enumerate(funcs):
            by_atom.setdefault(a, []).append((c, k))
        n = len(funcs)
        D = np.zeros((n, n))
        for a, items in by_atom.items():
            b = dest[a]
            if b not in by_atom or [c for c, _ in by_atom[b]] != [c for c, _ in items]:
                raise ValueError(f"atom {self.atom_labels[a]} is sent onto {self.atom_labels[b]}: "
                                 f"both must carry the same orbitals (include all the equivalent atoms)")
            comps = [c for c, _ in items]
            block = self._component_matrix(M, comps)
            rows = [k for _, k in by_atom[b]]
            cols = [k for _, k in items]
            D[np.ix_(rows, cols)] = block
        if show:
            from IPython.display import Math
            basis_txt = ", ".join(f"{c}({self.atom_labels[a]})" for a, c in funcs)
            name = op if isinstance(op, str) else identify(op)
            rows_tex = r" \\ ".join(" & ".join(_latex_entry(x) for x in row) for row in D)
            tr = _fmt(np.trace(D))
            display(Math(_latex_label(name) + r" \rightarrow \begin{pmatrix} " + rows_tex
                         + r" \end{pmatrix} \quad \chi(" + _latex_label(name) + ") = " + tr))
            print(("base : " if self.lang == "Fr" else "basis: ") + basis_txt)
            return None
        return D

    def orbital_symmetry(self, C, atoms=None, shell="pz", tol=1e-4, show=True):
        """
        Irreducible representation of a molecular orbital (or of a set of
        degenerate MOs) expanded on ONE atomic orbital per atom, all of the same
        type (s, px, py or pz): e.g. a Hückel π MO on the pz orbitals, or an MO
        built from the 1s orbitals of hydrogen atoms. The MOs may come from any
        calculation (Hückel, or the corresponding coefficients of an ab initio /
        DFT calculation), as long as the basis is one orbital per atom: MOs mixing
        several orbitals on the same atom (s, p and d...) are not handled.

        Args:
            C (array-like): coefficients of the MO, 1D array of length n (one
                coefficient per atom of `atoms`, in the same order), or of a set of
                degenerate MOs, 2D array of shape (n, number of MOs), one column per MO.
            atoms (list): indices (in the molecule) of the n atoms carrying the basis
                orbitals, in the order of the coefficients (default: all the atoms).
                All the atoms equivalent by symmetry must be included.
            shell (str): "s", "px", "py" or "pz": orbital on each atom (for a π system,
                the p orbital perpendicular to the molecular plane, see `orientation()`).
            tol (float): tolerance used to check that the MOs span a representation
                (numerical coefficients of a quantum chemistry code are not exact).
            show (bool): if True, prints the characters and the irrep.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            str: the irreducible representation(s), e.g. "E1g" (show=False).
        """
        C = np.asarray(C, dtype=float)
        if C.ndim == 1:
            C = C[:, None]
        if atoms is None:
            atoms = list(range(len(self.symbols)))
        if C.shape[0] != len(atoms):
            raise ValueError(f"{C.shape[0]} coefficients for {len(atoms)} atoms")
        if shell not in ("s", "px", "py", "pz"):
            raise ValueError("shell must be 's', 'px', 'py' or 'pz'")
        Cp = np.linalg.pinv(C)
        chi = []
        for _, _, M in self._table["classes"]:
            if shell != "s":
                i = "xyz".index(shell[1])
                if any(abs(abs(R[i, i]) - 1) > 1e-6 for R in _group_data(self.group)["elements"]):
                    raise ValueError(f"{shell} orbitals do not span a representation of {self.group}: "
                                     f"use the p orbital perpendicular to the molecular plane (see orientation())")
            D = self._basis_rep(M, atoms, shell)
            DC = D @ C
            if not np.allclose(C @ (Cp @ DC), DC, atol=tol):
                msg = ("ces OM ne forment pas une représentation : ajoutez les OM dégénérées"
                       if self.lang == "Fr" else
                       "these MOs do not span a representation: include all the degenerate MOs")
                raise ValueError(msg)
            chi.append(np.trace(Cp @ DC))
        chi = np.array(chi)
        irrep = self._decomposition(chi)
        if show:
            print(" ".join(f"{lab}:{_fmt(c)}" for lab, c in zip(self.labels, chi)) + f"  ->  {irrep}")
            return None
        return irrep

    def orbital_symmetries(self, coeffs, energies, atoms=None, shell="pz", tol=1e-4, show=True):
        """
        Irreducible representations of all the MOs of a set, degenerate MOs being
        grouped. The MOs are expanded on ONE atomic orbital per atom, all of the
        same type (s, px, py or pz), e.g. the π MOs of a Hückel calculation; see
        `orbital_symmetry()`.

        Args:
            coeffs (array-like): coefficients, 2D array of shape (n, number of MOs):
                one row per atom of `atoms` (same order), one column per MO
                (e.g. HMO.c_MOs).
            energies (array-like): energies of the MOs, one per column of coeffs
                (or Hückel x values, e.g. HMO.E_MOs), used to group the degenerate MOs;
                they do not need to be sorted.
            atoms (list): indices (in the molecule) of the n atoms carrying the basis
                orbitals, in the order of the rows of coeffs (default: all the atoms).
            shell (str): "s", "px", "py" or "pz".
            tol (float): two MOs are degenerate if their energies differ by less than
                tol; also used to check the symmetry of the MOs.
            show (bool): if True, displays the table. With show=True, nothing is
                         returned; use show=False to get the DataFrame.

        Returns:
            pandas.DataFrame: MOs (column indices), energy, irrep (show=False).
        """
        coeffs = np.asarray(coeffs, dtype=float)
        energies = np.asarray(energies, dtype=float)
        if coeffs.ndim != 2 or coeffs.shape[1] != len(energies):
            raise ValueError("coeffs must be a 2D array with one column per MO, "
                             "and energies must give one energy per column")
        groups = []
        for k, e in enumerate(energies):
            for g in groups:
                if abs(energies[g[0]] - e) < tol:
                    g.append(k)
                    break
            else:
                groups.append([k])
        rows = []
        for group in groups:
            irrep = self.orbital_symmetry(coeffs[:, group], atoms, shell, tol=tol, show=False)
            rows.append({"MO": ", ".join(str(g) for g in group),
                         "energy": round(float(energies[group[0]]), 4), "irrep": irrep})
        df = pd.DataFrame(rows)
        if show:
            display(df)
            return None
        return df

    def activity(self, chi, show=True):
        """
        IR and Raman activity of the irreducible representations contained in a
        representation: IR active if the irrep transforms as x, y or z, Raman
        active if it transforms as a quadratic function (x², xy...).

        Args:
            chi (array-like): characters of the representation, one per class.
            show (bool): if True, prints the active irreps.
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            dict: {"IR": [...], "Raman": [...]}, irreps with their multiplicities.
        """
        return ir_raman_activity(self.group, chi, show=show, lang=self.lang)

    def flowchart(self):
        """Flowchart (decision tree) of the point groups, with the path leading to the group of the molecule."""
        return point_group_flowchart(self.group, lang=self.lang)

    def orientation(self, show=True):
        """
        Describes the orientation of the molecule in the cartesian frame: shape
        (linear, planar or not), molecular plane (xy, xz, yz or other) or axis,
        and principal axis of the point group.

        Args:
            show (bool): if True, prints the description and returns nothing;
                         if False, returns the dict without printing.

        Returns:
            dict: {"shape": ..., "plane": ..., "axis": ..., "principal_axis": ...}
        """
        X = self.coords - self.coords.mean(axis=0)
        _, sv, vt = np.linalg.svd(X, full_matrices=True)
        sv = np.concatenate([sv, np.zeros(3 - len(sv))])
        info = {"shape": None, "plane": None, "axis": None, "principal_axis": None}
        if len(X) == 1:
            info["shape"] = "atom"
        elif sv[1] < 1e-3:
            info["shape"] = "linear"
            info["axis"] = _axis_name(vt[0] * np.sign(next(c for c in vt[0] if abs(c) > 1e-6)))
        elif sv[2] < 1e-3:
            info["shape"] = "planar"
            n = vt[2] * np.sign(next(c for c in vt[2] if abs(c) > 1e-6))
            planes = {"z": "xy", "y": "xz", "x": "yz"}
            name = _axis_name(n)
            info["plane"] = planes.get(name, f"⊥({name})")
        else:
            info["shape"] = "non planar"
        # principal axis: proper rotation of highest order (z in the standard orientation)
        best = None
        for M in _group_data(self.group)["elements"]:
            kind, u, order = self._element_of(M)
            if kind == "axis" and (best is None or order > best[0]):
                best = (order, u)
        if best is not None:
            info["principal_axis"] = f"C{best[0]}({_axis_name(best[1])})"
        if self.group in ("T", "Th", "Td", "O", "Oh", "I", "Ih"):
            # no unique principal axis: give the convention of the standard orientation
            orient = _ORIENTATION_FR if self.lang == "Fr" else _ORIENTATION
            info["principal_axis"] = orient[self.group]
        if show:
            fr = self.lang == "Fr"
            shape = {"atom": "atome", "linear": "linéaire", "planar": "plane", "non planar": "non plane"}
            txt = (f"{self.group} : molécule {shape[info['shape']]}" if fr
                   else f"{self.group}: {info['shape']} molecule")
            if info["plane"]:
                txt += (f", dans le plan {info['plane']}" if fr else f", in the {info['plane']} plane")
            if info["axis"]:
                txt += (f", selon l'axe {info['axis']}" if fr else f", along the {info['axis']} axis")
            if info["principal_axis"]:
                txt += (f" ; axe principal : {info['principal_axis']}" if fr
                        else f"; principal axis: {info['principal_axis']}")
            print(txt)
        return None if show else info

    def character_table(self, gamma=True, functions=True, complex=False):
        """
        Character table of the point group, as a pandas DataFrame.

        Args:
            gamma (bool): if True, adds the row Γ_atoms at the bottom of the table.
            functions (bool): if True, adds the columns of the basis functions.
            complex (bool): if True, complex form of the pairs of complex conjugate irreps.
        """
        df = character_table(self.group, functions=functions, complex=complex)
        if gamma:
            row = [str(c) for c in self.gamma_atoms(show=False)]
            row += [""] * (df.shape[1] - len(row))
            df.loc["Γ_atoms"] = row
        return df

    def reduce(self, chi=None, show=True):
        """
        Reduction of a representation into irreducible representations:
        n_i = (1/h) Σ_classes g_c χ(c) χ_i(c).
        A pair of complex conjugate irreps (row marked * in the character table)
        is counted as a whole.

        Args:
            chi (array-like): characters of the representation, one per class
                              (default: Γ_atoms of the molecule).
            show (bool): if True, prints the decomposition (e.g. Γ = 2A1 + B2).
                         With show=True, nothing is returned (no echo in a notebook);
                         use show=False to get the result.

        Returns:
            dict: number of times each irreducible representation appears.
        """
        if chi is None:
            chi = self.gamma_atoms(show=False)
        return reduce_representation(self.group, chi, show=show, lang=self.lang)

    # ------------------------------------------------------------------
    #                 3D visualization (py3Dmol)
    # ------------------------------------------------------------------
    _AXIS_COLORS = {2: "#d62728", 3: "#2ca02c", 4: "#1f77b4", 5: "#ff7f0e",
                    6: "#9467bd", 8: "#8c564b"}

    @staticmethod
    def _element_of(M):
        """
        Geometric symmetry element of a 3x3 operation matrix:
        ('center', None), ('plane', normal), ('axis', direction, order) or
        ('improper', direction, order).
        """
        if np.allclose(M, -np.eye(3)):
            return ("center", None, None)
        w, v = np.linalg.eig(M)
        det = np.linalg.det(M)
        target = 1 if det > 0 else -1
        k = int(np.argmin(np.abs(w - target)))
        u = np.real(v[:, k])
        u = u / np.linalg.norm(u)
        # unique orientation of the direction (first non-zero component positive)
        s = next(c for c in u if abs(c) > 1e-6)
        u = u * np.sign(s)
        order, P = 1, M.copy()
        while not np.allclose(P, np.eye(3)) and order < 24:
            P, order = P @ M, order + 1
        if det > 0:
            return ("axis", u, order)
        if np.isclose(np.trace(M), 1):
            return ("plane", u, None)
        return ("improper", u, order)

    def symmetry_elements(self, op=None):
        """
        Symmetry elements of the molecule (axes, planes, inversion center).

        Args:
            op (str or list of str): class label(s) (e.g. "C3", ["S4", "C2"]):
                      elements of these classes only. If None, all the elements:
                      proper axes (labelled by their highest order), planes and
                      inversion center.

        Returns:
            list of tuples (type, vector, label), type being 'axis', 'improper',
            'plane' or 'center'.
        """
        data = _group_data(self.group)
        labels = [lab for lab, _, _ in self._table["classes"]]
        selected = None if op is None else ([op] if isinstance(op, str) else list(op))
        for o in selected or []:
            if o not in labels:
                raise ValueError(f"Unknown operation '{o}' for {self.group}. Available: {', '.join(labels)}")
        found = {}
        for M, c in zip(data["elements"], data["class_of"]):
            if labels[c] == "E" or (selected is not None and labels[c] not in selected):
                continue
            kind, u, order = self._element_of(M)
            if selected is None and kind == "improper":
                continue
            key = (kind, None if u is None else tuple(np.round(u, 4)))
            if selected is not None:
                found[(labels[c],) + key] = (kind, u, labels[c])
            elif kind == "axis":
                old = found.get(key)
                if old is None or order > old[3]:
                    found[key] = (kind, u, f"C{order}", order)
            else:
                found[key] = (kind, u, "i" if kind == "center" else "σ", 0)
        return [f[:3] for f in found.values()]

    def _xyz_block(self, X):
        lines = [f"{s} {x:.5f} {y:.5f} {z:.5f}" for s, (x, y, z) in zip(self.symbols, X)]
        return f"{len(lines)}\n\n" + "\n".join(lines)

    def _draw_molecule(self, view, X, labels=True, viewer=None):
        kw = {} if viewer is None else {"viewer": viewer}
        view.addModel(self._xyz_block(X), "xyz", **kw)
        view.setStyle({}, {"stick": {"radius": 0.12, "colorscheme": "Jmol"},
                           "sphere": {"scale": 0.25, "colorscheme": "Jmol"}}, **kw)
        if labels:
            for lab, p in zip(self.atom_labels, X):
                view.addLabel(lab, {"position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                                    "fontSize": 12, "fontColor": "black", "backgroundColor": "white",
                                    "backgroundOpacity": 0.6, "inFront": True}, **kw)

    def _draw_element(self, view, element, size, viewer=None):
        kw = {} if viewer is None else {"viewer": viewer}
        kind, u, label = element
        if kind == "center":
            view.addSphere({"center": {"x": 0, "y": 0, "z": 0}, "radius": 0.15, "color": "black"}, **kw)
            return
        if kind == "plane":
            a = np.cross(u, [1, 0, 0] if abs(u[0]) < 0.9 else [0, 1, 0])
            a /= np.linalg.norm(a)
            b = np.cross(u, a)
            corners = [size * (sa * a + sb * b) for sa, sb in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            verts = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in corners]
            norm = {"x": float(u[0]), "y": float(u[1]), "z": float(u[2])}
            view.addCustom({"vertexArr": verts, "normalArr": [norm] * 4,
                            "faceArr": [0, 1, 2, 0, 2, 3],
                            "color": "#2171b5", "opacity": 0.55}, **kw)
            return
        m = re.match(r"[CS](\d+)", label)
        order = int(m.group(1)) if m else 2
        color = self._AXIS_COLORS.get(order, "#7f7f7f")
        start, end = -size * 1.15 * u, size * 1.15 * u
        spec = {"start": {"x": float(start[0]), "y": float(start[1]), "z": float(start[2])},
                "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
                "radius": 0.05, "color": color, "fromCap": True, "toCap": True}
        if kind == "improper":
            spec["dashed"] = True
        view.addCylinder(spec, **kw)
        tip = start if kind == "improper" else end      # Sn and Cn axes may coincide
        view.addLabel(label, {"position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
                              "fontSize": 14, "fontColor": color, "backgroundOpacity": 0.0}, **kw)

    def _draw_axes(self, view, size, viewer=None):
        """Cartesian axes x, y, z (thin grey arrows from the origin)."""
        kw = {} if viewer is None else {"viewer": viewer}
        L = 0.8 * size
        for name, e in zip("xyz", np.eye(3)):
            end = L * e
            view.addArrow({"start": {"x": 0, "y": 0, "z": 0},
                           "end": {"x": float(end[0]), "y": float(end[1]), "z": float(end[2])},
                           "radius": 0.025, "radiusRatio": 2.5, "mid": 0.9, "color": "#555555"}, **kw)
            tip = 1.08 * end
            view.addLabel(name, {"position": {"x": float(tip[0]), "y": float(tip[1]), "z": float(tip[2])},
                                 "fontSize": 13, "fontColor": "#333333", "backgroundOpacity": 0.0}, **kw)

    def _set_view(self, view, viewer=None):
        """
        Initial view: a planar molecule is seen face-on (slightly tilted), with z
        vertical when possible; otherwise an oblique view with z vertical.
        """
        kw = {} if viewer is None else {"viewer": viewer}
        plane = self.orientation(show=False)["plane"]
        if plane == "yz":
            rots = [(-90, "x"), (-90, "z"), (20, "y"), (-10, "x")]
        elif plane == "xz":
            rots = [(-90, "x"), (20, "y"), (-10, "x")]
        elif plane == "xy":
            rots = [(-30, "x")]
        else:
            rots = [(-70, "x"), (20, "y")]
        for angle, axis in rots:
            view.rotate(angle, axis, **kw)

    def _size(self):
        return float(np.max(np.linalg.norm(self.coords, axis=1))) + 1.0

    def show(self, labels=False, axes=False, width=500, height=400):
        """
        Interactive 3D view (py3Dmol) of the molecule alone, in its standard orientation.

        Args:
            labels (bool): if True, shows the atom labels (O0, H1...).
            axes (bool): if True, shows the cartesian axes x, y, z.
            width, height (int): size of the viewer in pixels.
        """
        import py3Dmol
        view = py3Dmol.view(width=width, height=height)
        self._draw_molecule(view, self.coords, labels=labels)
        if axes:
            self._draw_axes(view, self._size())
        view.zoomTo()
        self._set_view(view)
        self.view = view
        view.show()

    def show_elements(self, op=None, labels=False, axes=True, width=500, height=400):
        """
        Interactive 3D view (py3Dmol) of the molecule and of its symmetry elements.

        Args:
            op (str or list of str): class label(s) (e.g. "C3", ["S4", "C2"]):
                      shows only the elements of these classes. If None, shows all the proper axes, planes and the
                      inversion center (can be crowded for high symmetry groups).
            labels (bool): if True, shows the atom labels (O0, H1...).
            width, height (int): size of the viewer in pixels.

        Colors: C2 red, C3 green, C4 blue, C5 orange, C6 purple, C8 brown;
        Sn axes dashed; planes in translucent blue; inversion center in black.
        """
        import py3Dmol
        view = py3Dmol.view(width=width, height=height)
        self._draw_molecule(view, self.coords, labels=labels)
        if axes:
            self._draw_axes(view, self._size())
        size = self._size()
        for element in self.symmetry_elements(op):
            self._draw_element(view, element, size)
        view.zoomTo()
        self._set_view(view)
        self.view = view
        view.show()

    def show_operation(self, op, highlight=None, axes=True, width=700, height=350):
        """
        Interactive 3D view (py3Dmol) of the molecule before (left) and after
        (right) a symmetry operation. Each atom keeps its label, so that the
        atom tracking can be read on the figure. The two views are linked.

        Args:
            op (str): class label (e.g. "C3", "σv"); the representative of the class is used.
            highlight (int or list of int): atom(s) to follow: a translucent yellow
                sphere marks the atom before the operation (left) and the place where
                it arrives after the operation (right).
            axes (bool): if True, shows the cartesian axes x, y, z.
            width, height (int): size of the viewer in pixels.
        """
        import py3Dmol
        M = self._matrix(op)
        kind, u, order = self._element_of(M)
        element = (kind, u, op)
        view = py3Dmol.view(width=width, height=height, viewergrid=(1, 2), linked=True)
        size = self._size()
        self._draw_molecule(view, self.coords, viewer=(0, 0))
        self._draw_molecule(view, self.apply(op), viewer=(0, 1))
        if highlight is not None:
            marks = [highlight] if isinstance(highlight, (int, np.integer)) else list(highlight)
            after = self.apply(op)
            for k in marks:
                for X, v in ((self.coords, (0, 0)), (after, (0, 1))):
                    p = X[k]
                    view.addSphere({"center": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
                                    "radius": 0.55, "color": "#ffcc00", "opacity": 0.45}, viewer=v)
        if axes:
            for v in ((0, 0), (0, 1)):
                self._draw_axes(view, size, viewer=v)
        for v in ((0, 0), (0, 1)):
            self._draw_element(view, element, size, viewer=v)
        if self.lang == "Fr":
            print(f"À gauche : la molécule de départ. À droite : la molécule après l'opération {op}.")
        else:
            print(f"Left: initial molecule. Right: molecule after the operation {op}.")
        self.track(op)
        view.zoomTo()
        for v in ((0, 0), (0, 1)):
            self._set_view(view, viewer=v)
        self.view = view
        view.show()
