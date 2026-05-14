"""
Triangle impedance lens: m=0 radial Helmholtz outside an equilateral triangle in 2D.

For each polar direction ``theta``, the inner radius ``r_min(theta)`` hits the
triangle; the field in ``r > r_min`` is modeled as a linear combination of
``H_0^{(1)}(kr)`` and ``H_0^{(2)}(kr)`` (exact solutions of the radial m=0 Helmholtz
equation). Coefficients are fixed by a Robin condition at ``r_min`` and by matching
the reference outgoing Hankel ``H_0^{(1)}(kr_max)`` at the outer radius (numerically
replaces Sommerfeld while excluding the trivial ``psi=0`` nullspace of the fully
homogeneous two-point problem).

Transmission ``T`` for the Weinberg chain uses **geometric optics** with an interior
medium boosted by ``f_C^3`` (``G_eff = G_{\\mathrm{shear}} f_C^3``,
``c_{\\mathrm{eff}} = c\\sqrt{f_C^3}``, impedances ``Z_{\\mathrm{in}}=\\rho_V c_{\\mathrm{eff}}``,
``Z_{\\mathrm{out}}=\\rho_V c``) and normal-incidence Fresnel power coefficients; rays
approach each wall in sub-steps of length ``c_{\\mathrm{eff}}\\,\\Delta t`` summing to the
geometric chord. A legacy **geom-only** reference keeps the previous
``Z_{\\mathrm{out}}/Z_{\\mathrm{in}}=0.85`` oblique-Fresnel model for comparison.

The idealized radial Helmholtz ``H_0^{(1)}+H_0^{(2)}`` construction with inner Robin +
outer Dirichlet on ``psi`` alone enforces ``psi(r_{\\max})=H_0^{(1)}`` but generically
yields **negligible radial Poynting** ``\\mathrm{Im}(\\bar\\psi\\,\\partial_r\\psi)`` at
``r_{\\max}`` (destructive cancellation in the derivative while fixing amplitude). A
high-precision ``mpmath`` evaluation of that flux is still returned as
``T_helmholtz_radial_flux`` for diagnostics.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

try:
    from scipy import special
except ImportError as exc:  # pragma: no cover
    raise ImportError("triangle_impedance_lens requires scipy.special") from exc

try:
    import mpmath as _mpmath  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    _mpmath = None

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from constants import G_shear, c, rho_V  # noqa: E402
except ImportError:  # pragma: no cover
    G_shear = 4.94e10  # Pa
    rho_V = 5.50e-7  # kg/m^3
    c = 2.998e8  # m/s

# User / model inputs not all in constants.py
LAMBDA_SHOCK_M = 9.7e-17  # m
GAMMA_W_EFF_OVER_OMEGA_E = 0.537
K_EFF = 2.0
ETA_TRANSMISSION = 0.42
SIN2_THETA_W_EXP = 0.231

# Geometry
D_FM = 1.64e-15  # m, equilateral side
A_STRING_M = 0.01e-15  # m, string radius (display / future BC refinement)

# Discretization (angular quadrature for far-field flux)
N_THETA = 360
N_THETA_GO = 720
R_MAX_FACTOR = 10.0  # r_max = R_MAX_FACTOR * d


def triangle_vertices(d: float) -> np.ndarray:
    """Vertices v0, v1, v2 as rows (x, y); equilateral, centered at origin."""
    s3 = math.sqrt(3.0)
    v0 = np.array([0.0, d / s3], dtype=np.float64)
    v1 = np.array([-0.5 * d, -d / (2.0 * s3)], dtype=np.float64)
    v2 = np.array([0.5 * d, -d / (2.0 * s3)], dtype=np.float64)
    return np.stack([v0, v1, v2], axis=0)


def _intersect_ray_segment(
    origin: np.ndarray,
    direction: np.ndarray,
    p0: np.ndarray,
    p1: np.ndarray,
    *,
    eps: float = 1e-14,
) -> float | None:
    """Smallest t >= 0 with origin + t * direction on segment p0--p1."""
    d = np.asarray(direction, dtype=np.float64).reshape(2)
    dn = np.linalg.norm(d)
    if dn < eps:
        return None
    d = d / dn
    e = p1 - p0
    # origin + t d = p0 + s e  ->  [d, -e] [t; s] = p0 - origin
    b = p0 - origin
    mat = np.array([[d[0], -e[0]], [d[1], -e[1]]], dtype=np.float64)
    det = float(np.linalg.det(mat))
    scale = float(np.linalg.norm(d) * np.linalg.norm(e))
    if abs(det) < 100.0 * np.finfo(float).eps * max(scale, 1e-300):
        return None
    try:
        ts = np.linalg.solve(mat, b)
    except np.linalg.LinAlgError:
        return None
    t, s = float(ts[0]), float(ts[1])
    # Positive along-ray distance; exclude grazing double hits at t~0
    if t <= 1e-12 * max(np.linalg.norm(p0), np.linalg.norm(p1), 1e-30):
        return None
    if -1e-9 <= s <= 1.0 + 1e-9:
        return t
    return None


def r_min_along_ray(theta: float, verts: np.ndarray) -> float:
    """Distance from origin to triangle boundary along direction (cos θ, sin θ)."""
    d = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
    ts: list[float] = []
    for i in range(3):
        p0 = verts[i]
        p1 = verts[(i + 1) % 3]
        t = _intersect_ray_segment(np.zeros(2), d, p0, p1)
        if t is not None and t > 1e-20:
            ts.append(t)
    if not ts:
        raise RuntimeError(f"no triangle intersection for theta={theta}")
    return min(ts)


def hankel_radial_Z(r_min: float, r_max: float, k: float, alpha: float) -> tuple[complex, complex]:
    """
    psi(r) = Z1 H0^(1)(kr) + Z2 H0^(2)(kr) with
      (psi' + alpha psi)|_{r_min} = 0,
      psi(r_max) = H0^(1)(k r_max).
    """
    zi = k * r_min
    zo = k * r_max
    h1i, h2i = special.hankel1(0, zi), special.hankel2(0, zi)
    dh1i = -k * special.hankel1(1, zi)
    dh2i = -k * special.hankel2(1, zi)
    a1 = dh1i + alpha * h1i
    a2 = dh2i + alpha * h2i
    h1o, h2o = special.hankel1(0, zo), special.hankel2(0, zo)
    mat = np.array([[h1o, h2o], [a1, a2]], dtype=np.complex128)
    rhs = np.array([h1o, 0.0], dtype=np.complex128)
    try:
        z1, z2 = np.linalg.solve(mat, rhs)
    except np.linalg.LinAlgError:
        return 0.0j, 0.0j
    return complex(z1), complex(z2)


def psi_at_r(r: float, k: float, z1: complex, z2: complex) -> complex:
    z = k * r
    return z1 * special.hankel1(0, z) + z2 * special.hankel2(0, z)


def _radial_flux_at_rmax_mpmath(
    r_min: float,
    r_max: float,
    k: float,
    alpha: float,
    *,
    dps: int = 50,
) -> float:
    """
    ``r_max * Im(conj(psi) * dpsi/dr)`` for the Robin + outer-Hankel1 linear combination,
    using arbitrary-precision Hankel functions (``Z1,Z2`` are large and nearly cancel in
    float64, which destroys ``dpsi/dr`` in IEEE arithmetic).
    """
    if _mpmath is None:
        return float("nan")
    mp = _mpmath
    mp.mp.dps = int(dps)
    zi = mp.mpc(k * r_min)
    zo = mp.mpc(k * r_max)
    h1i, h2i = mp.hankel1(0, zi), mp.hankel2(0, zi)
    dh1i = -k * mp.hankel1(1, zi)
    dh2i = -k * mp.hankel2(1, zi)
    a1 = dh1i + alpha * h1i
    a2 = dh2i + alpha * h2i
    h1o, h2o = mp.hankel1(0, zo), mp.hankel2(0, zo)
    d1o = -k * mp.hankel1(1, zo)
    d2o = -k * mp.hankel2(1, zo)
    mat = mp.matrix([[h1o, h2o], [a1, a2]])
    rhs = mp.matrix([h1o, 0])
    sol = mp.lu_solve(mat, rhs)
    z1, z2 = sol[0], sol[1]
    psi_b = z1 * h1o + z2 * h2o
    dpsi_b = z1 * d1o + z2 * d2o
    return float(r_max * mp.im(mp.conj(psi_b) * dpsi_b))


def transmission_helmholtz_radial(
    *,
    d: float = D_FM,
    r_max_factor: float = R_MAX_FACTOR,
    n_theta: int = N_THETA,
    k: float | None = None,
    alpha: float | None = None,
) -> tuple[float, dict[str, object]]:
    verts = triangle_vertices(d)
    r_max = r_max_factor * d
    if k is None:
        k = 2.0 * math.pi / LAMBDA_SHOCK_M
    if alpha is None:
        alpha = c / (G_shear * LAMBDA_SHOCK_M)

    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    p_out = np.zeros(n_theta, dtype=np.float64)
    n_ok = 0
    meta_prec = "mpmath" if _mpmath is not None else "float64_unstable"
    for i, th in enumerate(thetas):
        rmin = r_min_along_ray(th, verts)
        if _mpmath is not None:
            p_out[i] = _radial_flux_at_rmax_mpmath(rmin, r_max, k, alpha, dps=50)
            if math.isfinite(p_out[i]):
                n_ok += 1
            continue
        z1, z2 = hankel_radial_Z(rmin, r_max, k, alpha)
        if not (
            math.isfinite(z1.real)
            and math.isfinite(z1.imag)
            and math.isfinite(z2.real)
            and math.isfinite(z2.imag)
        ):
            p_out[i] = float("nan")
            continue
        n_ok += 1
        psib = psi_at_r(r_max, k, z1, z2)
        h = max(1e-7 * r_max, 1e-24)
        dpsib = (psi_at_r(r_max + h, k, z1, z2) - psi_at_r(r_max - h, k, z1, z2)) / (2.0 * h)
        p_out[i] = r_max * float(np.imag(np.conjugate(psib) * dpsib))

    dtheta = 2.0 * math.pi / n_theta
    if hasattr(np, "trapezoid"):
        p_total = float(np.trapezoid(p_out, dx=dtheta))
    else:
        p_total = float(np.trapz(p_out, dx=dtheta))  # type: ignore[attr-defined]
    p_inc = 2.0 * math.pi * r_max * k
    t = p_total / max(p_inc, 1e-300)
    t = float(max(0.0, min(t, 2.0)))

    meta: dict[str, object] = {
        "method": "helmholtz_hankel_radial",
        "n_theta": n_theta,
        "n_theta_ok": n_ok,
        "k": k,
        "alpha": alpha,
        "r_max": r_max,
        "p_total": p_total,
        "p_inc": p_inc,
        "helmholtz_prec": meta_prec,
    }
    return float(t), meta


def _point_in_triangle(p: np.ndarray, verts: np.ndarray) -> bool:
    """Barycentric-style winding for CCW verts v0,v1,v2."""
    x, y = p[0], p[1]
    v0, v1, v2 = verts[0], verts[1], verts[2]

    def sign(pa: np.ndarray, pb: np.ndarray, pc: np.ndarray) -> float:
        return (pa[0] - pc[0]) * (pb[1] - pc[1]) - (pb[0] - pc[0]) * (pa[1] - pc[1])

    d1 = sign(np.array([x, y]), v0, v1)
    d2 = sign(np.array([x, y]), v1, v2)
    d3 = sign(np.array([x, y]), v2, v0)
    neg = (d1 < 0.0) or (d2 < 0.0) or (d3 < 0.0)
    pos = (d1 > 0.0) or (d2 > 0.0) or (d3 > 0.0)
    return not (neg and pos)


def _nudge_inside_from_vertex(
    vertex: np.ndarray,
    direction: np.ndarray,
    verts: np.ndarray,
    char_len: float,
) -> np.ndarray | None:
    """
    Smallest positive move ``vertex + eps * d_hat`` strictly inside the triangle.
    Directions that leave the wedge immediately return None.
    """
    d_hat = direction / max(np.linalg.norm(direction), 1e-30)
    eps0 = max(1e-18 * char_len, 1e-24)
    for k in range(1, 5000):
        eps = eps0 * float(k)
        p = vertex + eps * d_hat
        if _point_in_triangle(p, verts):
            return p.astype(np.float64)
        if eps > 2.0 * char_len:
            break
    return None


def _fresnel_power_transmission_oblique(z1: float, z2: float, cos_i: float, cos_t: float) -> float:
    """Acoustic intensity transmission (energy flux ratio) oblique interface."""
    ci = max(abs(cos_i), 1e-12)
    ct = max(abs(cos_t), 1e-12)
    num = 4.0 * z1 * z2 * ci * ct
    den = (z1 * ci + z2 * ct) ** 2
    return float(min(max(num / max(den, 1e-300), 0.0), 1.0))


def _normal_fresnel_power_R(z_in: float, z_out: float) -> tuple[float, float]:
    """Normal-incidence power reflection R and transmission T_fresnel = 1 - R (user impedance form)."""
    den = z_in + z_out
    if abs(den) < 1e-300:
        return 0.0, 1.0
    r_amp = (z_in - z_out) / den
    r_pow = float(r_amp**2)
    r_pow = min(max(r_pow, 0.0), 1.0)
    return r_pow, float(1.0 - r_pow)


# Segments along each chord so each sub-step length equals c_eff * dt with sum(c_eff*dt) = t_hit.
C_EFF_SUBSEGMENTS = 8


def _ray_exit_triangle(
    origin: np.ndarray,
    direction: np.ndarray,
    verts: np.ndarray,
    *,
    max_bounces: int = 80,
    z_in: float,
    z_out: float,
    c_eff: float | None = None,
    normal_fresnel_power: bool = False,
    energy_floor: float = 1e-2,
) -> float:
    """
    Trace ray inside triangle; accumulate transmitted power to exterior (first hit
    and subsequent escapes after re-entry not modeled — rays start inside only).
    """
    pos = origin.copy()
    d = direction / max(np.linalg.norm(direction), 1e-30)
    energy = 1.0
    transmitted = 0.0

    for _ in range(max_bounces):
        if energy < energy_floor:
            break
        # distances to all three edges (forward)
        hits: list[tuple[float, int, np.ndarray, np.ndarray]] = []
        for i in range(3):
            p0 = verts[i]
            p1 = verts[(i + 1) % 3]
            t = _intersect_ray_segment(pos, d, p0, p1)
            if t is None or t <= 0.0:
                continue
            hit = pos + t * d
            edge = p1 - p0
            edge_len = max(np.linalg.norm(edge), 1e-30)
            tangent = edge / edge_len
            # outward normal from triangle interior (CCW verts): rotate tangent +90
            n_out = np.array([tangent[1], -tangent[0]], dtype=np.float64)
            # ensure outward (away from centroid)
            centroid = verts.mean(axis=0)
            if np.dot(n_out, hit - centroid) < 0:
                n_out = -n_out
            cos_i = float(np.dot(d, n_out))
            if cos_i <= 1e-9:
                continue
            hits.append((t, i, hit, n_out))

        if not hits:
            break
        hits.sort(key=lambda h: h[0])
        t_hit, _i_edge, hit, n_out = hits[0]
        # Advance to the wall in sub-steps of length c_eff * dt with (sum dt) = t_hit / c_eff.
        if c_eff is not None and c_eff > 0.0 and t_hit > 0.0:
            n_seg = max(1, int(C_EFF_SUBSEGMENTS))
            dt_ray = t_hit / (n_seg * c_eff)
            cur = pos.copy()
            for _seg in range(n_seg):
                ds = c_eff * dt_ray
                cur = cur + ds * d
            pos = cur - 1e-10 * n_out
        else:
            pos = hit - 1e-10 * n_out
        cos_i = max(min(float(np.dot(d, n_out)), 1.0), 1e-9)
        if normal_fresnel_power:
            r_power, t_power = _normal_fresnel_power_R(z_in, z_out)
        else:
            # Snell for acoustic: cos_t from (sin_i / c1 = sin_t / c2) with c1=c2 here -> cos_t = cos_i
            cos_t = cos_i
            t_power = _fresnel_power_transmission_oblique(z_in, z_out, cos_i, cos_t)
            r_power = max(0.0, 1.0 - t_power)
        transmitted += energy * t_power
        energy *= r_power
        # reflect direction (outward normal): incoming toward +n has I·n>0; reflect to stay inside
        d = d - 2.0 * np.dot(d, n_out) * n_out
        d = d / max(np.linalg.norm(d), 1e-30)
        if not _point_in_triangle(pos, verts):
            break

    return float(transmitted)


def _vertex_interior_directions(
    vertex_idx: int,
    verts: np.ndarray,
    n_rays: int,
) -> list[np.ndarray]:
    """
    ``n_rays`` unit directions uniformly in the interior wedge at ``vertex_idx``.
    For an equilateral triangle the interior angle is ``pi/3``; bisector points toward
    the centroid. This avoids diluting ``T`` by directions that never enter the
    triangle from a boundary vertex.
    """
    v = verts[int(vertex_idx) % 3].astype(np.float64)
    centroid = verts.mean(axis=0)
    bis = centroid - v
    phi0 = math.atan2(float(bis[1]), float(bis[0]))
    # Equilateral: interior half-angle = pi/6 (30 deg)
    half = math.pi / 6.0
    out: list[np.ndarray] = []
    for j in range(n_rays):
        phi = phi0 - half + (2.0 * half) * (j + 0.5) / float(n_rays)
        out.append(np.array([math.cos(phi), math.sin(phi)], dtype=np.float64))
    return out


def transmission_geometric_optics(
    *,
    d: float = D_FM,
    n_rays: int = N_THETA_GO,
    z_in: float | None = None,
    z_out: float | None = None,
    z_out_over_in: float = 0.85,
    c_eff: float | None = None,
    normal_fresnel_power: bool = False,
    source_vertex_index: int | None = None,
    vertex_full_sphere: bool = False,
) -> tuple[float, dict[str, object]]:
    verts = triangle_vertices(d)
    if z_in is None:
        z_in = float(rho_V * c)
    if z_out is None:
        z_out = float(z_in * z_out_over_in)
    if source_vertex_index is None:
        origin0 = verts.mean(axis=0)
    else:
        iv = int(source_vertex_index) % 3
        origin0 = verts[iv].astype(np.float64)

    out_sum = 0.0
    n_valid = 0
    if source_vertex_index is not None and not vertex_full_sphere:
        dir_list = _vertex_interior_directions(int(source_vertex_index) % 3, verts, n_rays)
    else:
        dir_list = None

    for j in range(n_rays):
        if dir_list is not None:
            direction = dir_list[j]
        else:
            ang = 2.0 * math.pi * (j + 0.5) / n_rays
            direction = np.array([math.cos(ang), math.sin(ang)], dtype=np.float64)
        if source_vertex_index is None:
            origin = origin0.copy()
            out_sum += _ray_exit_triangle(
                origin,
                direction,
                verts,
                z_in=z_in,
                z_out=z_out,
                c_eff=c_eff,
                normal_fresnel_power=normal_fresnel_power,
            )
        else:
            p_in = _nudge_inside_from_vertex(origin0, direction, verts, char_len=d)
            if p_in is None:
                out_sum += 0.0
            else:
                n_valid += 1
                out_sum += _ray_exit_triangle(
                    p_in,
                    direction,
                    verts,
                    z_in=z_in,
                    z_out=z_out,
                    c_eff=c_eff,
                    normal_fresnel_power=normal_fresnel_power,
                )

    t = out_sum / max(n_rays, 1)
    meta = {
        "method": "geometric_optics",
        "n_rays": n_rays,
        "z_in": z_in,
        "z_out": z_out,
        "c_eff_ray": c_eff,
        "normal_fresnel_power": normal_fresnel_power,
        "source_vertex_index": source_vertex_index,
        "vertex_full_sphere": vertex_full_sphere,
        "vertex_direction_sampling": (
            "interior_wedge_720" if source_vertex_index is not None and not vertex_full_sphere else "full_2pi_720"
        ),
        "n_rays_valid_vertex_nudge": n_valid if source_vertex_index is not None else n_rays,
    }
    return float(t), meta


def weinberg_from_T(t: float) -> dict[str, float]:
    t = max(float(t), 1e-30)
    omega_leak = t * 4.0 * math.pi
    eta_guiding = (4.0 * math.pi) / max(omega_leak, 1e-300)
    k_eff_effective = K_EFF * ETA_TRANSMISSION * eta_guiding
    sin2 = GAMMA_W_EFF_OVER_OMEGA_E / (GAMMA_W_EFF_OVER_OMEGA_E + k_eff_effective)
    return {
        "T": float(t),
        "Omega_leak_sr": float(omega_leak),
        "eta_guiding": float(eta_guiding),
        "k_eff_effective": float(k_eff_effective),
        "sin2_thetaW": float(sin2),
    }


def run_triangle_lens(
    *,
    force_geometric: bool = False,
    n_theta_helm_diag: int = 36,
) -> dict[str, object]:
    verts = triangle_vertices(D_FM)
    r_max = R_MAX_FACTOR * D_FM
    k = 2.0 * math.pi / LAMBDA_SHOCK_M
    alpha = c / (G_shear * LAMBDA_SHOCK_M)

    # Interior: f_C^3 stiffening (confined medium vs exterior vacuum impedance)
    f_c_lens = 2.71828  # e
    g_eff = float(G_shear * (f_c_lens**3))
    c_eff = float(c * math.sqrt(f_c_lens**3))
    z_in_conf = float(rho_V * c_eff)
    z_out_conf = float(rho_V * c)
    r_fresnel, t_fresnel_normal = _normal_fresnel_power_R(z_in_conf, z_out_conf)

    t_go_conf_center, meta_go_conf_center = transmission_geometric_optics(
        d=D_FM,
        n_rays=N_THETA_GO,
        z_in=z_in_conf,
        z_out=z_out_conf,
        c_eff=c_eff,
        normal_fresnel_power=True,
        source_vertex_index=None,
    )
    t_go_conf_vertex, meta_go_conf_vertex = transmission_geometric_optics(
        d=D_FM,
        n_rays=N_THETA_GO,
        z_in=z_in_conf,
        z_out=z_out_conf,
        c_eff=c_eff,
        normal_fresnel_power=True,
        source_vertex_index=0,
    )
    t_go_geom, meta_go_geom = transmission_geometric_optics(d=D_FM, n_rays=N_THETA_GO)

    chain = weinberg_from_T(t_go_conf_vertex)
    geom_w = weinberg_from_T(t_go_geom)
    center_conf_w = weinberg_from_T(t_go_conf_center)

    chain["method_used"] = "f_c_confinement_go_vertex_primary"
    chain["go_meta"] = meta_go_conf_vertex
    chain["go_meta_f_c_center"] = meta_go_conf_center
    chain["go_meta_geom_only"] = meta_go_geom
    chain["ray_source_primary"] = "vertex_index_0"
    chain["ray_source_f_c_center"] = "triangle_centroid"
    chain["ray_source_geom_only"] = "triangle_centroid"
    chain["vertex_direction_sampling"] = str(meta_go_conf_vertex.get("vertex_direction_sampling", ""))
    chain["T"] = float(t_go_conf_vertex)
    chain["T_geometric_optics"] = float(t_go_conf_vertex)
    chain["T_f_c_center"] = float(t_go_conf_center)
    chain["T_geom_only"] = float(t_go_geom)
    chain["eta_guiding_f_c_center"] = float(center_conf_w["eta_guiding"])
    chain["sin2_thetaW_f_c_center"] = float(center_conf_w["sin2_thetaW"])
    chain["Omega_leak_f_c_center_sr"] = float(center_conf_w["Omega_leak_sr"])
    chain["f_C_lens"] = f_c_lens
    chain["G_eff"] = g_eff
    chain["c_eff"] = c_eff
    chain["G_eff_over_G_shear"] = float(g_eff / max(G_shear, 1e-300))
    chain["c_eff_over_c"] = float(c_eff / max(c, 1e-300))
    chain["Z_in_confinement"] = z_in_conf
    chain["Z_out_vacuum"] = z_out_conf
    chain["fresnel_R_power"] = float(r_fresnel)
    chain["fresnel_T_power"] = float(t_fresnel_normal)
    chain["eta_guiding_geom_only"] = float(geom_w["eta_guiding"])
    chain["sin2_thetaW_geom_only"] = float(geom_w["sin2_thetaW"])
    chain["Omega_leak_geom_only_sr"] = float(geom_w["Omega_leak_sr"])

    if not force_geometric:
        t_h, meta_h = transmission_helmholtz_radial(
            d=D_FM,
            r_max_factor=R_MAX_FACTOR,
            n_theta=max(8, int(n_theta_helm_diag)),
            k=k,
            alpha=alpha,
        )
        chain["helmholtz_meta"] = meta_h
        chain["T_helmholtz_radial_flux_Ttrap"] = float(t_h)
        chain["method_used"] = "f_c_vertex_go_primary_helmholtz_diagnostic"
    else:
        chain["method_used"] = "f_c_vertex_go_forced"

    dev_abs = abs(chain["sin2_thetaW"] - SIN2_THETA_W_EXP)
    rel_pct = dev_abs / max(SIN2_THETA_W_EXP, 1e-30) * 100.0
    closure = dev_abs < 0.05 * SIN2_THETA_W_EXP
    chain["sin2_thetaW_exp"] = SIN2_THETA_W_EXP
    chain["abs_deviation_sin2"] = float(dev_abs)
    chain["rel_deviation_pct"] = float(rel_pct)
    chain["closure_sin2_thetaW"] = bool(closure)
    chain["triangle_side_m"] = D_FM
    chain["r_max_m"] = r_max
    chain["k_inv_m"] = k
    chain["alpha_robin"] = alpha
    chain["vertices_m"] = verts
    return chain


def main() -> None:
    r = run_triangle_lens()
    print("=== triangle_impedance_lens ===")
    print(f"  method: {r['method_used']}")
    print(f"  ray source (primary f_C GO): {r['ray_source_primary']}")
    print(f"  ray source (f_C GO center comparison): {r['ray_source_f_c_center']}")
    print(f"  ray source (geom-only): {r['ray_source_geom_only']}")
    print(f"  vertex direction sampling (primary): {r.get('vertex_direction_sampling', '')}")
    print(f"  f_C (lens) = {r['f_C_lens']:.8f}")
    print(f"  G_eff / G_shear = {r['G_eff_over_G_shear']:.8f}")
    print(f"  c_eff / c = {r['c_eff_over_c']:.8f}")
    print(f"  Fresnel R (normal, power) = {r['fresnel_R_power']:.8f}  T_fresnel = 1 - R = {r['fresnel_T_power']:.8f}")
    print(f"  Z_in (confined) = {r['Z_in_confinement']:.8e}  Z_out (rho_V*c) = {r['Z_out_vacuum']:.8e}")
    print("  --- comparison: T, eta_guiding, sin2_thetaW ---")
    print(
        f"    geom-only (centroid):     T={r['T_geom_only']:.8e}  "
        f"eta={r['eta_guiding_geom_only']:.8e}  sin2={r['sin2_thetaW_geom_only']:.8f}"
    )
    print(
        f"    f_C + centroid:           T={r['T_f_c_center']:.8e}  "
        f"eta={r['eta_guiding_f_c_center']:.8e}  sin2={r['sin2_thetaW_f_c_center']:.8f}"
    )
    print(
        f"    f_C + vertex v0 (primary): T={r['T']:.8e}  "
        f"eta={r['eta_guiding']:.8e}  sin2={r['sin2_thetaW']:.8f}  exp={SIN2_THETA_W_EXP:.8f}"
    )
    print(
        f"  primary |sin2 - exp| = {r['abs_deviation_sin2']:.8e}  "
        f"({r['rel_deviation_pct']:.4f}% of exp)  (vertex v0 source)"
    )
    if r["closure_sin2_thetaW"]:
        print("  [closure:sin2_thetaW]")
    if "T_helmholtz_radial_flux_Ttrap" in r:
        print(
            "  T_helmholtz_radial_flux (diagnostic, radial Poynting / P_inc) =",
            f"{r['T_helmholtz_radial_flux_Ttrap']:.8e}",
        )
    if "helmholtz_meta" in r:
        hm = r["helmholtz_meta"]
        print(
            f"  Helmholtz diagnostic: angles ok {hm['n_theta_ok']}/{hm['n_theta']}, "
            f"P_total={hm['p_total']:.6e}, P_inc={hm['p_inc']:.6e}, prec={hm.get('helmholtz_prec', '')}"
        )


if __name__ == "__main__":
    main()
