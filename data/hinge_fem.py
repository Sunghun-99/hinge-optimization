"""2D elliptical-flexure hinge starter. Geometry/meshing: Gmsh; stiffness: warp.fem.

IMPORTANT: numerical output is not validated until you run the verification commands.
Units: mm, MPa, mm displacement. 2D plane stress, unit out-of-plane thickness.
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import spsolve

try:
    import gmsh
    import warp as wp
    import warp.fem as fem
except ImportError as exc:
    raise SystemExit(f"Missing dependency: {exc}. Install packages in README.md")

# TEST VALUES ONLY: replace E, nu, delta, L with team-agreed physical inputs.
E_DEFAULT = 70000.0           # MPa (illustrative, NOT specified in proposal)
NU_DEFAULT = 0.33             # illustrative, NOT specified in proposal
DELTA_DEFAULT = 0.005        # mm (illustrative, NOT specified in proposal)
L_DEFAULT = 12.0             # mm (illustrative, NOT specified in proposal)


@fem.integrand
def plane_stress_stiffness(s: fem.Sample, u: fem.Field, v: fem.Field,
                           lam: float, mu: float):
    """∫ [ 2μ ε(u):ε(v) + λ_plane_stress tr(εu) tr(εv)] dΩ."""
    gu = fem.grad(u, s)
    gv = fem.grad(v, s)
    uxx = gu[0, 0]
    uyy = gu[1, 1]
    uxy = 0.5 * (gu[0, 1] + gu[1, 0])
    vxx = gv[0, 0]
    vyy = gv[1, 1]
    vxy = 0.5 * (gv[0, 1] + gv[1, 0])
    return 2.0 * mu * (uxx * vxx + uyy * vyy + 2.0 * uxy * vxy) + lam * (uxx + uyy) * (vxx + vyy)


def make_mesh(a: float = 3.0, b: float = 1.5, t: float = 0.7,
              L: float = L_DEFAULT, mesh_factor: float = 1.,
              beam_height: float | None = None):
    """Symmetric top+bottom elliptical notches. For hinge: outer H = t+2b.
    Two parametric half-ellipses (not a central elliptical through-hole).
    """
    if mesh_factor <= 0.0 or L <= 0.0:
        raise ValueError('mesh_factor and L must be positive')
    if beam_height is None and not (a > 0.0 and b > 0.0 and t > 0.0 and 2 * a < L):
        raise ValueError('Require 0 < 2a < L and b,t > 0')
    if beam_height is not None and beam_height <= 0:
        raise ValueError('Beam height must be positive')

    gmsh.clear()
    gmsh.model.add('cantilever_beam' if beam_height is not None else 'elliptical_flexure_hinge')
    geo = gmsh.model.geo

    def pt(x, y):
        return geo.addPoint(float(x), float(y), 0.0)

    if beam_height is not None:
        H = beam_height
        A, B, C, D = pt(0, -H/2), pt(L, -H/2), pt(L, H/2), pt(0, H/2)
        curves = [geo.addLine(A, B), geo.addLine(B, C), geo.addLine(C, D), geo.addLine(D, A)]
        notch_curves = []
        hmin = min(H / 5.0, 0.16) * mesh_factor
        hmax = 0.28 * mesh_factor
    else:
        H = t + 2.0*b
        cx = L / 2.0
        A = pt(0., -H/2.)
        BL = pt(cx-a, -H/2.)
        # Bottom notch follows upper half-ellipse left -> right
        bottom_interior = [pt(cx + a*np.cos(theta), -H/2. + b*np.sin(theta))
                           for theta in np.linspace(np.pi, 0., 31)[1:-1]]
        BR = pt(cx+a, -H/2.)
        C = pt(L, -H/2.)
        D = pt(L, H/2.)
        TR = pt(cx+a, H/2.)
        # Top notch follows lower half-ellipse right -> left
        top_interior = [pt(cx + a*np.cos(theta), H/2. - b*np.sin(theta))
                        for theta in np.linspace(0., np.pi, 31)[1:-1]]
        TL = pt(cx-a, H/2.)
        G = pt(0., H/2.)
        curve_bottom = geo.addSpline([BL, *bottom_interior, BR])
        curve_top = geo.addSpline([TR, *top_interior, TL])
        curves = [geo.addLine(A, BL), curve_bottom, geo.addLine(BR, C),
                  geo.addLine(C, D), geo.addLine(D, TR), curve_top,
                  geo.addLine(TL, G), geo.addLine(G, A)]
        notch_curves = [curve_bottom, curve_top]
        hmin = min(0.045, t/10.0) * mesh_factor
        hmax = 0.28 * mesh_factor

    loop = geo.addCurveLoop(curves)
    geo.addPlaneSurface([loop])
    geo.synchronize()

    gmsh.option.setNumber('Mesh.Algorithm', 6)
    gmsh.option.setNumber('Mesh.ElementOrder', 1)
    gmsh.option.setNumber('Mesh.MeshSizeFromPoints', 0)
    gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', 0)
    gmsh.option.setNumber('Mesh.MeshSizeExtendFromBoundary', 0)
    gmsh.option.setNumber('Mesh.MeshSizeMin', hmin)
    gmsh.option.setNumber('Mesh.MeshSizeMax', hmax)
    if notch_curves:
        distance = gmsh.model.mesh.field.add('Distance')
        gmsh.model.mesh.field.setNumbers(distance, 'CurvesList', notch_curves)
        gmsh.model.mesh.field.setNumber(distance, 'Sampling', 120)
        threshold = gmsh.model.mesh.field.add('Threshold')
        gmsh.model.mesh.field.setNumber(threshold, 'InField', distance)
        gmsh.model.mesh.field.setNumber(threshold, 'SizeMin', hmin)
        gmsh.model.mesh.field.setNumber(threshold, 'SizeMax', hmax)
        gmsh.model.mesh.field.setNumber(threshold, 'DistMin', 0.10)
        gmsh.model.mesh.field.setNumber(threshold, 'DistMax', 1.0)
        gmsh.model.mesh.field.setAsBackgroundMesh(threshold)
    gmsh.model.mesh.generate(2)

    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    pos = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[:, :2]
    node_index = {int(tag): i for i, tag in enumerate(tags)}
    tri_type = gmsh.model.mesh.getElementType('Triangle', 1)
    _, conn = gmsh.model.mesh.getElementsByType(tri_type)
    tris = np.asarray([node_index[int(tag)] for tag in conn], dtype=np.int32).reshape(-1, 3)
    if len(pos) < 3 or len(tris) < 1:
        raise RuntimeError('No triangles produced by Gmsh')

    # Gmsh's global getNodes() can include standalone 0D nodes at spline
    # interpolation/control points that are NOT used by any 2D triangle.
    # Such unused vertices create zero-stiffness DOFs and a singular K matrix.
    used = np.unique(tris)
    unused_count = len(pos) - len(used)
    print(f'[mesh] total Gmsh nodes={len(pos)}, triangle nodes={len(used)}, '
          f'unused nodes removed={unused_count}, triangles={len(tris)}')
    if unused_count:
        old_to_new = np.full(len(pos), -1, dtype=np.int32)
        old_to_new[used] = np.arange(len(used), dtype=np.int32)
        tris = old_to_new[tris]
        pos = pos[used]
    return pos, tris


def assemble_warp_stiffness(pos, tris, young, nu, device):
    """FEM stiffness integrals happen in warp.fem, not in SciPy/Gmsh."""
    lam = young * nu / (1.0 - nu**2)  # PLANE STRESS, not plane strain
    mu = young / (2.0 * (1.0 + nu))
    with wp.ScopedDevice(device):
        coords_wp = wp.array(pos.astype(np.float32), dtype=wp.vec2)
        tri_wp = wp.array(np.ascontiguousarray(tris, dtype=np.int32), dtype=wp.int32)
        geo = fem.Trimesh2D(tri_vertex_indices=tri_wp, positions=coords_wp)
        space = fem.make_polynomial_space(geo, degree=1, dtype=wp.vec2)
        cells = fem.Cells(geometry=geo)
        trial = fem.make_trial(space=space, domain=cells)
        test = fem.make_test(space=space, domain=cells)
        K_warp = fem.integrate(plane_stress_stiffness, fields={'u': trial, 'v': test},
                               values={'lam': float(lam), 'mu': float(mu)}, output_dtype=float)
        # Export a Warp-assembled block CSR matrix to SciPy for robust constrained solving.
        rowptr = np.asarray(K_warp.offsets.numpy(), dtype=np.int64)[:K_warp.nrow+1]
        nnz = int(rowptr[-1])
        cols = np.asarray(K_warp.columns.numpy(), dtype=np.int32)[:nnz]
        blocks = np.asarray(K_warp.values.numpy(), dtype=np.float64)[:nnz]
        if K_warp.block_shape != (2, 2):
            raise RuntimeError(f'Expected 2x2 displacement blocks, found {K_warp.block_shape}')
        K = sparse.bsr_matrix((blocks, cols, rowptr), shape=(2*K_warp.nrow, 2*K_warp.ncol)).tocsr()
    if K.shape[0] != 2*len(pos):
        raise RuntimeError('FEM degree-of-freedom ordering not vertex P1 as expected')
    diag = np.asarray(K.diagonal())
    bad = np.flatnonzero(~np.isfinite(diag) | (diag <= 0))
    if len(bad):
        raise RuntimeError(f'Assembled stiffness has {len(bad)} non-positive/invalid '
                           f'diagonal DOFs; first indices={bad[:12].tolist()}')
    return K


def solve_with_displacement_bc(K, pos, L, delta):
    """Left ux=uy=0, right uy=delta, right ux free; all else natural traction-free."""
    left = np.flatnonzero(np.abs(pos[:, 0]) < 1e-5)
    right = np.flatnonzero(np.abs(pos[:, 0] - L) < 1e-5)
    print(f'[BC] left fixed nodes={len(left)}, right prescribed-uy nodes={len(right)}')
    if len(left) < 2 or len(right) < 2:
        raise RuntimeError('Could not find fixed/right nodes on the geometric end faces')
    fixed = np.concatenate([2*left, 2*left+1, 2*right+1])
    fixed = np.unique(fixed)
    imposed = np.zeros(K.shape[0], dtype=np.float64)
    imposed[2*right + 1] = delta
    mask = np.ones(K.shape[0], dtype=bool)
    mask[fixed] = False
    free = np.flatnonzero(mask)
    rhs = -K[free, :][:, fixed] @ imposed[fixed]
    u = imposed.copy()
    u[free] = spsolve(K[free, :][:, free], rhs)
    if not np.all(np.isfinite(u)):
        raise RuntimeError('Non-finite FEM displacement solution')
    reactions = np.asarray(K @ u)
    free_resid = np.max(np.abs(reactions[free]))
    reaction_right_y = reactions[2*right + 1].sum()  # force per unit out-of-plane width
    return u.reshape(-1, 2), float(reaction_right_y), float(free_resid)


def element_von_mises(pos, tris, nodal_u, E, nu):
    """Element-constant P1 stress evaluated from strain. Plane stress [MPa]."""
    p = pos[tris]
    d = nodal_u[tris]
    x0, x1, x2 = p[:, 0, 0], p[:, 1, 0], p[:, 2, 0]
    y0, y1, y2 = p[:, 0, 1], p[:, 1, 1], p[:, 2, 1]
    twice_area = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)
    if np.any(np.abs(twice_area) < 1e-12):
        raise RuntimeError('Degenerate triangle')
    b = np.column_stack([y1-y2, y2-y0, y0-y1]) / twice_area[:, None]
    c = np.column_stack([x2-x1, x0-x2, x1-x0]) / twice_area[:, None]
    exx = (d[:, :, 0] * b).sum(axis=1)
    eyy = (d[:, :, 1] * c).sum(axis=1)
    exy = 0.5*((d[:, :, 0] * c).sum(axis=1) + (d[:, :, 1] * b).sum(axis=1))
    lam = E * nu / (1.0 - nu**2)
    mu = E / (2*(1 + nu))
    sx = lam*(exx+eyy) + 2*mu*exx
    sy = lam*(exx+eyy) + 2*mu*eyy
    txy = 2*mu*exy
    vm = np.sqrt(np.maximum(0., sx*sx - sx*sy + sy*sy + 3*txy*txy))
    return vm


def run_one(a, b, t, *, L, E, nu, delta, mesh_factor, device, beam_height=None,
            output_png=None, mesh_png=None):
    start = time.perf_counter()
    pos, tris = make_mesh(a, b, t, L=L, mesh_factor=mesh_factor, beam_height=beam_height)
    mesh_seconds = time.perf_counter() - start
    K = assemble_warp_stiffness(pos, tris, E, nu, device)
    assembly_seconds = time.perf_counter() - start - mesh_seconds
    u, reaction, residual = solve_with_displacement_bc(K, pos, L, delta)
    stress = element_von_mises(pos, tris, u, E, nu)
    solve_and_post_seconds = time.perf_counter() - start - mesh_seconds - assembly_seconds
    if output_png or mesh_png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        tri_view = mtri.Triangulation(pos[:, 0], pos[:, 1], tris)
        if output_png:
            fig, ax = plt.subplots(figsize=(10, 4))
            im = ax.tripcolor(tri_view, facecolors=stress, shading='flat', cmap='viridis')
            fig.colorbar(im, ax=ax, label='von Mises stress [MPa]')
            ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            ax.set_title('Warp FEM | plane stress | element P1')
            fig.tight_layout(); fig.savefig(output_png, dpi=180); plt.close(fig)
        if mesh_png:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.triplot(tri_view, color='#444444', linewidth=0.20)
            ax.set_aspect('equal'); ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
            ax.set_title('Gmsh triangular mesh')
            fig.tight_layout(); fig.savefig(mesh_png, dpi=180); plt.close(fig)
    return {
        'a_mm': a, 'b_mm': b, 't_mm': t,
        'sigma_max_MPa': float(np.max(stress)),
        'sigma_p99_MPa': float(np.percentile(stress, 99)),
        'node_count': int(len(pos)), 'triangle_count': int(len(tris)),
        'tip_reaction_per_width_N_per_mm': reaction,
        'free_dof_residual_N_per_mm': residual,
        'mesh_seconds': round(mesh_seconds, 4),
        'assembly_seconds': round(assembly_seconds, 4),
        'solve_post_seconds': round(solve_and_post_seconds, 4),
        'total_seconds': round(time.perf_counter() - start, 4),
        'E_MPa': E, 'nu': nu, 'delta_mm': delta, 'L_mm': L, 'mesh_factor': mesh_factor,
        'device': device, 'warp_version': wp.__version__,
        'gmsh_version': getattr(gmsh, '__version__', 'unknown'),
        'python_version': platform.python_version(),
        'scipy_version': scipy.__version__
    }


def grid_cases():
    for i, j, k in itertools.product(range(5), range(11), range(6)):
        yield round(2.0 + 0.5*i, 3), round(1.0 + 0.1*j, 3), round(0.5 + 0.1*k, 3)


def append_row(path: Path, row, keys):
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser(description='Warp FEM elliptical hinge simulator')
    p.add_argument('--a', type=float, default=3.0)
    p.add_argument('--b', type=float, default=1.5)
    p.add_argument('--t', type=float, default=0.7)
    p.add_argument('--L', type=float, default=L_DEFAULT)
    p.add_argument('--E', type=float, default=E_DEFAULT)
    p.add_argument('--nu', type=float, default=NU_DEFAULT)
    p.add_argument('--delta', type=float, default=DELTA_DEFAULT)
    p.add_argument('--mesh-factor', type=float, default=1.0)
    p.add_argument('--device', choices=['cpu','cuda:0'], default='cpu')
    p.add_argument('--out', type=Path, default=Path('outputs'))
    p.add_argument('--batch', action='store_true', help='Run all 330 designs')
    p.add_argument('--limit', type=int, default=None, help='Only first N designs (smoke test)')
    p.add_argument('--verify-beam', action='store_true', help='No cutout; compare end force to EB beam theory')
    p.add_argument('--refine', action='store_true', help='Run mesh convergence at a,b,t')
    p.add_argument('--plots', action='store_true', help='Save PNG for single run')
    args = p.parse_args()
    if args.E <= 0 or not -0.99 < args.nu < 0.49:
        p.error('Check E and nu (0 < E, -0.99 < nu < 0.49)')
    args.out.mkdir(exist_ok=True, parents=True)
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    wp.init()
    try:
        if args.verify_beam:
            # Euler-Bernoulli limit: no hole, L/H=20, prescribed tip displacement
            length, height, disp = 20.0, 1.0, 0.001
            r = run_one(args.a, args.b, args.t, L=length, E=args.E, nu=args.nu,
                        delta=disp, mesh_factor=args.mesh_factor, device=args.device,
                        beam_height=height,
                        output_png=args.out/'beam_stress.png' if args.plots else None)
            I_unit_width = height**3 / 12.0
            force_EB = 3.0*args.E*I_unit_width*disp/length**3
            r['EB_reaction_per_width_N_per_mm'] = force_EB
            r['reaction_relative_error'] = abs(abs(r['tip_reaction_per_width_N_per_mm']) - force_EB)/force_EB
            (args.out/'beam_verification.json').write_text(json.dumps(r, indent=2), encoding='utf-8')
            print('Beam theory reaction (per width):', force_EB)
            print('Warp FEM reaction (per width):', r['tip_reaction_per_width_N_per_mm'])
            print('Relative error:', r['reaction_relative_error'])
            return

        if args.refine:
            rows=[]
            for factor in (1.6, 1.0, 0.65):
                r=run_one(args.a,args.b,args.t,L=args.L,E=args.E,nu=args.nu,
                          delta=args.delta,mesh_factor=factor,device=args.device)
                rows.append(r)
                print(f"mesh factor={factor}: triangles={r['triangle_count']}, smax={r['sigma_max_MPa']:.4g}")
            for i in range(1,len(rows)):
                prev=rows[i-1]['sigma_max_MPa']; now=rows[i]['sigma_max_MPa']
                rows[i]['change_from_previous_pct'] = 100*abs(now-prev)/max(abs(now),1e-12)
            path=args.out/'convergence.csv'
            with path.open('w',newline='',encoding='utf-8-sig') as fh:
                w=csv.DictWriter(fh,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
                w.writeheader(); w.writerows(rows)
            print('Saved',path)
            print('NOTE: Ellipse-to-flat intersections may create stress singularities; failure to converge is important.')
            return

        if args.batch:
            path = args.out/'hinge_330_results.csv'
            completed = set()
            if path.exists():
                with path.open('r',encoding='utf-8-sig',newline='') as fh:
                    for r in csv.DictReader(fh):
                        if r.get('status')=='ok':
                            completed.add((float(r['a_mm']),float(r['b_mm']),float(r['t_mm'])))
            keys = ['a_mm','b_mm','t_mm','status','sigma_max_MPa','sigma_p99_MPa',
                    'node_count','triangle_count','tip_reaction_per_width_N_per_mm',
                    'free_dof_residual_N_per_mm','mesh_seconds','assembly_seconds',
                    'solve_post_seconds','total_seconds','E_MPa','nu','delta_mm',
                    'L_mm','mesh_factor','device','warp_version','gmsh_version',
                    'python_version','scipy_version','error']
            cases=list(grid_cases())
            if args.limit is not None:
                cases=cases[:args.limit]
            print(f'Starting {len(cases)} designs (already completed: {len(completed)})')
            for i, (a,b,t) in enumerate(cases,1):
                if (a,b,t) in completed:
                    print(f'[{i}/{len(cases)}] skipping existing ({a}, {b}, {t})')
                    continue
                try:
                    row=run_one(a,b,t,L=args.L,E=args.E,nu=args.nu,
                                delta=args.delta,mesh_factor=args.mesh_factor,
                                device=args.device)
                    row['status']='ok'
                    print(f"[{i}/{len(cases)}] {a} {b} {t} -> {row['sigma_max_MPa']:.3f} MPa | {row['total_seconds']} s")
                except Exception as exc:
                    row={'a_mm':a,'b_mm':b,'t_mm':t,'status':'error','error':str(exc)}
                    print(f'[{i}/{len(cases)}] FAILED {a} {b} {t}: {exc}',file=sys.stderr)
                append_row(path,row,keys)
            print('Saved:', path)
            return

        r = run_one(args.a,args.b,args.t,L=args.L,E=args.E,nu=args.nu,
                    delta=args.delta,mesh_factor=args.mesh_factor,device=args.device,
                    output_png=args.out/'hinge_stress.png' if args.plots else None,
                    mesh_png=args.out/'hinge_mesh.png' if args.plots else None)
        (args.out/'single_result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
        print(json.dumps(r,indent=2))
    finally:
        gmsh.finalize()


if __name__ == '__main__':
    main()
