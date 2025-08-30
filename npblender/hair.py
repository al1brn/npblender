import numpy as np

EPS = 1e-8

import numpy as np

# ====================================================================================================
# Rigid curve
# ====================================================================================================

def rigid_step(S, N, splines, inv_mass, rest_len, max_theta, acc, dt):

    pos   = splines.position.reshape(S, N, 3)
    speed = splines.speed.reshape(S, N, 3)

    dspeed = acc*(inv_mass[None, :, None]*dt)

    new_speed = speed + dspeed
    new_pos = pos + (speed + new_speed)*(dt/2)

    min_cos = np.cos(max_theta)
    min_sin = np.sin(max_theta)

    direction = np.resize(np.array((0., 0., 1.)), (S, 3))
    for i in range(1, N):
        p = new_pos[:, i]
        p_prec = new_pos[:, i-1]
        segm = p - p_prec
        segm_len = np.linalg.norm(segm, keepdims=True, axis=-1)

        segm_dir = segm / segm_len
        segm_cos = np.einsum('ij,ij->i', segm_dir, direction)

        # ------ Too much bended
        sel = segm_cos < min_cos
        if np.any(sel):
            dirs = direction[sel]
            perp = np.cross(dirs, segm_dir[sel])
            v_i = np.cross(perp, dirs)
            print("SHAPE", dirs.shape, sel.shape, perp.shape, direction[sel].shape)
            segm_dir[sel] = dirs*min_cos + perp*min_sin

        # ----- Adjust length
        segm = segm_dir*rest_len[i-1]
        new_pos[:, i] = p_prec + segm

        direction = segm_dir

    splines.position = new_pos.reshape(-1, 3)
    splines.speed = ((new_pos - pos)/dt).reshape(-1, 3)

def test_rigid():

    from . engine import engine
    from .curve import Curve

    rng = np.random.default_rng(0)

    S, N = 10, 10

    start = rng.uniform(-5, 5, (S, 3))
    start[:, 2] = 0
    curve = Curve.line(start=start, end=start + (0, 0, 3), resolution=N)

    curve.points.position += rng.normal(0, .01, curve.points.position.shape)
    curve.points.new_vector("speed")

    curve.to_object("Splines BEFORE")

    # ----- Simulation

    L = 3.0 / (N - 1)
    rest_len = np.full((N-1,), L)

    inv_mass = np.ones((N,))
    inv_mass[0] = 0

    acc = np.array((0, 0, -10))
    for _ in range(10000):
        rigid_step(S, N, curve.points,
                inv_mass = inv_mass, 
                rest_len = rest_len, 
                max_theta = np.deg2rad(20),
                acc = acc,
                dt=1/60,
        )

    curve.to_object("Splines AFTER")







        













    




















# ====================================================================================================
# Gauss–Seidel distance-constraint projection
# ====================================================================================================

def project_distance_constraints_gs(
    pos,
    inv_mass,
    rest_len,
    *,
    stiffness=1.0,
    passes=4,
    both_directions=True,
    anchors=None,
    eps=1e-8,
):
    """
    Gauss–Seidel distance-constraint projection for S independent open chains.

    Inputs
    ------
    pos        : (S, N, D) float
                 Current point positions for S splines with N points in D dims.
    inv_mass   : (S, N, 1) float
                 Inverse masses per point. Use 0 for fixed (anchored) points,
                 1 for fully free points (or any nonnegative values).
    rest_len   : (S, N-1, 1) float
                 Target length for each edge (i, i+1).

    Parameters
    ----------
    stiffness       : [0..1], fraction of the edge correction applied per visit.
                      1.0 snaps the edge to its rest length in one visit when one end is fixed.
    passes          : number of Gauss–Seidel sweeps.
    both_directions : if True, do a backward sweep after the forward sweep (helps convergence).
    anchors         : None or (S, D). If given, re-impose pos[:,0] = anchors at each sweep.
    eps             : small epsilon to avoid division by zero.

    What it does (precise)
    ----------------------
    For each sweep, it visits edges (i, i+1) in order (and optionally in reverse order),
    and adjusts the two endpoints so that the edge length ||p_{i+1} - p_i|| moves toward its
    rest length r_i. The update is *physically correct* PBD:

        delta = p_{i+1} - p_i                         # (S, D)
        dist  = ||delta||                             # (S, 1)
        dir   = delta / max(dist, eps)                # (S, D)
        C     = dist - r_i                            # (S, 1)   (positive if too long)
        corr  = stiffness * C * dir                   # (S, D)

        p_i   += (w_i   / (w_i + w_{i+1})) * corr
        p_{i+1} -= (w_{i+1} / (w_i + w_{i+1})) * corr

    Rationale for the signs:
      - If the edge is **too long** (dist > r_i), then C > 0 and `corr` points from i→i+1.
        We move p_i *toward* p_{i+1} (+corr) and p_{i+1} *toward* p_i (−corr): the edge shrinks.
      - If it is **too short** (dist < r_i), C < 0 and the same formula stretches it.

    With 0 ≤ stiffness ≤ 1 this monotonically reduces |dist − r_i| on each visit (for w_i+w_{i+1}>0).

    Returns
    -------
    pos : (S, N, D) float
          The corrected positions after the specified number of sweeps.
    """
    pos = np.asarray(pos, float).copy()
    inv_mass = np.asarray(inv_mass, float)
    rest_len = np.asarray(rest_len, float)

    S, N, D = pos.shape
    inv_mass = np.broadcast_to(inv_mass, (S, N, 1))
    rest_len = np.broadcast_to(rest_len, (S, N-1, 1))
    if anchors is not None:
        anchors = np.asarray(anchors, float).reshape(S, 1, D)

    def relax_edge(i: int):
        pi = pos[:, i,   :]          # (S, D)
        pj = pos[:, i+1, :]          # (S, D)

        wL = inv_mass[:, i,   :1]    # (S, 1)
        wR = inv_mass[:, i+1, :1]    # (S, 1)
        wsum = wL + wR + eps

        delta = pj - pi
        dist  = np.sqrt((delta * delta).sum(-1, keepdims=True) + eps)
        dirn  = delta / dist
        rl    = rest_len[:, i, :1]

        C     = dist - rl                    # >0 if too long, <0 if too short
        corr  = stiffness * C * dirn         # (S, D)

        pi += (wL / wsum) * corr
        pj -= (wR / wsum) * corr

    for _ in range(int(passes)):

        # forward
        for i in range(N-1):
            relax_edge(i)
        # backward
        if both_directions:
            for i in range(N-2, -1, -1):
                relax_edge(i)
        # optional hard re-anchoring (prevents slow drift if you want exact roots)
        if anchors is not None:
            pos[:, 0:1, :] = anchors

    return pos

# ====================================================================================================
# Gauss–Seidel bending projection using the skip-edge distance (i ↔ i+2)
# ====================================================================================================

def project_bending_constraints_gs(
    pos,
    inv_mass,
    rest_bend,
    *,
    stiffness=0.2,
    passes=2,
    both_directions=True,
    anchors=None,
    eps=1e-8,
):
    """
    Gauss–Seidel bending projection using the skip-edge distance (i ↔ i+2).

    Purpose
    -------
    Enforce a soft bending constraint along open chains by driving the distance
    between non-adjacent points p_i and p_{i+2} toward a given rest length.
    This is the classic lightweight PBD "distance bending" (not a full angle
    hinge), which is cheap and vectorizable for large batches of splines.

    Inputs
    ------
    pos       : (S, N, D) float
        Current positions for S splines, each with N points in D dimensions.
    inv_mass  : (S, N, 1) float
        Inverse masses per point; 0 for fixed points (anchors), >0 for movable.
        Values are only read (no modification).
    rest_bend : (S, N-2, 1) float
        Target distances between points (i, i+2). For a straight chain with
        uniform edge length `rest`, a good default is `rest_bend = 2*rest`.

    Parameters
    ----------
    stiffness : float in [0, 1]
        Fraction of the bending correction applied at each visit. Typical
        values are 0.05–0.4. Larger values converge faster but can fight
        against edge-length constraints if applied too aggressively.
    passes : int
        Number of Gauss–Seidel sweeps. Each sweep visits all skip-edges in
        order, and optionally in reverse if `both_directions=True`.
    both_directions : bool
        If True, perform a backward sweep after the forward sweep for improved
        convergence along the chain.
    anchors : None or (S, D)
        Optional fixed root positions. If provided, re-impose pos[:, 0] at the
        end of each sweep (use inv_mass[:, 0] = 0 as well for the root).
    eps : float
        Small epsilon to avoid division by zero.

    Update rule (per skip-edge i → i+2)
    -----------------------------------
        delta = p_{i+2} - p_i                      # (S, D)
        dist  = ||delta||                          # (S, 1)
        dir   = delta / max(dist, eps)             # (S, D)
        C     = dist - rest_bend_i                 # (S, 1)  (>0: too long, <0: too short)
        corr  = stiffness * C * dir                # (S, D)

        p_i   += (w_i   / (w_i + w_{i+2})) * corr
        p_{i+2} -= (w_{i+2} / (w_i + w_{i+2})) * corr

    Signs rationale:
      - If dist > rest_bend, C > 0 and corr points from i→i+2.
        Moving p_i by +corr and p_{i+2} by −corr reduces the skip distance.
      - If dist < rest_bend, the same formula increases the distance.

    Notes
    -----
    - This bending step slightly perturbs adjacent edge lengths; in practice
      it is alternated with edge-length projections (distance constraints).
    - Because only i and i+2 are moved, the method is stable and fast while
      still producing smooth, resistant strands for hair/grass use cases.

    Returns
    -------
    pos : (S, N, D) float
        Corrected positions after `passes` sweeps.
    """
    pos = np.asarray(pos, dtype=float).copy()
    inv_mass = np.asarray(inv_mass, dtype=float)
    rest_bend = np.asarray(rest_bend, dtype=float)

    S, N, D = pos.shape
    inv_mass = np.broadcast_to(inv_mass, (S, N, 1))
    rest_bend = np.broadcast_to(rest_bend, (S, N - 2, 1))

    if anchors is not None:
        anchors = np.asarray(anchors, dtype=float).reshape(S, 1, D)

    def relax_skip(i: int):
        # endpoints of the bending "edge": i and i+2
        pa = pos[:, i,   :]          # (S, D)
        pc = pos[:, i+2, :]          # (S, D)

        wA = inv_mass[:, i,   :1]    # (S, 1)
        wC = inv_mass[:, i+2, :1]    # (S, 1)
        wsum = wA + wC + eps

        delta = pc - pa
        dist  = np.sqrt((delta * delta).sum(axis=-1, keepdims=True) + eps)
        dirn  = delta / dist
        rl    = rest_bend[:, i, :1]

        C     = dist - rl
        corr  = stiffness * C * dirn

        pa += (wA / wsum) * corr
        pc -= (wC / wsum) * corr

    for _ in range(int(passes)):
        for i in range(N - 2):
            relax_skip(i)
        if both_directions:
            for i in range(N - 3, -1, -1):
                relax_skip(i)
        if anchors is not None:
            pos[:, :1, :] = anchors

    return pos

# ====================================================================================================
# Forward per-joint cone limiter with a *constant* angle.
# ====================================================================================================

def project_relative_cone_constant(
    pos,            # (S, N, D)
    rest_len,       # (S, N-1, 1) or (1, N-1, 1) broadcastable
    inv_mass,       # (S, N, 1) ; 0 = fixed (e.g., root)
    theta=np.deg2rad(5.0),   # max turning angle per joint (radians)
    passes=1,                 # forward sweeps (base -> tip)
    eps=1e-8
):
    """
    Forward per-joint cone limiter with a *constant* angle.

    Goal
    ----
    Enforce, for every joint j (between edges j-1 and j), that the direction of
    edge j cannot deviate by more than `theta` from the direction of edge j-1.
    This is applied in a *forward* sweep, so moving a joint moves all children,
    which produces the desired “the rest of the curve follows” behavior.

    Shapes
    ------
    pos      : (S, N, D) current positions (D = 2 or 3), S splines, N points.
    rest_len : (S, N-1, 1) edge rest lengths (can be broadcast across S).
    inv_mass : (S, N, 1) inverse masses, 0 means fixed (e.g., root).

    Algorithm
    ---------
    For joints j = 1..N-2:
      u = normalize(p_j   - p_{j-1})             # parent edge direction
      v = normalize(p_{j+1} - p_j)               # current edge direction
      if angle(u, v) > theta:
          clamp v inside the cone around u with aperture theta:
              let c = cos(theta), s = sin(theta)
              w_hat = normalized component of v orthogonal to u
              v' = c*u + s*w_hat
          set p_{j+1} = p_j + L_j * v'           # preserve length L_j = rest_len[:, j]

    Notes
    -----
    * Only the child point p_{j+1} is moved (if it’s not fixed). This avoids fighting
      the root anchor and gives a clean forward propagation of the orientation.
    * Call this **after** your distance constraints (and optionally after a mild
      bending smoother) inside your outer loop.
    * Because the limit is *per joint*, the tip can accumulate up to ~(N-1)*theta
      relative to the root, matching the “constant rigidity” you want.

    Returns
    -------
    pos (same array, updated in-place and returned for convenience).
    """
    pos = np.asarray(pos, dtype=float)
    rest_len = np.asarray(rest_len, dtype=float)
    inv_mass = np.asarray(inv_mass, dtype=float)

    S, N, D = pos.shape
    if N < 3:
        return pos

    c = float(np.cos(theta))
    s = float(np.sqrt(max(0.0, 1.0 - c*c)))

    # Utility: get a unit vector orthogonal to u (robust in 2D/3D)
    def _orth_unit(u):
        # u: (S,D) unit
        if D == 2:
            # rotate by +90°: (-uy, ux)
            w = np.stack([-u[:, 1], u[:, 0]], axis=-1)
        else:  # D == 3
            # cross with a safe axis
            ax = np.array([1.0, 0.0, 0.0]) if np.abs(u[:, 0]).mean() < 0.9 else np.array([0.0, 1.0, 0.0])
            w = np.cross(u, ax[None, :])
            # if degeneracy (u parallel to ax), cross with the other axis
            bad = (np.linalg.norm(w, axis=-1, keepdims=True) < 1e-12).flatten()
            if np.any(bad):
                w_bad = np.cross(u[bad], np.array([0.0, 0.0, 1.0])[None, :])
                w[bad] = w_bad
        wn = np.linalg.norm(w, axis=-1, keepdims=True) + eps
        return w / wn

    free_child = (inv_mass[..., 0] > 0.0)  # (S,N) boolean

    for _ in range(int(passes)):
        # forward joints j = 1 .. N-2
        for j in range(1, N - 1):
            # parent edge (j-1)
            du = pos[:, j]   - pos[:, j-1]              # (S,D)
            un = np.linalg.norm(du, axis=-1, keepdims=True) + eps
            u = du / un

            # current edge (j)
            dv = pos[:, j+1] - pos[:, j]                # (S,D)
            vn = np.linalg.norm(dv, axis=-1, keepdims=True) + eps
            v = dv / vn

            dot = (u * v).sum(axis=-1, keepdims=True)   # (S,1)
            need = (dot < c) & free_child[:, j+1:j+2]   # (S,1)

            if not np.any(need):
                continue

            # orthogonal component; if nearly collinear, pick any orthogonal direction
            w = v - dot * u
            wn = np.linalg.norm(w, axis=-1, keepdims=True)
            w_hat = np.where(wn > 1e-8, w / (wn + eps), _orth_unit(u))

            # clamped direction
            v_prime = c * u + s * w_hat                 # (S,D)

            # place child at exact rest length
            L = np.broadcast_to(rest_len[:, j:j+1, 0], (S, 1))    # (S,1)
            p_j = pos[:, j:j+1, :]                                  # (S,1,D)
            p_next = p_j + L[..., None] * v_prime[:, None, :]        # (S,1,D)

            mask = need.astype(float)[:, :, None]                    # (S,1,1)
            pos[:, j+1:j+2, :] = mask * p_next + (1.0 - mask) * pos[:, j+1:j+2, :]

    return pos

# ====================================================================================================
# Test gravity
# ====================================================================================================

def step_gravity_only(pos, inv_mass, rest_len, rest_bend, *,
                      anchors=None, g=(0, 0, -9.81), dt=1/60,
                      edge_k=1.0, bend_k=0.2, sweeps=6):
    """
    1) Predict: x <- x + dt^2 * g  (seulement si inv_mass>0)
    2) Constraints: quelques sweeps GS de longueur puis bending.
    """
    g = np.asarray(g, float).reshape(1,1,-1)
    # predict (ne bouge pas les ancrés: inv_mass=0)
    pos = pos + (dt*dt) * g * inv_mass

    for _ in range(sweeps):
        project_distance_constraints_gs(pos, inv_mass, rest_len, stiffness=1.0, passes=1, both_directions=True)
        # optionnel, petit lissage:
        # project_bending_distance_simple(pos, inv_mass, rest_bend, stiffness=0.2, passes=1)
        project_relative_cone_constant(pos, rest_len, inv_mass,
                                    theta=np.deg2rad(5.0), passes=1)

    return pos





def test1():

    from . engine import engine
    from .curve import Curve

    rng = np.random.default_rng(0)

    S, N = 100, 10

    start = rng.uniform(-5, 5, (S, 3))
    start[:, 2] = 0
    curve = Curve.line(start=start, end=start + (0, 0, 3), resolution=N)

    curve.points.position += rng.normal(0, .01, curve.points.position.shape)


    curve.to_object("Splines BEFORE")

    L = 3.0 / (N - 1)
    rest_len = np.full((1, N-1, 1), L)

    inv_mass = np.ones((1, N, 1))
    inv_mass[0, 0] = 0
    inv_mass = np.resize(inv_mass, (1, N, 1))

    rest_bend = rest_len[:, :-1] + rest_len[:, 1:]


    pos = curve.points.position.reshape(S, N, 3)
    for _ in range(600):
        pos = step_gravity_only(
                        pos, 
                        inv_mass, rest_len, rest_bend,
                        anchors=None,
                        g=(0.01, -10, 0), 
                        dt=1/60,
                        edge_k=1.0, bend_k=0.2, 
                        sweeps=3)

    curve.points.position = pos.reshape(-1, 3)
    curve.to_object("Splines AFTER")














def project_bending_constraints(pos, inv_mass, rest_bend, stiffness=0.2):
    """
    Bending simple via distance i↔i+2, corrigé pour les shapes.
    rest_bend: (S,N-2,1)
    """
    S, N, D = pos.shape
    if N < 3: 
        return
    for i in range(N-2):
        delta = pos[:, i+2, :] - pos[:, i, :]                           # (S,D)
        dist  = np.linalg.norm(delta, axis=1, keepdims=True) + EPS      # (S,1)
        n     = delta / dist

        C     = dist - rest_bend[:, i, 0:1]                             # (S,1)
        wA    = inv_mass[:, i,   0:1]
        wC    = inv_mass[:, i+2, 0:1]
        wsum  = wA + wC + EPS

        corr  = stiffness * C * n                                       # (S,D)
        pos[:, i,   :] -= (wA/wsum) * corr
        pos[:, i+2, :] += (wC/wsum) * corr


def pbd_verlet_predict_no_forces(pos, prev_pos, inv_mass, dt, damping=0.1):
    """
    Prédiction sans forces ; NE BOUGE PAS les points ancrés (inv_mass=0).
    """
    v = (pos - prev_pos) * (1.0 - float(damping))   # (S,N,D)
    return pos + v * inv_mass                       # (S,N,D) ; anchors (w=0) restent fixes


def velocity_damp(pos, prev_pos, amount=0.05):
    if amount <= 0.0:
        return pos
    v = pos - prev_pos
    return prev_pos + (1.0 - float(amount)) * v


def pbd_step_A(pos, prev_pos, inv_mass, rest_len, anchors, dt,
               iters=4, gs_passes=2, damping=0.1, vel_damp=0.05,
               rest_bend=None, bend_stiffness=0.0):
    # 1) prédire (sans forces)
    pred = pbd_verlet_predict_no_forces(pos, prev_pos, inv_mass, dt, damping)
    new_prev = pos.copy()
    pos = pred

    # 2) contraintes (GS) + ancrage à chaque itération
    for _ in range(iters):
        project_distance_constraints_gs(pos, inv_mass, rest_len, passes=gs_passes)
        if rest_bend is not None and bend_stiffness > 0.0:
            project_bending_constraints(pos, inv_mass, rest_bend, stiffness=bend_stiffness)
        pos[:, 0, :] = anchors
    new_prev[:, 0, :] = anchors

    # 3) damping de vitesse post-projection
    pos = velocity_damp(pos, new_prev, amount=vel_damp)
    pos[:, 0, :] = anchors; new_prev[:, 0, :] = anchors
    return pos, new_prev


def diag_stretch(pos, rest_len):
    delta = pos[:,1:] - pos[:,:-1]
    dist  = np.sqrt((delta*delta).sum(axis=-1))
    rl    = rest_len[...,0]
    stretch = np.abs(dist - rl) / (rl + 1e-12)
    return float(stretch.max()), float(stretch.mean())

def test():
    # 5000 brins, 16 points, 3D
    S, N, D = 1000, 16, 3
    # racines posées sur une grille (champ de blé)
    gx = int(np.sqrt(S)); gy = (S + gx - 1)//gx
    xs = np.linspace(-5, 5, gx); ys = np.linspace(-5, 5, gy)
    X, Y = np.meshgrid(xs, ys, indexing='xy')
    anchors = np.stack([X.ravel()[:S], np.zeros(S), Y.ravel()[:S]], axis=-1)  # (S,3)

    from .engine import engine
    from .curve import Curve

    # après build_straight_chains(...)
    edge_lambdas = np.zeros((S, N-1, 1), float)

    pos, prev_pos, inv_mass, rest_len, rest_bend = build_straight_chains(
        anchors, n_points=N, dir=(0,1,0), rest=0.08
    )
    param = {'pos': pos, 'prev_pos': prev_pos, 'edge_lambdas': edge_lambdas}

    def update():
        pos = param['pos']; prev_pos = param['prev_pos']; lamb = param['edge_lambdas']

        wf = wind_field(engine.time, amplitude=2.0, wavelength=2.0, swirl=0.5)  # un peu plus doux

        print("CALL WITH delta_time :", engine.delta_time)
        print(pos[:1])

        pos, prev_pos = pbd_step_A(
            pos, prev_pos, inv_mass, rest_len,
            anchors=anchors,
            dt=engine.delta_time,
            iters=4,          # 3–5 ok
            gs_passes=2,      # aller+retour
            damping=0.1,
            vel_damp=0.05
        )

        print("Result")
        print(pos[:1])
        print()

        # debug toutes les ~20 frames :
        if (int(engine.time*60) % 20) == 0:
            mmax, mmean = diag_stretch(pos, rest_len)
            # print ou log chez toi :
            print("stretch max/mean:", mmax, mmean)

        param['pos'] = pos
        param['prev_pos'] = prev_pos
        # (lamb reste en place dans param pour la frame suivante)

        #print("DEBUG", pos.shape)
        #print(pos[:5])

        curve = Curve(points=pos, splines=N, curve_type=1)  
        curve.to_object("Field")

    engine.go(update, subframes=0)














            


















def pbd_verlet_predict(pos, prev_pos, inv_mass, dt,
                       gravity=(0, -9.81, 0), damping=0.12,
                       accel_extra=None, vel_clamp=None):
    """
    Verlet: pos' = pos + (1-damping)*(pos - prev_pos) + dt^2 * a
    - inv_mass: (S,N,1) ; 0 => ancré (ne bouge pas à la prédiction)
    - vel_clamp: float | None, vitesse max par pas (optionnel)
    """
    S, N, D = pos.shape
    g = np.asarray(gravity, float).reshape(1, 1, -1)
    a = np.broadcast_to(g, (S, N, D)).copy()
    if accel_extra is not None:
        a += accel_extra(pos)  # (S,N,D)

    v = pos - prev_pos                          # (S,N,D)
    if vel_clamp is not None:
        vn = np.linalg.norm(v, axis=-1, keepdims=True) + EPS
        v = v * np.minimum(1.0, vel_clamp / vn)

    v *= (1.0 - damping)
    delta = v + (dt * dt) * a

    # <<< NE PAS BOUGER LES ANCRÉS À LA PRÉDICTION >>>
    # si inv_mass vaut {0,1}, c'est parfait ; sinon mets 0/1 dans inv_mass pour les racines
    new_pos = pos + delta * inv_mass
    return new_pos


def project_distance_constraints(pos, inv_mass, rest_len, stiffness=1.0):
    """
    Contrainte de longueur pour tous les segments (i,i+1), vectorisée.
    pos: (S,N,D), inv_mass: (S,N,1), rest_len: (S,N-1,1)
    stiffness in (0,1]: fraction de correction appliquée (Jacobi-like).
    """
    # deltas sur toutes les arêtes
    delta = pos[:, 1:] - pos[:, :-1]                    # (S,N-1,D)
    dist = np.sqrt((delta * delta).sum(axis=-1, keepdims=True) + EPS)  # (S,N-1,1)
    diff = (dist - rest_len)                            # (S,N-1,1)
    dirn = delta / dist                                 # (S,N-1,D)
    corr = stiffness * diff * dirn                      # (S,N-1,D)

    wL = inv_mass[:, :-1]                               # (S,N-1,1)
    wR = inv_mass[:,  1:]                               # (S,N-1,1)
    wsum = wL + wR + EPS
    # répartir proportionnellement aux masses inverses
    pos[:, :-1] -= (wL / wsum) * corr
    pos[:,  1:] += (wR / wsum) * corr

def project_distance_constraints_xpbd(pos, inv_mass, rest_len, lambdas, dt,
                                      compliance=0.0, max_corr=None):
    """
    XPBD distance (i,i+1) pour toutes les chaînes, vectorisé.
    pos:      (S,N,D)
    inv_mass: (S,N,1)
    rest_len: (S,N-1,1)
    lambdas:  (S,N-1,1) accumulateur de multiplicateurs de Lagrange (persiste d'une frame à l'autre)
    dt:       float
    compliance: α (>=0). 0 => PBD “incompressible”, >0 => plus stable mais un peu extensible
    max_corr: float|None, limite la correction par itération (sécurité)
    """
    EPS = 1e-8
    delta = pos[:, 1:] - pos[:, :-1]                          # (S,N-1,D)
    dist  = np.sqrt((delta*delta).sum(axis=-1, keepdims=True) + EPS)  # (S,N-1,1)
    n     = delta / dist                                       # (S,N-1,D)
    C     = dist - rest_len                                    # (S,N-1,1)

    wL = inv_mass[:, :-1]                                      # (S,N-1,1)
    wR = inv_mass[:,  1:]                                      # (S,N-1,1)
    wsum = wL + wR                                             # (S,N-1,1)

    alpha_tilde = float(compliance) / (dt*dt)                  # (scalaire)
    denom = wsum + alpha_tilde                                 # (S,N-1,1)

    dlam = (-C - alpha_tilde * lambdas) / (denom + EPS)        # (S,N-1,1)
    corr = dlam * n                                            # (S,N-1,D)

    if max_corr is not None:
        cn = np.linalg.norm(corr, axis=-1, keepdims=True) + EPS
        corr = corr * np.minimum(1.0, float(max_corr) / cn)

    pos[:, :-1] -= wL * corr
    pos[:,  1:] += wR * corr
    lambdas += dlam

def velocity_damp(pos, prev_pos, amount=0.05):
    """Damping simple des vitesses implicites (toujours stable)."""
    if amount <= 0.0:
        return pos
    v = pos - prev_pos
    return prev_pos + (1.0 - float(amount)) * v

def project_bending_constraints(pos, inv_mass, rest_bend, stiffness=0.2):
    """
    Bending simple via distance i↔i+2 (optionnel, pour lisser).
    rest_bend: (S,N-2,1)
    """
    delta = pos[:, 2:] - pos[:, :-2]                    # (S,N-2,D)
    dist = np.sqrt((delta * delta).sum(axis=-1, keepdims=True) + EPS)  # (S,N-2,1)
    diff = (dist - rest_bend)
    dirn = delta / dist
    corr = stiffness * diff * dirn                      # (S,N-2,D)

    wA = inv_mass[:, :-2]       # i
    wC = inv_mass[:,  2:]       # i+2
    wsum = wA + wC + EPS
    pos[:, :-2] -= (wA / wsum) * corr
    pos[:,  2:] += (wC / wsum) * corr

def pbd_step(pos, prev_pos, inv_mass, rest_len, anchors=None, dt=1/60,
             gravity=(0,-9.81,0), damping=0.12, iters=2,
             edge_stiffness=1.0,  # ignoré par XPBD, gardé pour compat
             rest_bend=None, bend_stiffness=0.15, accel_extra=None,
             edge_lambdas=None, edge_compliance=1e-6, max_edge_corr=None,
             vel_damp=0.05):
    """
    Pas PBD/XPBD vectorisé.
    - edge_lambdas: (S,N-1,1) persistant pour XPBD (obligatoire avec XPBD)
    - edge_compliance: α (>=0). 0 => PBD rigide, 1e-6..1e-4 stabilisent bien
    - max_edge_corr: limite la correction par itération (ex. 0.2*rest moyen)
    - vel_damp: damping post-projection (5% par défaut)
    """
    # 1) prédiction Verlet (ancrés masqués par inv_mass)
    pred = pbd_verlet_predict(pos, prev_pos, inv_mass, dt, gravity, damping, accel_extra)
    new_prev = pos.copy()
    pos = pred

    # 2) projections
    if edge_lambdas is None:
        raise ValueError("XPBD: fournir edge_lambdas de shape (S,N-1,1).")
    for _ in range(iters):
        project_distance_constraints_xpbd(
            pos, inv_mass, rest_len, edge_lambdas, dt,
            compliance=edge_compliance, max_corr=max_edge_corr
        )
        if rest_bend is not None and bend_stiffness > 0.0:
            project_bending_constraints(pos, inv_mass, rest_bend, stiffness=bend_stiffness)
        if anchors is not None:
            pos[:, 0] = anchors

    # 3) damping de vitesse après corrections (stabilise l’énergie)
    pos = velocity_damp(pos, new_prev, amount=vel_damp)

    # 4) recoller les ancrés dans prev (pas de "vitesse fantôme")
    if anchors is not None:
        new_prev[:, 0] = anchors
        pos[:, 0] = anchors

    return pos, new_prev


# Utilitaires d'init
def build_straight_chains(anchors, n_points=12, dir=(0,1,0), rest=0.05):
    """
    anchors: (S,D) racines
    Retourne pos, prev_pos, inv_mass, rest_len, rest_bend.
    """
    anchors = np.asarray(anchors, float)
    S, D = anchors.shape
    dir = np.asarray(dir, float)[:D]
    dir = dir / (np.linalg.norm(dir) + EPS)
    # pos initial: segments alignés le long de dir
    offsets = (np.arange(n_points).reshape(1, n_points, 1) * rest * dir.reshape(1, 1, D))
    pos = anchors[:, None, :] + offsets                   # (S,N,D)
    prev_pos = pos.copy()
    inv_mass = np.ones((S, n_points, 1), float)
    inv_mass[:, 0, 0] = 0.0                               # racine ancrée
    rest_len = np.full((S, n_points-1, 1), rest, float)
    # bending i↔i+2 = 2*rest
    rest_bend = np.full((S, n_points-2, 1), 2.0 * rest, float)
    return pos, prev_pos, inv_mass, rest_len, rest_bend


def wind_field(time, amplitude=4.0, wavelength=1.5, swirl=0.7):
    def f(pos):
        # pos: (S,N,D)
        x = pos[..., 0]
        y = pos[..., 1]
        z = pos[..., 2] if pos.shape[-1] > 2 else 0.0
        # petite oscillation sinusoïdale + swirl
        w = (np.sin((x+z)* (2*np.pi/wavelength) + 1.7*time) * (1.0 - 0.2*y)
             + swirl*np.cos((y+0.3*z)*(2*np.pi/wavelength) + 0.9*time))
        ax = amplitude * 0.4 * w
        ay = amplitude * 0.1 * np.sin(0.5*time + 0.5*x)
        az = amplitude * 0.4 * w
        if pos.shape[-1] == 2:
            return np.stack([ax, ay], axis=-1)
        return np.stack([ax, ay, az], axis=-1)
    return f


