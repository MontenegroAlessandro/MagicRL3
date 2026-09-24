"""Linear Quadratic Regulator env.

s_{t+1} = A s_t + B a_t + noise
r_{t+1} = - ( s_t^T Q s_t + a_t^T R a_t + 2 a_t^T M_mix s_t )

Arbitrary A, B, Q, R (scalar or matrix), optional cross term `M_mix` and
terminal cost `Q_final`, with state_dim != action_dim supported, plus a
closed-form value toolkit (infinite- and finite-horizon optimal gains/returns,
V/Q functions, quadratic Q-features).

The environment owns the *system* only: it has no discount and no horizon, as
those belong to the algorithm and to the usual Gymnasium time limit. Every
closed-form helper therefore takes `discount` (and, where relevant, `horizon`)
as an explicit argument. `Q_final` is likewise an argument of the finite-horizon
helpers only: a running environment has no way of knowing which step is the
last, so `step` never adds a terminal cost.

The default system is A = 0.9 I, B = 0.9 I, Q = R = I, uniform initial state
on [-5, 5]^d, no process noise. The initial state can instead be Gaussian
(`init_dist="gaussian"`) or fixed and deterministic (`init_dist="fixed"`,
`init_state=...`).

Gain convention: K is `action_dim x state_dim` and the linear policy is
`a = K s` (consistent with `computeOptimalK` and the rest of the repo).

Interface: a plain `gymnasium.Env`, registered once as "LQR-v0", so every
algorithm in this repo can use it. Choose the system when you build it::

    gym.make("LQR-v0", state_dim=3, action_dim=2, noise=0.1)
    gym.make("LQR-v0", state_dim=3, max_episode_steps=50)   # optional time limit
    make_vec_env("LQR-v0", n_envs=8, env_kwargs=dict(state_dim=3, action_dim=2))

Episodes never terminate and the environment never truncates them: the horizon
is whatever the caller imposes, either Gymnasium's `max_episode_steps` time
limit or the algorithm's own `n_steps` cap. Without a time limit the episode
statistics SB3 derives from episode ends (`rollout/ep_rew_mean`) stay empty;
`eval/mean_discounted_return` is unaffected.

All randomness (initial state and process noise) comes from the environment's
seeded `np_random`, which the finite-difference algorithms rely on to replay or
pair rollouts.
"""

# imports
import warnings
from numbers import Number

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding


# class
class LQR(gym.Env):
    """Gymnasium environment implementing an LQR problem."""

    metadata = {"render_modes": [], "render_fps": 30}

    def __init__(
            self,
            action_dim=1,
            state_dim=1,
            noise=0,
            max_action=10.0,
            seed=None,
            render_mode=None,
            # --- optional generalisations ---
            A=None,
            B=None,
            Q=None,
            R=None,
            M_mix=0.0,
            Q_final=0.0,   # terminal cost: used by the finite-horizon helpers only
            init_dist="uniform",
            init_bound=5.0,
            init_mean=0.0,
            init_std=1.0,
            init_state=None,
            check_controllability=False
        ) -> None:

        super().__init__()

        self.name = "LQR"

        # ---- system matrices --------------------------------------------------
        # "dimension mode": dimensions are given, matrices fall back to the
        # defaults (A = 0.9 I, B = 0.9 I, Q = R = I).
        # "matrix mode": A is provided, dimensions are inferred from A and B.
        # A scalar A (or B) means that scalar times the identity of the given size.
        if A is None or np.isscalar(A):
            self.state_dim = int(state_dim)
            A = 0.9 if A is None else A
            A = float(A) * np.eye(self.state_dim)
        else:
            A = self._as_matrix(A)
            if A.shape[0] != A.shape[1]:
                raise ValueError("A must be a square matrix")
            self.state_dim = A.shape[0]
        if B is None or np.isscalar(B):
            self.action_dim = int(action_dim)
            B = 0.9 if B is None else B
            B = float(B) * np.eye(self.state_dim, self.action_dim)
        else:
            B = self._as_matrix(B, rows=self.state_dim)
            self.action_dim = B.shape[1]

        ds, da = self.state_dim, self.action_dim
        if ds < 1 or da < 1:
            raise ValueError("state_dim and action_dim must be >= 1")
        Q = np.eye(ds) if Q is None else Q
        R = np.eye(da) if R is None else R

        self.A = self._as_matrix(A, rows=ds, cols=ds)
        self.B = self._as_matrix(B, rows=ds, cols=da)
        self.Q = self._symmetric_matrix(Q, ds, "Q", strict=True)
        self.R = self._symmetric_matrix(R, da, "R", strict=True)
        self.Q_final = self._symmetric_matrix(Q_final, ds, "Q_final", strict=False)

        # cross term: cost contribution 2 a^T M_mix s  (M_mix is da x ds)
        if np.isscalar(M_mix):
            M_mix = np.zeros((da, ds)) if np.isclose(M_mix, 0.0) \
                else M_mix * np.ones((da, ds))
        M_mix = np.asarray(M_mix, dtype=float)
        if M_mix.shape != (da, ds):
            raise ValueError(f"M_mix should be a {da}x{ds} matrix")
        self.M_mix = M_mix

        if check_controllability:
            self._check_controllability()

        # ---- bounds & noise ---------------------------------------------------
        # max_pos bounds the *reward* analysis only: the observation space is
        # unbounded, because a linear system's state is not confined to a box.
        self.max_pos = 10.0 * np.ones(ds)
        self.max_action = max_action * np.ones(da)
        # scalar `noise` -> per-dimension std vector
        if np.isscalar(noise):
            noise_std = float(noise) * np.ones(ds)
        else:
            noise_std = np.asarray(noise, dtype=float).ravel()
            if noise_std.shape != (ds,):
                raise ValueError(f"noise should be a scalar or a vector of {ds} stds")
        if np.any(noise_std < 0):
            raise ValueError("noise standard deviations must be non-negative")
        self.noise_std = noise_std
        self.sigma_noise = np.diag(noise_std)

        # ---- initial-state distribution --------------------------------------
        # Uniform on [-init_bound, init_bound]^ds, Gaussian (mean, std), or a fixed,
        # deterministic `init_state` (stored as a zero-variance mean).
        # `init_second_moment` is the E[x0 x0^T] used by the closed forms.
        self.init_dist = init_dist
        self.init_bound = None
        self.init_mean = np.zeros(ds)
        self.init_std = np.zeros(ds)
        if init_dist == "uniform":
            self.init_bound = init_bound * np.ones(ds) if np.isscalar(init_bound) \
                else np.asarray(init_bound, dtype=float)
            if self.init_bound.shape != (ds,):
                raise ValueError(f"init_bound should be a scalar or a vector of {ds} bounds")
            self.init_second_moment = np.diag((self.init_bound ** 2) / 3.0)
        elif init_dist == "gaussian":
            self.init_mean = init_mean * np.ones(ds) if np.isscalar(init_mean) \
                else np.asarray(init_mean, dtype=float)
            self.init_std = init_std * np.ones(ds) if np.isscalar(init_std) \
                else np.asarray(init_std, dtype=float)
            if self.init_mean.shape != (ds,) or self.init_std.shape != (ds,):
                raise ValueError(f"init_mean/init_std should be scalars or vectors of {ds} entries")
            self.init_second_moment = (np.outer(self.init_mean, self.init_mean)
                                       + np.diag(self.init_std ** 2))
        elif init_dist == "fixed":
            if init_state is None:
                raise ValueError("init_dist='fixed' needs an init_state")
            self.init_mean = init_state * np.ones(ds) if np.isscalar(init_state) \
                else np.asarray(init_state, dtype=float).ravel()
            if self.init_mean.shape != (ds,):
                raise ValueError(f"init_state should be a scalar or a vector of {ds} entries")
            self.init_second_moment = np.outer(self.init_mean, self.init_mean)
        else:
            raise ValueError("init_dist must be 'uniform', 'gaussian' or 'fixed'")

        # ---- gymnasium spaces -------------------------------------------------
        self.action_space = spaces.Box(
            low=-self.max_action,
            high=self.max_action,
            shape=(da,),
            dtype=np.float64
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(ds,),
            dtype=np.float64
        )

        # An LQR has nothing to draw. The argument is accepted (and ignored) because
        # SB3's make_vec_env passes render_mode="rgb_array" by default, and refusing it
        # would make the environment unusable through the repo's runner.
        self.render_mode = None

        # ---- initialise -------------------------------------------------------
        # Step counter kept for information only: the horizon is imposed from outside.
        self.timestep = 0
        self.state = np.zeros(ds)
        self.seed(seed)
        self.reset(seed=seed)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _as_matrix(M, rows=None, cols=None):
        """Coerce a scalar / vector / matrix into a 2-D array (and validate)."""
        if np.isscalar(M):
            if rows is not None and cols is not None:
                return float(M) * np.eye(rows, cols)
            return float(M) * np.eye(1)
        M = np.asarray(M, dtype=float)
        if M.ndim == 1 and rows is not None and M.shape[0] == rows:
            M = M[:, None]
        if M.ndim != 2:
            raise ValueError("expected a 2-D matrix")
        if rows is not None and M.shape[0] != rows:
            raise ValueError(f"matrix should have {rows} rows, got {M.shape[0]}")
        if cols is not None and M.shape[1] != cols:
            raise ValueError(f"matrix should have {cols} cols, got {M.shape[1]}")
        return M

    @staticmethod
    def _symmetric_matrix(M, dim, name, strict):
        if np.isscalar(M):
            M = float(M) * np.eye(dim)
        M = np.asarray(M, dtype=float)
        if M.shape != (dim, dim):
            raise ValueError(f"{name} should be a {dim}x{dim} matrix")
        if not np.allclose(M, M.T):
            raise ValueError(f"{name} should be symmetric")
        eig = np.linalg.eigvalsh(M)
        if strict and not np.all(eig > 0):
            raise ValueError(f"{name} should be symmetric positive definite")
        if (not strict) and not np.all(eig >= -1e-10):
            raise ValueError(f"{name} should be symmetric positive semi-definite")
        return M

    def _check_controllability(self):
        powers = [self.B]
        for _ in range(self.state_dim - 1):
            powers.append(self.A @ powers[-1])
        C = np.concatenate(powers, axis=1)
        if np.linalg.matrix_rank(C) < self.state_dim:
            warnings.warn("The system is not controllable!", UserWarning)

    def _gain(self, K):
        """Coerce a gain into the (action_dim, state_dim) convention a = K s."""
        return np.asarray(K, dtype=float).reshape(self.action_dim, self.state_dim)

    @staticmethod
    def _check_discount(discount, what):
        """The infinite-horizon closed forms are geometric series: they need gamma < 1."""
        if not 0 < discount < 1:
            raise ValueError(f"{what} needs a discount strictly inside (0, 1), got {discount}")

    # --------------------------------------------------------------- gym API
    def step(self, action):
        u = np.ravel(np.asarray(action, dtype=float))
        if u.shape != (self.action_dim,):
            raise ValueError(f"action should have {self.action_dim} entries, got {u.shape}")

        cost = (self.state @ self.Q @ self.state
                + u @ self.R @ u
                + 2.0 * (u @ self.M_mix @ self.state))

        # Process noise is drawn from the environment's own generator, so a run is
        # reproducible from its reset seed -- which FDPG's common random numbers and
        # ZDPG's state replay both depend on.
        noise = self.sigma_noise @ self.np_random.standard_normal(self.state_dim)
        self.state = np.ravel(self.A @ self.state + self.B @ u + noise)

        self.timestep += 1
        # An LQR problem has no terminal states, and the horizon is not the
        # environment's business: a TimeLimit wrapper (max_episode_steps) or the
        # algorithm's n_steps cap decides when a rollout stops.
        return self.get_state(), -float(cost), False, False, {}

    def reset(self, *, seed=None, options=None):
        """`options={"state": s}` starts the episode from a chosen state."""
        super().reset(seed=seed)
        self.timestep = 0
        state = None if options is None else options.get("state")
        if state is not None:
            self.set_state(state)
        elif self.init_dist == "uniform":
            self.state = self.np_random.uniform(low=-self.init_bound, high=self.init_bound)
        elif self.init_dist == "fixed":
            self.state = self.init_mean.copy()
        else:  # gaussian
            self.state = (self.init_mean
                          + self.np_random.standard_normal(self.state_dim) * self.init_std)
        return self.get_state(), {}

    def get_state(self):
        return np.array(self.state, dtype=np.float64)

    def set_state(self, state):
        state = np.ravel(np.asarray(state, dtype=float))
        if state.shape != (self.state_dim,):
            raise ValueError(f"state should have {self.state_dim} entries, got {state.shape}")
        self.state = state

    def seed(self, seed=None):
        """Reseed without resetting. Gymnasium seeds through reset(seed=...); this
        stays for callers that seed the environment explicitly."""
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self):
        """LQR is not renderable; `render_mode` is accepted only for API compatibility."""
        return None

    def r_max(self, max_action=None):
        """Upper bound on the per-step cost over |s| <= max_pos, |a| <= max_action."""
        bound_a = self.max_action if max_action is None else max_action * np.ones(self.action_dim)
        # |x^T Q x| <= |x|^T |Q| |x|: with off-diagonal entries of either sign the
        # maximum over the box is not at x = max_pos, so bound entrywise.
        state_term = float(self.max_pos @ np.abs(self.Q) @ self.max_pos)
        action_term = float(bound_a @ np.abs(self.R) @ bound_a)
        cross_term = 2.0 * abs(float(bound_a @ np.abs(self.M_mix) @ self.max_pos))
        return state_term + action_term + cross_term

    def computer_r_max(self, episodes=None):
        """Backward-compatible alias of `r_max` (`episodes` is ignored)."""
        return self.r_max()

    # ============================================================ value toolkit
    # All methods use the convention a = K s with K of shape (action_dim, state_dim).
    def _closed_loop_P(self, K, discount, max_iterations=100):
        """Riccati P for the *fixed* linear policy a = K s (policy evaluation).

        Solves P = S + discount (A + B K)^T P (A + B K), with
        S = Q + K^T R K + K^T M_mix + M_mix^T K the closed-loop stage-cost matrix.

        The Lyapunov equation is solved exactly through its vectorised form
        (I - discount A_cl^T (x) A_cl^T) vec(P) = vec(S). When
        discount * rho(A_cl)^2 >= 1 the discounted cost diverges and P is +inf.
        `max_iterations` is kept for backward compatibility and ignored.
        """
        K = self._gain(K)
        A_cl = self.A + self.B @ K
        S = self.Q + K.T @ self.R @ K + K.T @ self.M_mix + self.M_mix.T @ K
        rho = np.max(np.abs(np.linalg.eigvals(A_cl)))
        if discount * rho ** 2 >= 1.0:
            return np.full_like(S, np.inf)
        n = self.state_dim
        lhs = np.eye(n * n) - discount * np.kron(A_cl.T, A_cl.T)
        P = np.linalg.solve(lhs, S.reshape(-1)).reshape(n, n)
        return 0.5 * (P + P.T)

    def _computeP2(self, K, *, discount, max_iterations=100):
        """Backward-compatible alias for the policy-evaluation Riccati."""
        return self._closed_loop_P(K, discount, max_iterations)

    def discounted_P_matrix(self, discount, max_iterations=10_000):
        """Optimal (control) Riccati matrix for the discounted problem."""
        P = self.Q.copy()
        for _ in range(max_iterations):
            inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
            M = self.M_mix + discount * self.B.T @ P @ self.A
            P_next = self.Q + discount * (self.A.T @ P @ self.A) - M.T @ inverse @ M
            # Tight tolerance: with discount close to 1 the increments shrink slowly,
            # and a loose test would stop far from the fixed point.
            if np.allclose(P_next, P, rtol=1e-12, atol=1e-12):
                return P_next
            P = P_next
        warnings.warn("Computation of optimal P did not converge")
        return P

    def discounted_optimal_gain(self, discount, max_iterations=10_000):
        """Optimal discounted gain K* (action_dim x state_dim), a = K* s."""
        P = self.discounted_P_matrix(discount, max_iterations)
        inverse = np.linalg.inv(self.R + discount * self.B.T @ P @ self.B)
        return - inverse @ (self.M_mix + discount * self.B.T @ P @ self.A)

    def computeOptimalK(self, discount):
        """Optimal linear controller (a = K s). Backward-compatible name."""
        return self.discounted_optimal_gain(discount)

    def _noise_terms(self, P, discount, policy_std):
        """Constant additive return terms from state noise and policy noise.

        Per step the policy noise costs tr(Sigma R) immediately and
        discount * tr(Sigma B^T P B) through the next state; the process noise
        costs discount * tr(W P). Summing the geometric series gives the terms below.
        """
        self._check_discount(discount, "the infinite-horizon value")
        state_term = discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1.0 - discount)
        action_term = np.trace((self.R + discount * self.B.T @ P @ self.B)
                               * (policy_std ** 2)) / (1.0 - discount)
        return state_term + action_term

    def discounted_optimal_return(self, discount, policy_std=0., max_iterations=10_000):
        """Closed-form optimal discounted return (scalar policy std)."""
        P = self.discounted_P_matrix(discount, max_iterations)
        init_term = np.trace(P @ self.init_second_moment)
        return - init_term - self._noise_terms(P, discount, policy_std)

    def computeJ(self, K, *, discount, Sigma=1., n_random_x0=None, max_iterations=100):
        """Discounted return of the linear policy a = K s + N(0, Sigma).

        Closed form (matrix, multi-dimensional). `n_random_x0` is accepted for
        backward compatibility and ignored. Sigma may be a scalar or a matrix.
        """
        self._check_discount(discount, "computeJ")
        P = self._closed_loop_P(K, discount, max_iterations)
        if not np.all(np.isfinite(P)):  # unstable closed loop: the cost diverges
            return -np.inf
        if np.isscalar(Sigma):
            Sigma = float(Sigma) * np.eye(self.action_dim)
        Sigma = np.asarray(Sigma, dtype=float)

        init_term = np.trace(P @ self.init_second_moment)
        state_term = discount * np.trace(np.diag(self.noise_std ** 2) @ P) / (1.0 - discount)
        action_term = np.trace(Sigma @ (self.R + discount * self.B.T @ P @ self.B)) / (1.0 - discount)
        return - init_term - state_term - action_term

    def discounted_v(self, state, policy_param, *, discount, policy_std=0., max_iterations=100):
        """State-value V(s) of the linear policy a = K s (+ noise)."""
        P = self._closed_loop_P(policy_param, discount, max_iterations)
        if not np.all(np.isfinite(P)):  # unstable closed loop: the cost diverges
            return -np.inf
        state = np.ravel(state)
        return - state @ P @ state - self._noise_terms(P, discount, policy_std)

    def discounted_q(self, state, action, policy_param, *, discount, policy_std=0., max_iterations=100):
        """Action-value Q(s, a) of the linear policy a = K s (+ noise)."""
        self._check_discount(discount, "discounted_q")
        P = self._closed_loop_P(policy_param, discount, max_iterations)
        if not np.all(np.isfinite(P)):  # unstable closed loop: the cost diverges
            return -np.inf
        state, action = np.ravel(state), np.ravel(action)
        Q_11 = self.Q + discount * self.A.T @ P @ self.A
        Q_12 = self.M_mix.T + discount * self.A.T @ P @ self.B
        Q_21 = self.M_mix + discount * self.B.T @ P @ self.A
        Q_22 = self.R + discount * self.B.T @ P @ self.B
        sa_term = - (state @ Q_11 @ state
                     + state @ Q_12 @ action
                     + action @ Q_21 @ state
                     + action @ Q_22 @ action)
        # The two noise sources enter differently here. The first transition's process
        # noise already acts, so its constant is the same as the V-function's,
        # discount * tr(W P) / (1 - discount); the first *action* is given, so the policy
        # noise only starts at t = 1 and carries one extra factor `discount`.
        W = np.diag(self.noise_std ** 2)
        state_term = discount * np.trace(W @ P) / (1.0 - discount)
        action_term = discount * np.trace((self.R + discount * self.B.T @ P @ self.B)
                                          * (policy_std ** 2)) / (1.0 - discount)
        return sa_term - state_term - action_term

    def q_representation(self, state, action):
        """Quadratic feature vector phi(s, a) for a linear Q-function model."""
        state, action = np.ravel(state), np.ravel(action)
        if state.shape != (self.state_dim,) or action.shape != (self.action_dim,):
            raise ValueError("Invalid state or action shape")
        x = np.concatenate((state, action))
        outer = np.outer(x, x)
        triu = outer[np.triu_indices(self.state_dim + self.action_dim)]
        return np.concatenate((np.ones(1), triu))

    # ---------------------------------------------------- finite-horizon optimum
    def P_matrices(self, horizon, discount):
        """Time-varying optimal Riccati matrices [P_0, ..., P_H = Q_final]."""
        Ps = [self.Q_final]
        for _ in range(horizon):
            P_next = Ps[0]
            M = self.M_mix + discount * self.B.T @ P_next @ self.A
            inverse = np.linalg.inv(self.R + discount * self.B.T @ P_next @ self.B)
            P = self.Q + discount * (self.A.T @ P_next @ self.A) - M.T @ inverse @ M
            Ps.insert(0, P)
        return Ps

    def optimal_gains(self, horizon, discount):
        """Time-varying optimal gains [K_0, ..., K_{H-1}], a_t = K_t s_t."""
        Ps = self.P_matrices(horizon, discount)
        return [- np.linalg.inv(self.R + discount * self.B.T @ Ps[h + 1] @ self.B)
                @ (self.M_mix + discount * self.B.T @ Ps[h + 1] @ self.A)
                for h in range(horizon)]

    def optimal_return(self, horizon, discount, policy_std=0.):
        """Closed-form optimal finite-horizon return (scalar policy std).

        Stage h is discounted by discount^h, so its process noise (which is felt
        through P_{h+1}) carries discount^(h+1), and its action noise carries
        discount^h immediately plus discount^(h+1) through the next state.
        """
        Ps = self.P_matrices(horizon, discount)
        W = np.diag(self.noise_std ** 2)
        init_term = np.trace(Ps[0] @ self.init_second_moment)
        state_term = sum(discount ** (h + 1) * np.trace(W @ Ps[h + 1]) for h in range(horizon))
        action_term = sum(discount ** h
                          * np.trace((self.R + discount * self.B.T @ Ps[h + 1] @ self.B)
                                     * (policy_std ** 2))
                          for h in range(horizon))
        return - init_term - state_term - action_term

    def finite_horizon_return(self, K, horizon, discount, policy_std=0.):
        """Closed-form finite-horizon return of the *fixed* gain a = K s (+ noise)."""
        K = self._gain(K)
        A_cl = self.A + self.B @ K
        S = self.Q + K.T @ self.R @ K + K.T @ self.M_mix + self.M_mix.T @ K
        W = np.diag(self.noise_std ** 2)
        Sigma = (policy_std ** 2) * np.eye(self.action_dim)

        P = self.Q_final.copy()
        constant = 0.0
        for _ in range(horizon):  # backward recursion over the stages
            constant = (np.trace(Sigma @ (self.R + discount * self.B.T @ P @ self.B))
                        + discount * np.trace(W @ P) + discount * constant)
            P = S + discount * (A_cl.T @ P @ A_cl)
        return - np.trace(P @ self.init_second_moment) - constant

    # ------------------------------------------------- legacy scalar gradients
    def _check_scalar_identity_system(self):
        """The closed-form gradients below hold for the scalar system A = B = 1, M_mix = 0."""
        if (self.state_dim != 1 or self.action_dim != 1
                or self.A[0, 0] != 1.0 or self.B[0, 0] != 1.0 or self.M_mix[0, 0] != 0.0):
            raise NotImplementedError("closed-form gradients need a scalar system "
                                      "with A = B = 1 and M_mix = 0")

    def grad_K(self, K, Sigma, *, discount):
        """Policy gradient wrt K (scalar A = B = I case only)."""
        self._check_scalar_identity_system()
        if not isinstance(K, Number) or not isinstance(Sigma, Number):
            raise NotImplementedError
        self._check_discount(discount, "grad_K")
        theta, sigma = float(K), float(Sigma)
        q, r = float(self.Q[0, 0]), float(self.R[0, 0])
        den = 1 - discount * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * r / den
                      + discount * (q + theta ** 2 * r) * (1 + theta) / den ** 2)
        # J = -P E[x0^2] - (sigma (r + discount P) + discount w P) / (1 - discount),
        # with w the process-noise variance and E[x0^2] from `init_second_moment`
        w = float(self.noise_std[0] ** 2)
        return float(- dePdeK * (self.init_second_moment[0, 0]
                                 + discount * (sigma + w) / (1 - discount)))

    def grad_Sigma(self, K, Sigma=None, *, discount):
        self._check_scalar_identity_system()
        if not isinstance(K, Number):
            raise NotImplementedError
        self._check_discount(discount, "grad_Sigma")
        P = self._computeP2(K, discount=discount)
        return float(-(self.R[0, 0] + discount * P[0, 0]) / (1 - discount))

    def grad_mixed(self, K, Sigma=None, *, discount):
        self._check_scalar_identity_system()
        if not isinstance(K, Number):
            raise NotImplementedError
        self._check_discount(discount, "grad_mixed")
        theta = float(K)
        q, r = float(self.Q[0, 0]), float(self.R[0, 0])
        den = 1 - discount * (1 + 2 * theta + theta ** 2)
        dePdeK = 2 * (theta * r / den
                      + discount * (q + theta ** 2 * r) * (1 + theta) / den ** 2)
        return float(-dePdeK * discount / (1 - discount))

    def computeQFunction(self, x, u, K, Sigma, *, discount, n_random_xn=100):
        """Monte-Carlo Q-value of (x, u) under a = K x + N(0, Sigma).

        The first action u is given (no policy noise at t = 0); the next state is
        sampled and valued with the closed-form V of the policy, so the estimate
        is unbiased for `discounted_q`.
        """
        x = np.ravel(np.asarray(x, dtype=float))
        u = np.ravel(np.asarray(u, dtype=float))
        if np.isscalar(Sigma):
            Sigma = float(Sigma) * np.eye(self.action_dim)
        Sigma = np.asarray(Sigma, dtype=float)

        self._check_discount(discount, "computeQFunction")
        P = self._computeP2(K, discount=discount)
        if not np.all(np.isfinite(P)):
            return -np.inf
        W = np.diag(self.noise_std ** 2)
        # constant part of V: -(tr(Sigma (R + discount B^T P B)) + discount tr(W P)) / (1 - discount)
        v_constant = (np.trace(Sigma @ (self.R + discount * self.B.T @ P @ self.B))
                      + discount * np.trace(W @ P)) / (1.0 - discount)
        noise = self.np_random.standard_normal((n_random_xn, self.state_dim)) @ self.sigma_noise.T
        nextstates = self.A @ x + self.B @ u + noise
        next_values = - np.einsum("ni,ij,nj->n", nextstates, P, nextstates) - v_constant
        cost = x @ self.Q @ x + u @ self.R @ u + 2.0 * (u @ self.M_mix @ x)
        return float(- cost + discount * next_values.mean())
