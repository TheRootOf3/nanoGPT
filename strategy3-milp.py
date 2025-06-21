import pulp


def balance_with_pulp(B, t, S, n_max=None):
    """
    Solve
        minimize max_i(n[i]*t[i]) - min_i(n[i]*t[i])
    subject to
        sum(B[i] * n[i]) == S
        0 <= n[i] <= n_max  (if provided)
        n[i] integer
    Returns: (n_sol, times, spread)
    """
    N = len(B)
    prob = pulp.LpProblem("Balance_Times", pulp.LpMinimize)

    # Upper‐bound for n[i]
    if n_max is None:
        n_bound = S // min(b for b in B if b > 0)
    else:
        n_bound = n_max

    n = pulp.LpVariable.dicts(
        "n", range(N), lowBound=0, upBound=n_bound, cat=pulp.LpInteger
    )
    M = pulp.LpVariable("M", lowBound=0)
    m = pulp.LpVariable("m", lowBound=0)

    prob += pulp.lpSum(B[i] * n[i] for i in range(N)) == S

    for i in range(N):
        prob += n[i] * t[i] <= M
        prob += n[i] * t[i] >= m

    prob += M - m

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=30)
    result = prob.solve(solver)
    if pulp.LpStatus[result] not in ("Optimal", "Feasible"):
        raise RuntimeError(f"No solution found: status {pulp.LpStatus[result]}")

    n_sol = [int(pulp.value(n[i])) for i in range(N)]
    times = [n_sol[i] * t[i] for i in range(N)]
    spread = pulp.value(M) - pulp.value(m)
    return n_sol, times, spread


if __name__ == "__main__":
    # Config-1
    # Bs = [32 for _ in range(1)] + [16 for _ in range(1)]
    # ts = [0.165 for _ in range(1)] + [0.129 for _ in range(1)]
    # S = 8192

    # Config-2
    Bs = [32 for _ in range(1)] + [16 for _ in range(2)] + [8 for _ in range(1)]
    ts = (
        [0.165 for _ in range(1)]
        + [0.129 for _ in range(2)]
        + [0.112 for _ in range(1)]
    )
    S = 16_384

    # Config-2 for SMHA
    Bs = [32 for _ in range(1)] + [16 for _ in range(2)] + [8 for _ in range(1)]
    ts = (
        [0.165 for _ in range(1)]
        + [0.120 for _ in range(2)]
        + [0.100 for _ in range(1)]
    )
    S = 16_384

    n_sol, times, spread = balance_with_pulp(Bs, ts, S)
    print("n_i     =", n_sol)
    print("n_i*t_i =", times)
    print("Spread  =", spread)
