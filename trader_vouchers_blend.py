"""
QuantiFi 2026 - BLENDED voucher trader: BS-EMA fair value + quadratic-smile fair value.
Built Oct 28, 2026 from strategies/vouchers-bs/trader_vouchers_bs.py (execution, skew,
delta hedge unchanged) and raw/code/vouchers_trader.py (smile fit).

ONE book, ONE position per product (limit 50 per voucher, 200 LYCHEE), so the two models
never fight over the limit. Each tick:
  fair_bs    = BS(S, K, EMA of strike's own implied total vol)      (slow, per strike)
  fair_smile = BS(S, K, T, quadratic smile fitted across strikes)   (fast, cross-section)
  fair       = W_BS * fair_bs + (1 - W_BS) * fair_smile
  gap        = |fair_bs - fair_smile|  -> take/make edges widened by GAP_K * gap
W_BS = 1, GAP_K = 0 reproduces the BS-EMA trader exactly.
"""
import json
import math
from typing import Dict, List

from datamodel import Order, OrderDepth, TradingState

STRIKES = [6800, 7000, 7200, 7300, 7400, 7500, 7600, 7750, 7900, 8100]
UNDERLYING = "LYCHEE"
VOUCHER_LIMIT = 50
LYCHEE_LIMIT = 200

PARAMS = {
    "ema_span": 800,       # ticks of memory for each strike's implied total vol
    "warmup": 50,          # ticks to observe before trading (EMA needs a start)
    "take_edge": 4.0,      # coins beyond fair value to cross the spread
    "make_edge": 3.0,      # minimum coins between a resting quote and fair value
    "skew": 3.0,           # coins fair value moves per full position (50 vouchers)
    "max_pos": 50,         # voucher position cap (<= 50)
    "hedge": True,         # delta hedge with LYCHEE
    "hedge_band": 40,      # cross the LYCHEE spread only when |position - target| > band;
                           # inside the band we hedge with passive quotes
    "w_bs": 0.5,           # weight on the BS-EMA fair value (1 - w on the smile fit)
    "gap_k": 0.0,          # extra edge per coin of model disagreement
    "smile_deg": 2,        # 2 = quadratic smile (v1), 3 = cubic
    "smile_vega": False,   # weight the smile fit by each strike's vega
    "expiry_day": 6,       # smile fit needs T: 8 days left at start of day -2
    "first_day": -2,
    "delta_skew": 10.0,     # coins of fair value per 100 net deltas, times the strike's delta:
                           # when the book is long LYCHEE exposure, every voucher is valued a
                           # bit lower (more so high-delta ones), so the voucher trades
                           # themselves pull the book back toward delta-neutral
}

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def bs_call(S: float, K: float, nu: float) -> float:
    """Black-Scholes call with r = 0; nu = sigma * sqrt(T)."""
    if nu <= 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * nu * nu) / nu
    return S * norm_cdf(d1) - K * norm_cdf(d1 - nu)


def bs_delta(S: float, K: float, nu: float) -> float:
    if nu <= 1e-9:
        return 1.0 if S > K else 0.0
    return norm_cdf((math.log(S / K) + 0.5 * nu * nu) / nu)


def implied_nu(C: float, S: float, K: float, guess: float):
    """Total implied vol from a call price. Newton from the guess, bisection fallback."""
    if C <= max(S - K, 0.0) + 0.5 or C >= S:   # too close to intrinsic: no information
        return None
    nu = guess if 0.005 < guess < 1.0 else 0.06
    for _ in range(8):
        d1 = (math.log(S / K) + 0.5 * nu * nu) / nu
        price = S * norm_cdf(d1) - K * norm_cdf(d1 - nu)
        vega = S * INV_SQRT_2PI * math.exp(-0.5 * d1 * d1)
        diff = price - C
        if abs(diff) < 1e-4:
            return nu
        if vega < 1e-6:
            break
        nu_new = nu - diff / vega
        if not (1e-4 < nu_new < 2.0):
            break
        nu = nu_new
    lo, hi = 1e-4, 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, mid) > C:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def mid_of(od: OrderDepth):
    if not od.buy_orders or not od.sell_orders:
        return None
    return (max(od.buy_orders) + min(od.sell_orders)) / 2.0


def bs_call_T(S, K, T, vol):
    return bs_call(S, K, vol * math.sqrt(T))


def fit_quadratic(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    s = [sum(x ** k for x in xs) for k in range(5)]
    t = [sum(y * x ** k for x, y in zip(xs, ys)) for k in range(3)]
    A = [[s[0], s[1], s[2]], [s[1], s[2], s[3]], [s[2], s[3], s[4]]]

    def det(M):
        return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
    D = det(A)
    if abs(D) < 1e-12:
        return None
    out = []
    for i in range(3):
        M = [r[:] for r in A]
        for r in range(3):
            M[r][i] = t[r]
        out.append(det(M) / D)
    return out


def fit_poly(xs, ys, ws, deg):
    """Weighted least squares polynomial, Gaussian elimination on normal equations."""
    k = deg + 1
    if len(xs) < k + 1:
        return None
    A = [[sum(w * x ** (i + j) for x, w in zip(xs, ws)) for j in range(k)] for i in range(k)]
    b = [sum(w * y * x ** i for x, y, w in zip(xs, ys, ws)) for i in range(k)]
    for c in range(k):
        piv = max(range(c, k), key=lambda r: abs(A[r][c]))
        if abs(A[piv][c]) < 1e-14:
            return None
        A[c], A[piv] = A[piv], A[c]
        b[c], b[piv] = b[piv], b[c]
        for r in range(c + 1, k):
            f_ = A[r][c] / A[c][c]
            for j in range(c, k):
                A[r][j] -= f_ * A[c][j]
            b[r] -= f_ * b[c]
    out = [0.0] * k
    for i in range(k - 1, -1, -1):
        out[i] = (b[i] - sum(A[i][j] * out[j] for j in range(i + 1, k))) / A[i][i]
    return out


def smile_fairs(state, S, T, deg=2, vega_w=False):
    """Quadratic smile in m = ln(K/S)/sqrt(T), fitted to this tick's mids (as v1)."""
    pts = {}
    for K in STRIKES:
        od = state.order_depths.get(f"VOUCHER_{K}")
        if od is None or not od.buy_orders or not od.sell_orders:
            continue
        C = mid_of(od)
        nu = implied_nu(C, S, K, 0.06)
        if nu is None:
            continue
        d1 = (math.log(S / K) + 0.5 * nu * nu) / nu
        vega = S * INV_SQRT_2PI * math.exp(-0.5 * d1 * d1) if vega_w else 1.0
        pts[K] = (math.log(K / S) / math.sqrt(T), nu / math.sqrt(T), vega)
    coef = fit_poly([v[0] for v in pts.values()], [v[1] for v in pts.values()],
                    [v[2] for v in pts.values()], deg)
    if coef is None:
        return {}
    out = {}
    for K in STRIKES:
        m = math.log(K / S) / math.sqrt(T)
        vol = max(sum(c * m ** i for i, c in enumerate(coef)), 1e-4)
        out[K] = bs_call_T(S, K, T, vol)
    return out


class Trader:
    def __init__(self, params=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)

    # ------------------------------------------------------------------ vouchers
    def voucher_orders(self, sym: str, od: OrderDepth, pos: int, fair: float,
                       delta: float, book_delta: float, gap: float = 0.0) -> List[Order]:
        p = dict(self.p)
        p["take_edge"] = p["take_edge"] + p["gap_k"] * gap
        p["make_edge"] = p["make_edge"] + p["gap_k"] * gap
        orders: List[Order] = []
        cap = min(p["max_pos"], VOUCHER_LIMIT)
        # inventory skew: long -> value it less (easier to sell, harder to buy)
        fv = fair - p["skew"] * pos / cap
        # delta skew: long LYCHEE exposure overall -> value high-delta vouchers less
        fv -= p["delta_skew"] * delta * book_delta / 100.0
        buy_room = cap - pos
        sell_room = cap + pos

        # TAKE
        for ask in sorted(od.sell_orders):
            if buy_room <= 0 or ask > fv - p["take_edge"]:
                break
            q = min(-od.sell_orders[ask], buy_room)
            orders.append(Order(sym, ask, q))
            buy_room -= q
        for bid in sorted(od.buy_orders, reverse=True):
            if sell_room <= 0 or bid < fv + p["take_edge"]:
                break
            q = min(od.buy_orders[bid], sell_room)
            orders.append(Order(sym, bid, -q))
            sell_room -= q

        # MAKE: one tick inside the touch, never closer to fair than make_edge
        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        bid_px = min(best_bid + 1, math.floor(fv - p["make_edge"]))
        ask_px = max(best_ask - 1, math.ceil(fv + p["make_edge"]))
        if buy_room > 0 and bid_px < best_ask:
            orders.append(Order(sym, bid_px, buy_room))
        if sell_room > 0 and ask_px > best_bid:
            orders.append(Order(sym, ask_px, -sell_room))
        return orders

    # ------------------------------------------------------------------ hedge
    def hedge_orders(self, od: OrderDepth, pos: int, target: int) -> List[Order]:
        orders: List[Order] = []
        target = max(-LYCHEE_LIMIT, min(LYCHEE_LIMIT, target))
        gap = target - pos
        band = self.p["hedge_band"]
        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        if gap > 0:   # need to buy LYCHEE
            room = min(gap, LYCHEE_LIMIT - pos)
            # outside the band: cross the spread, but only back to the band's edge
            urgent = min(room, gap - band) if gap > band else 0
            if urgent > 0:
                q = min(urgent, -od.sell_orders[best_ask])
                orders.append(Order(UNDERLYING, best_ask, q))
                room -= q
            # the rest: a passive bid one tick inside the spread (earns spread, may not fill)
            if room > 0 and best_bid + 1 < best_ask:
                orders.append(Order(UNDERLYING, best_bid + 1, room))
        elif gap < 0:
            room = min(-gap, LYCHEE_LIMIT + pos)
            urgent = min(room, -gap - band) if -gap > band else 0
            if urgent > 0:
                q = min(urgent, od.buy_orders[best_bid])
                orders.append(Order(UNDERLYING, best_bid, -q))
                room -= q
            if room > 0 and best_ask - 1 > best_bid:
                orders.append(Order(UNDERLYING, best_ask - 1, -room))
        return orders

    # ------------------------------------------------------------------ main
    def run(self, state: TradingState):
        p = self.p
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        ema: Dict[str, float] = mem.get("ema", {})
        n = mem.get("n", 0) + 1
        day = mem.get("day", p["first_day"])
        last_ts = mem.get("ts", -1)
        if state.timestamp < last_ts:
            day += 1
        alpha = 2.0 / (p["ema_span"] + 1.0)

        result: Dict[str, List[Order]] = {}
        und = state.order_depths.get(UNDERLYING)
        S = mid_of(und) if und is not None else None
        if S is None:
            return result, 0, json.dumps({"ema": ema, "n": n, "day": day, "ts": state.timestamp})
        T = max(p["expiry_day"] - day - state.timestamp / 1_000_000, 1e-3)
        sf = smile_fairs(state, S, T, p["smile_deg"], p["smile_vega"]) if p["w_bs"] < 1.0 or p["gap_k"] > 0 else {}

        # pass 1: fair value and delta of every strike from the PAST smile (no look-ahead)
        info = {}
        net_delta = 0.0
        for K in STRIKES:
            sym = f"VOUCHER_{K}"
            pos = state.position.get(sym, 0)
            prev = ema.get(str(K))
            if prev is not None:
                info[K] = (bs_call(S, K, prev), bs_delta(S, K, prev))
                net_delta += pos * info[K][1]
            elif pos:
                net_delta += pos * (1.0 if S > K else 0.0)
        book_delta = net_delta + state.position.get(UNDERLYING, 0)

        # pass 2: trade each voucher, then update its smile point with today's price
        for K in STRIKES:
            sym = f"VOUCHER_{K}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            C = mid_of(od)
            if K in info and n > p["warmup"] and C is not None:
                fair_bs, delta = info[K]
                fs = sf.get(K, fair_bs)
                fair = p["w_bs"] * fair_bs + (1.0 - p["w_bs"]) * fs
                gap = abs(fair_bs - fs)
                orders = self.voucher_orders(sym, od, state.position.get(sym, 0), fair,
                                             delta, book_delta, gap)
                if orders:
                    result[sym] = orders
            prev = ema.get(str(K))
            nu_obs = implied_nu(C, S, K, prev if prev else 0.06) if C is not None else None
            if nu_obs is not None:
                # early in the run the EMA is a plain running mean (weight 1/n), so one
                # noisy first price does not anchor the smile for hundreds of ticks
                a = max(alpha, 1.0 / n)
                ema[str(K)] = nu_obs if prev is None else prev + a * (nu_obs - prev)

        if p["hedge"]:
            target = -int(round(net_delta))
            h = self.hedge_orders(und, state.position.get(UNDERLYING, 0), target)
            if h:
                result[UNDERLYING] = h

        return result, 0, json.dumps({"ema": {k: round(v, 7) for k, v in ema.items()}, "n": n,
                                  "day": day, "ts": state.timestamp})
