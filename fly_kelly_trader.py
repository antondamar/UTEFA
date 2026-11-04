"""
fly_kelly_trader.py - QuantiFi 2026. One self-contained file: upload it to the portal or run
    python backtest/harness.py backtest/kelly/fly_kelly_trader.py

TWO ENGINES, NO SHARED PRODUCTS
1. Vouchers -> butterfly arbitrage (wiki/code/butterfly_arb_v2.py, unchanged). A fly bought
   below 0 cannot lose at expiry: it is (almost) a sure bet, and for a sure bet Kelly says bet
   as much as you can, so its size is set by the position limits, not by a Kelly fraction.
2. Every other product -> market maker sized by fractional Kelly (Thorp 2006):
       cap = min(limit, C * K * edge / variance)
   edge = realized 20-tick markout of our own fills, shrunk toward 0; no proven edge -> a 3-lot
   probe, then nothing. C = 0.25 (best out-of-sample in the sweep), K = 114.7 (day -2).
   Vouchers are excluded here: the sweep showed the quoter loses on them.
"""
import json
from math import gcd
from typing import Dict, List, Tuple

from datamodel import Order, TradingState

STRIKES = [6800, 7000, 7200, 7300, 7400, 7500, 7600, 7750, 7900, 8100]
SYM = {k: f"VOUCHER_{k}" for k in STRIKES}

PARAMS = {
    "position_limit": 50,        # per voucher (organizers' wiki)
    "min_entry_edge": 1.0,       # marginal fly must cost <= -this to buy
    "exit_value": 3.0,           # marginal fly must sell for >= this to unwind (trial-003 beat 0.0 on all 3 days)
    "max_open_per_triple": 25,
    "max_flies_per_tick": 25,    # per triple
    "repair": True,              # fix naked legs left by partial fills
}


def build_triples(strikes=STRIKES):
    out = []
    for i in range(len(strikes) - 2):
        a, b, c = strikes[i], strikes[i + 1], strikes[i + 2]
        g = gcd(c - b, b - a)
        na, nc = (c - b) // g, (b - a) // g
        out.append((a, b, c, na, na + nc, nc))
    return out


class Ladder:
    """Remaining book volume on one side, best price first, consumed as we add flies."""

    def __init__(self, levels):
        self.lv = [[p, v] for p, v in levels if v > 0]

    def cost(self, q):
        """Price of taking q more units, or None if the book is too thin."""
        tot, need = 0, q
        for p, v in self.lv:
            t = min(v, need)
            tot += t * p
            need -= t
            if need == 0:
                return tot
        return None

    def take(self, q):
        """Consume q units and return a list of (price, qty) slices."""
        out, need = [], q
        for lvl in self.lv:
            if need == 0:
                break
            t = min(lvl[1], need)
            if t:
                out.append((lvl[0], t))
                lvl[1] -= t
                need -= t
        self.lv = [l for l in self.lv if l[1] > 0]
        return out


class ButterflyArb:
    def __init__(self, params: Dict = None):
        self.p = dict(PARAMS, **(params or {}))
        self.triples = build_triples()

    def run(self, state: TradingState, st: Dict) -> Tuple[Dict[str, List[Order]], Dict]:
        p = self.p
        open_n = st.get("open", {})
        depths = {k: state.order_depths.get(SYM[k]) for k in STRIKES}
        if any(d is None or not d.buy_orders or not d.sell_orders for d in depths.values()):
            return {}, st

        asks = {k: Ladder(sorted((px, -v) for px, v in depths[k].sell_orders.items())) for k in STRIKES}
        bids = {k: Ladder(sorted(((px, v) for px, v in depths[k].buy_orders.items()), reverse=True))
                for k in STRIKES}
        pos = {k: state.position.get(SYM[k], 0) for k in STRIKES}
        hi, lo = dict(pos), dict(pos)          # worst-case long / short after our orders
        lim = p["position_limit"]
        orders: Dict[str, List[Order]] = {}

        def emit(k, slices, sign):
            for px, q in slices:
                orders.setdefault(SYM[k], []).append(Order(SYM[k], int(px), sign * int(q)))

        def buy(k, q):
            emit(k, asks[k].take(q), +1); hi[k] += q

        def sell(k, q):
            emit(k, bids[k].take(q), -1); lo[k] -= q

        # ---- 1) repair: make actual positions match what our open flies imply
        if p["repair"]:
            expected = {k: 0 for k in STRIKES}
            for (a, b, c, na, nb, nc) in self.triples:
                n = int(open_n.get(f"{a}_{b}_{c}", 0))
                expected[a] += na * n; expected[b] -= nb * n; expected[c] += nc * n
            for k in STRIKES:
                gap = expected[k] - pos[k]
                if gap > 0:
                    q = min(gap, lim - hi[k])
                    if q > 0 and asks[k].cost(q) is not None:
                        buy(k, q)
                elif gap < 0:
                    q = min(-gap, lim + lo[k])
                    if q > 0 and bids[k].cost(q) is not None:
                        sell(k, q)

        for (a, b, c, na, nb, nc) in self.triples:
            key = f"{a}_{b}_{c}"
            n_open = int(open_n.get(key, 0))

            # ---- 2) exit: sell flies one at a time while each still sells for >= exit_value
            done = 0
            while n_open > 0 and done < p["max_flies_per_tick"]:
                ra, rc, rb = bids[a].cost(na), bids[c].cost(nc), asks[b].cost(nb)
                if None in (ra, rc, rb) or ra + rc - rb < p["exit_value"]:
                    break
                if lo[a] - na < -lim or lo[c] - nc < -lim or hi[b] + nb > lim:
                    break
                sell(a, na); sell(c, nc); buy(b, nb)
                n_open -= 1; done += 1

            # ---- 3) entry: buy flies one at a time while each still costs <= -min_entry_edge
            done = 0
            while n_open < p["max_open_per_triple"] and done < p["max_flies_per_tick"]:
                ca, cc, cb = asks[a].cost(na), asks[c].cost(nc), bids[b].cost(nb)
                if None in (ca, cc, cb) or ca + cc - cb > -p["min_entry_edge"]:
                    break
                if hi[a] + na > lim or hi[c] + nc > lim or lo[b] - nb < -lim:
                    break
                buy(a, na); buy(c, nc); sell(b, nb)
                n_open += 1; done += 1

            open_n[key] = n_open

        # net buys against sells on the same voucher (a strike can be an outer leg of one fly
        # and the middle of the next): crossing ourselves would pay the spread twice.
        merged = {}
        for sym, lst in orders.items():
            buys = sorted([[o.price, o.quantity] for o in lst if o.quantity > 0])            # cheapest first
            sells = sorted([[o.price, -o.quantity] for o in lst if o.quantity < 0], reverse=True)  # richest first
            overlap = min(sum(q for _, q in buys), sum(q for _, q in sells))
            for side in (buys, sells):          # drop the overlap from the worst-priced end
                cut = overlap
                while cut > 0 and side:
                    t = min(side[-1][1], cut)
                    side[-1][1] -= t; cut -= t
                    if side[-1][1] == 0:
                        side.pop()
            agg = {}
            for px, q in buys:
                agg[(px, 1)] = agg.get((px, 1), 0) + q
            for px, q in sells:
                agg[(px, -1)] = agg.get((px, -1), 0) + q
            out = [Order(sym, px, sgn * q) for (px, sgn), q in agg.items() if q > 0]
            if out:
                merged[sym] = out
        st["open"] = open_n
        return merged, st




# ===================== Kelly market maker =====================
LIMITS = {
    "AMETHYST": 50, "TOFFEE_CAKE": 50, "PAPER_LANTERNS": 50, "TELLURIUM": 40,
    "JELLY": 50, "PEANUT_BUTTER": 50, "DRAGONFRUIT": 40, "GRANADILLA": 40,
    "MANGOSTEEN": 40, "FRUIT_BASKET": 40, "LYCHEE": 200,
    "WHITE_PEARLS": 20, "BLACK_PEARLS": 20, "GOLDEN_PEARLS": 20, "PINK_PEARLS": 20,
    "BLUE_PEARLS": 20,
}
C = 0.25        # Kelly fraction (best out-of-sample in our sweep)
K = 114.7       # scale, calibrated on day -2
SKEW = 2.0
A = 0.01          # EWMA weight (~100 ticks)
WARMUP = 50       # ticks before trading a product


H = 20         # markout horizon, ticks
N0 = 100.0    # prior units at zero edge (shrinkage)
PROBE = 3  # size used while edge is unproven
PROBE_UNITS = 300                              # stop probing after this many units
PROBE_ONLY = False


class KellyMM:
    """Edge = realized markout of our own fills H ticks later, shrunk toward 0:
         e_hat = sum(qty * markout) / (sum(qty) + N0)
    Kelly: no proven edge -> no bet (beyond a small probe while we learn)."""

    def run(self, state, mem):
        result = {}
        for prod, od in state.order_depths.items():
            if prod.startswith("VOUCHER"):
                continue          # vouchers belong to the butterfly arb
            if not od.buy_orders or not od.sell_orders:
                continue
            bb, ba = max(od.buy_orders), min(od.sell_orders)
            mid = (bb + ba) / 2
            m = mem.get(prod, {"mid": mid, "s2": 1.0, "n": 0, "pend": [],
                               "mq": 0.0, "q": 0.0})
            dm = mid - m["mid"]
            m["s2"] = (1 - A) * m["s2"] + A * dm * dm
            m["mid"], m["n"] = mid, m["n"] + 1
            for t in state.own_trades.get(prod, []):          # last tick's fills
                sgn = 1 if t.buyer == "SUBMISSION" else -1
                m["pend"].append([m["n"], t.price, sgn * t.quantity])
            keep = []
            for n0, px, qty in m["pend"]:                     # mature markouts
                if m["n"] - n0 >= H:
                    m["mq"] += qty * (mid - px)
                    m["q"] += abs(qty)
                else:
                    keep.append([n0, px, qty])
            m["pend"] = keep
            mem[prod] = m
            lim = LIMITS.get(prod, 50)
            if m["n"] < WARMUP:
                continue
            e_hat = m["mq"] / (m["q"] + N0)
            if PROBE_ONLY:
                cap = PROBE
            elif e_hat > 0:
                cap = max(PROBE if m["q"] < PROBE_UNITS else 0,
                          int(min(lim, C * K * e_hat / max(m["s2"], 1e-6))))
            else:
                cap = PROBE if m["q"] < PROBE_UNITS else 0
            pos = state.position.get(prod, 0)
            if cap <= 0 and pos == 0:
                continue
            capd = max(cap, 1)
            shift = -SKEW * pos / capd
            bid = min(bb + 1, int(mid + shift - 0.5))
            ask = max(ba - 1, int(mid + shift + 1.5))
            if bid >= ask:
                continue
            orders = []
            buy_q = min(max(0, cap - pos), lim - pos)
            sell_q = min(max(0, cap + pos), lim + pos)
            if pos < 0 and cap == 0:   # cap went to zero: unwind only
                buy_q = min(-pos, lim - pos)
            if pos > 0 and cap == 0:
                sell_q = min(pos, lim + pos)
            if buy_q > 0:
                orders.append(Order(prod, bid, buy_q))
            if sell_q > 0:
                orders.append(Order(prod, ask, -sell_q))
            if orders:
                result[prod] = orders
        return result, mem


class Trader:
    def __init__(self):
        self.fly = ButterflyArb()
        self.mm = KellyMM()

    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            saved = {}
        orders, saved["fly"] = self.fly.run(state, saved.get("fly", {}))
        mm_orders, saved["mm"] = self.mm.run(state, saved.get("mm", {}))
        orders.update(mm_orders)
        return orders, 0, json.dumps(saved)
