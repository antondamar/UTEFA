from __future__ import annotations

import copy
import json
from datamodel import Order, TradingState

import json
import math
from typing import Dict, List

from datamodel import Order, OrderDepth, TradingState

STRIKES = [6800, 7000, 7200, 7300, 7400, 7500, 7600, 7750, 7900, 8100]
UNDERLYING = "LYCHEE"
VOUCHER_LIMIT = 50
LYCHEE_LIMIT = 200

PARAMS = {
    "ema_span": 1000,
    "warmup": 50,
    "take_edge": 4.0,
    "make_edge": 3.0,
    "skew": 3.0,
    "max_pos": 50,
    "hedge": True,
    "hedge_band": 40,

    "w_bs": 0.5,
    "gap_k": 0.0,
    "smile_deg": 2,
    "smile_vega": False,
    "expiry_day": 6,
    "first_day": -2,
    "delta_skew": 10.0,

}

SQRT2 = math.sqrt(2.0)
INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call(S: float, K: float, nu: float) -> float:

    if nu <= 1e-9:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * nu * nu) / nu
    return S * norm_cdf(d1) - K * norm_cdf(d1 - nu)

def bs_delta(S: float, K: float, nu: float) -> float:
    if nu <= 1e-9:
        return 1.0 if S > K else 0.0
    return norm_cdf((math.log(S / K) + 0.5 * nu * nu) / nu)

def implied_nu(C: float, S: float, K: float, guess: float):

    if C <= max(S - K, 0.0) + 0.5 or C >= S:
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

    pts = {}
    for K in STRIKES:
        od = state.order_depths.get(voucher_symbol(state, K))
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

def voucher_symbol(state: TradingState, strike: int) -> str:

    current = f"LYCHEE_VOUCHERS_{strike}"
    legacy = f"VOUCHER_{strike}"
    if current in state.order_depths:
        return current
    if legacy in state.order_depths:
        return legacy
    if current in state.position:
        return current
    if legacy in state.position:
        return legacy
    return current

class VoucherStrategy:
    def __init__(self, params=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)

    def voucher_orders(self, sym: str, od: OrderDepth, pos: int, fair: float,
                       delta: float, book_delta: float, gap: float = 0.0) -> List[Order]:
        p = dict(self.p)
        p["take_edge"] = p["take_edge"] + p["gap_k"] * gap
        p["make_edge"] = p["make_edge"] + p["gap_k"] * gap
        orders: List[Order] = []
        cap = min(p["max_pos"], VOUCHER_LIMIT)

        fv = fair - p["skew"] * pos / cap

        fv -= p["delta_skew"] * delta * book_delta / 100.0
        buy_room = cap - pos
        sell_room = cap + pos

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

        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        bid_px = min(best_bid + 1, math.floor(fv - p["make_edge"]))
        ask_px = max(best_ask - 1, math.ceil(fv + p["make_edge"]))
        if buy_room > 0 and bid_px < best_ask:
            orders.append(Order(sym, bid_px, buy_room))
        if sell_room > 0 and ask_px > best_bid:
            orders.append(Order(sym, ask_px, -sell_room))
        return orders

    def hedge_orders(self, od: OrderDepth, pos: int, target: int) -> List[Order]:
        orders: List[Order] = []
        target = max(-LYCHEE_LIMIT, min(LYCHEE_LIMIT, target))
        gap = target - pos
        band = self.p["hedge_band"]
        best_bid, best_ask = max(od.buy_orders), min(od.sell_orders)
        if gap > 0:
            room = min(gap, LYCHEE_LIMIT - pos)

            urgent = min(room, gap - band) if gap > band else 0
            if urgent > 0:
                q = min(urgent, -od.sell_orders[best_ask])
                orders.append(Order(UNDERLYING, best_ask, q))
                room -= q

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

        info = {}
        net_delta = 0.0
        for K in STRIKES:
            sym = voucher_symbol(state, K)
            pos = state.position.get(sym, 0)
            prev = ema.get(str(K))
            if prev is not None:
                info[K] = (bs_call(S, K, prev), bs_delta(S, K, prev))
                net_delta += pos * info[K][1]
            elif pos:
                net_delta += pos * (1.0 if S > K else 0.0)
        book_delta = net_delta + state.position.get(UNDERLYING, 0)

        for K in STRIKES:
            sym = voucher_symbol(state, K)
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

                a = max(alpha, 1.0 / n)
                ema[str(K)] = nu_obs if prev is None else prev + a * (nu_obs - prev)

        if p["hedge"]:
            target = -int(round(net_delta))
            h = self.hedge_orders(und, state.position.get(UNDERLYING, 0), target)
            if h:
                result[UNDERLYING] = h

        return result, 0, json.dumps({"ema": {k: round(v, 7) for k, v in ema.items()}, "n": n,
                                  "day": day, "ts": state.timestamp})

import math

from datamodel import Order, OrderDepth, TradingState

PEARLS = (
    "WHITE_PEARLS",
    "BLACK_PEARLS",
    "GOLDEN_PEARLS",
    "PINK_PEARLS",
    "BLUE_PEARLS",
)
PEARLS_POSITION_LIMIT = 20
PEARLS_EDGE = 3
PEARLS_SIZE = 5
SUM_BAND = 20
FAIR_SUM = 50_000

class PearlStrategy:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        mids: dict[str, float] = {}
        for product in PEARLS:
            mid = self.mid(state.order_depths.get(product))
            if mid is None:
                return {}, 0, ""
            mids[product] = mid

        positions = {product: state.position.get(product, 0) for product in PEARLS}
        net = sum(positions.values())
        orders: dict[str, list[Order]] = {}
        for product in PEARLS:
            others = sum(mid for name, mid in mids.items() if name != product)
            fair = FAIR_SUM - others
            quoted = self.quote(
                product,
                state.order_depths[product],
                fair,
                positions[product],
                allow_bid=net < SUM_BAND,
                allow_ask=net > -SUM_BAND,
            )
            if quoted:
                orders[product] = quoted
        return orders, 0, ""

    def quote(
        self,
        product: str,
        book: OrderDepth,
        fair: float,
        position: int,
        allow_bid: bool,
        allow_ask: bool,
    ) -> list[Order]:
        best_bid = max(book.buy_orders) if book.buy_orders else None
        best_ask = min(book.sell_orders) if book.sell_orders else None
        orders: list[Order] = []

        buy_quantity = min(PEARLS_SIZE, PEARLS_POSITION_LIMIT - position)
        if allow_bid and best_bid is not None and buy_quantity > 0:
            bid_price = min(best_bid, math.floor(fair - PEARLS_EDGE))
            if best_ask is None or bid_price < best_ask:
                orders.append(Order(product, bid_price, buy_quantity))

        sell_quantity = min(PEARLS_SIZE, PEARLS_POSITION_LIMIT + position)
        if allow_ask and best_ask is not None and sell_quantity > 0:
            ask_price = max(best_ask, math.ceil(fair + PEARLS_EDGE))
            if best_bid is None or ask_price > best_bid:
                orders.append(Order(product, ask_price, -sell_quantity))

        return orders

    def mid(self, book: OrderDepth | None) -> float | None:
        if book is None or not book.buy_orders or not book.sell_orders:
            return None
        return (max(book.buy_orders) + min(book.sell_orders)) / 2

import json

from datamodel import Order, OrderDepth, TradingState

BASKET = "FRUIT_BASKET"
DRAGON = "DRAGONFRUIT"
GRAN = "GRANADILLA"
MANGO = "MANGOSTEEN"
PRODUCTS = (BASKET, DRAGON, GRAN, MANGO)

DRAGON_BETA = 1.79
GRAN_BETA = 2.59
MANGO_BETA = 0.93

DRAGON_HEDGE = 2
GRAN_HEDGE = 3
MANGO_HEDGE = 1
HEDGES = {DRAGON: DRAGON_HEDGE, GRAN: GRAN_HEDGE, MANGO: MANGO_HEDGE}

FRUIT_POSITION_LIMIT = 40

MAX_BASKETS = FRUIT_POSITION_LIMIT // GRAN_HEDGE
FRUIT_EDGE = 60
FRUIT_SIZE = 1
HEDGE_BAND = 2
EMA_SPAN = 1200
EMA_ALPHA = 2 / (EMA_SPAN + 1)

class FruitStrategy:
    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        mids = {product: self.mid(state.order_depths.get(product)) for product in PRODUCTS}
        if any(mid is None for mid in mids.values()):
            return {}, 0, state.traderData

        synthetic = (
            DRAGON_BETA * mids[DRAGON]
            + GRAN_BETA * mids[GRAN]
            + MANGO_BETA * mids[MANGO]
        )
        residual = mids[BASKET] - synthetic
        previous = self.load_ema(state.traderData)
        if previous is None:
            return {}, 0, self.dump_ema(residual)

        fair = synthetic + previous
        ema = previous + EMA_ALPHA * (residual - previous)
        saved = self.dump_ema(ema)

        basket_book = state.order_depths[BASKET]
        basket_pos = state.position.get(BASKET, 0)
        basket_orders = self.take_basket(basket_book, fair, basket_pos)
        if basket_orders:
            basket_pos += basket_orders[0].quantity

        orders: dict[str, list[Order]] = {}
        if basket_orders:
            orders[BASKET] = basket_orders
        for product, hedge in HEDGES.items():
            fruit_orders = self.hedge_fruit(
                product,
                state.order_depths[product],
                target=-hedge * basket_pos,
                position=state.position.get(product, 0),
                hedge=hedge,
            )
            if fruit_orders:
                orders[product] = fruit_orders
        return orders, 0, saved

    def take_basket(self, book: OrderDepth, fair: float, position: int) -> list[Order]:
        if not book.buy_orders or not book.sell_orders:
            return []
        ask = min(book.sell_orders)
        bid = max(book.buy_orders)
        buy_room = min(FRUIT_POSITION_LIMIT - position, MAX_BASKETS - position)
        sell_room = min(FRUIT_POSITION_LIMIT + position, MAX_BASKETS + position)

        if bid - fair > FRUIT_EDGE and buy_room > 0:
            quantity = min(FRUIT_SIZE, buy_room, abs(book.sell_orders[ask]))
            if quantity > 0:
                return [Order(BASKET, ask, quantity)]
        if fair - ask > FRUIT_EDGE and sell_room > 0:
            quantity = min(FRUIT_SIZE, sell_room, abs(book.buy_orders[bid]))
            if quantity > 0:
                return [Order(BASKET, bid, -quantity)]
        return []

    def hedge_fruit(
        self,
        product: str,
        book: OrderDepth,
        target: int,
        position: int,
        hedge: int,
    ) -> list[Order]:
        diff = target - position
        if abs(diff) < HEDGE_BAND:
            return []
        cap = FRUIT_SIZE * hedge
        if diff > 0:
            if not book.sell_orders:
                return []
            ask = min(book.sell_orders)
            quantity = min(diff, cap, FRUIT_POSITION_LIMIT - position, abs(book.sell_orders[ask]))
            if quantity <= 0:
                return []
            return [Order(product, ask, quantity)]
        if not book.buy_orders:
            return []
        bid = max(book.buy_orders)
        quantity = min(-diff, cap, FRUIT_POSITION_LIMIT + position, abs(book.buy_orders[bid]))
        if quantity <= 0:
            return []
        return [Order(product, bid, -quantity)]

    def mid(self, book: OrderDepth | None) -> float | None:
        if book is None or not book.buy_orders or not book.sell_orders:
            return None
        return (max(book.buy_orders) + min(book.sell_orders)) / 2

    def load_ema(self, trader_data: str) -> float | None:
        if not trader_data:
            return None
        try:
            payload = json.loads(trader_data)
        except json.JSONDecodeError:
            return None
        value = payload.get("ema")
        if value is None:
            return None
        return float(value)

    def dump_ema(self, ema: float) -> str:
        return json.dumps({"ema": ema})

import json
from datamodel import Order, TradingState

class ToffeeStrategy:
    PRODUCT = "TOFFEE_CAKE"

    POSITION_LIMIT = 50
    FAIR_VALUE = 5000
    ENTRY_EDGE = 30
    EXIT_EDGE = 5
    MR_MAX_PER_TICK = 18

    TIER_1_EDGE = 30
    TIER_2_EDGE = 40
    TIER_3_EDGE = 55
    TIER_1_TARGET = 20
    TIER_2_TARGET = 35
    TIER_3_TARGET = 50

    ENABLE_ADAPTIVE_ENTRY = False
    ENABLE_DYNAMIC_SIZE = True
    ENABLE_PARTIAL_EXITS = False

    ENTRY_VOL_SENSITIVITY = 3.0
    ENTRY_EDGE_FLOOR = 26
    ENTRY_EDGE_CEILING = 36

    ENTRY_SIZE_LOW = 12
    ENTRY_SIZE_MEDIUM = 18
    ENTRY_SIZE_HIGH = 24

    PARTIAL_EXIT_EDGE = 10
    PARTIAL_HOLD_FRACTION = 0.60

    ENABLE_MAKER = True
    MAKER_SOFT_LIMIT = 16
    MAKER_BASE_SIZE = 5
    MAKER_MAX_SIZE = 8
    MAKER_BAND = 22
    MAKER_MIN_BOOK_SPREAD = 9
    MAKER_MIN_EDGE = 2
    QUOTE_IMPROVEMENT = 1

    GAMMA = 0.015
    GAMMA_INVENTORY_BOOST = 0.6
    HORIZON = 15.0
    FAIR_PULL = 0.30

    FILL_SPREAD_COMPONENT = 10.67236102313819

    VOL_ALPHA = 0.06
    INITIAL_VARIANCE = 4.0
    MIN_VARIANCE = 0.25
    MAX_VARIANCE = 16.0
    MAX_TICK_MOVE = 20.0

    def _entry_threshold(self, variance):

        if not self.ENABLE_ADAPTIVE_ENTRY:
            return self.ENTRY_EDGE
        volatility = variance ** 0.5
        threshold = self.ENTRY_EDGE + int(round(
            self.ENTRY_VOL_SENSITIVITY * (volatility - 2.0)
        ))
        return max(self.ENTRY_EDGE_FLOOR,
                   min(self.ENTRY_EDGE_CEILING, threshold))

    def _exposure_target(self, price_edge, entry_threshold):

        if price_edge >= entry_threshold + 25:
            return min(self.TIER_3_TARGET, self.POSITION_LIMIT)
        if price_edge >= entry_threshold + 10:
            return min(self.TIER_2_TARGET, self.POSITION_LIMIT)
        if price_edge >= entry_threshold:
            return min(self.TIER_1_TARGET, self.POSITION_LIMIT)
        return 0

    def _entry_order_size(self, price_edge, entry_threshold):

        if not self.ENABLE_DYNAMIC_SIZE:
            return self.MR_MAX_PER_TICK
        if price_edge >= entry_threshold + 25:
            return self.ENTRY_SIZE_HIGH
        if price_edge >= entry_threshold + 10:
            return self.ENTRY_SIZE_MEDIUM
        return self.ENTRY_SIZE_LOW

    def _track_inventory_peak(self, position, timestamp, data):

        previous = data.get("toffee_v8")
        if not isinstance(previous, dict):
            previous = {}
        sign = 1 if position > 0 else (-1 if position < 0 else 0)
        try:
            old_timestamp = int(previous.get("timestamp", -1))
            old_sign = int(previous.get("sign", 0))
            old_peak = max(0, min(self.POSITION_LIMIT,
                                  int(previous.get("peak", 0))))
        except (TypeError, ValueError):
            old_timestamp, old_sign, old_peak = -1, 0, 0
        if sign == 0:
            peak = 0
        elif sign != old_sign or timestamp < old_timestamp:
            peak = abs(position)
        else:
            peak = max(abs(position), old_peak)
        data["toffee_v8"] = {
            "timestamp": int(timestamp), "sign": sign, "peak": peak
        }
        return peak

    def _mean_reversion_orders(self, book, position, variance, peak):

        orders = []
        entry_threshold = self._entry_threshold(variance)

        buy_budget = self.MR_MAX_PER_TICK
        sell_budget = self.MR_MAX_PER_TICK
        if position >= 0 and book.sell_orders:
            best_ask = min(book.sell_orders)
            best_edge = self.FAIR_VALUE - best_ask
            if best_edge >= entry_threshold:
                buy_budget = self._entry_order_size(best_edge, entry_threshold)
        if position <= 0 and book.buy_orders:
            best_bid = max(book.buy_orders)
            best_edge = best_bid - self.FAIR_VALUE
            if best_edge >= entry_threshold:
                sell_budget = self._entry_order_size(best_edge, entry_threshold)

        planned_buys = 0
        planned_sells = 0
        retained = int(peak * self.PARTIAL_HOLD_FRACTION + 0.5)

        for price, volume in sorted(book.sell_orders.items()):
            if buy_budget <= 0:
                break
            if position < 0:
                if price <= self.FAIR_VALUE + self.EXIT_EDGE:
                    capacity = max(0, -position - planned_buys)
                elif (self.ENABLE_PARTIAL_EXITS and
                      price <= self.FAIR_VALUE + self.PARTIAL_EXIT_EDGE):
                    capacity = max(0, -position - retained - planned_buys)
                else:
                    break
            else:
                edge = self.FAIR_VALUE - price
                if edge < entry_threshold:
                    break
                target = self._exposure_target(edge, entry_threshold)
                capacity = max(0, target - position - planned_buys)
            qty = min(buy_budget, capacity, abs(volume),
                      max(0, self.POSITION_LIMIT - position - planned_buys))
            if qty > 0:
                orders.append(Order(self.PRODUCT, int(price), int(qty)))
                planned_buys += qty
                buy_budget -= qty

        for price, volume in sorted(book.buy_orders.items(), reverse=True):
            if sell_budget <= 0:
                break
            if position > 0:
                if price >= self.FAIR_VALUE - self.EXIT_EDGE:
                    capacity = max(0, position - planned_sells)
                elif (self.ENABLE_PARTIAL_EXITS and
                      price >= self.FAIR_VALUE - self.PARTIAL_EXIT_EDGE):
                    capacity = max(0, position - retained - planned_sells)
                else:
                    break
            else:
                edge = price - self.FAIR_VALUE
                if edge < entry_threshold:
                    break
                target = self._exposure_target(edge, entry_threshold)
                capacity = max(0, target + position - planned_sells)
            qty = min(sell_budget, capacity, abs(volume),
                      max(0, self.POSITION_LIMIT + position - planned_sells))
            if qty > 0:
                orders.append(Order(self.PRODUCT, int(price), -int(qty)))
                planned_sells += qty
                sell_budget -= qty
        return orders

    def _update_variance(self, timestamp, mid, data):
        saved = data.get("toffee_v5")
        if not isinstance(saved, dict):
            saved = {}
        try:
            variance = float(saved.get("variance", self.INITIAL_VARIANCE))
            if not self.MIN_VARIANCE <= variance <= self.MAX_VARIANCE:
                variance = self.INITIAL_VARIANCE
        except (ValueError, TypeError):
            variance = self.INITIAL_VARIANCE

        prev_mid = saved.get("mid")
        prev_t = saved.get("timestamp")
        if prev_mid is not None and prev_t is not None:
            try:
                if timestamp < int(prev_t):
                    variance = self.INITIAL_VARIANCE
                elif timestamp > int(prev_t):
                    move = mid - float(prev_mid)
                    move = max(-self.MAX_TICK_MOVE, min(self.MAX_TICK_MOVE, move))
                    variance = (1 - self.VOL_ALPHA) * variance + self.VOL_ALPHA * move * move
            except (TypeError, ValueError):
                variance = self.INITIAL_VARIANCE

        variance = max(self.MIN_VARIANCE, min(self.MAX_VARIANCE, variance))
        data["toffee_v5"] = {
            "timestamp": timestamp,
            "mid": mid,
            "variance": variance,
        }
        return variance

    def _maker_orders(self, best_bid, best_ask, position, variance):
        if not self.ENABLE_MAKER:
            return []
        spread = best_ask - best_bid
        mid = (best_bid + best_ask) / 2.0
        if spread < self.MAKER_MIN_BOOK_SPREAD:
            return []
        if abs(mid - self.FAIR_VALUE) > self.MAKER_BAND:
            return []

        exposure = min(1.0, abs(position) / float(self.MAKER_SOFT_LIMIT))
        gamma = self.GAMMA * (1.0 + self.GAMMA_INVENTORY_BOOST * exposure)
        risk = gamma * variance * self.HORIZON
        reservation = mid + self.FAIR_PULL * (self.FAIR_VALUE - mid) - risk * position
        half_spread = (risk + self.FILL_SPREAD_COMPONENT) / 2.0
        target_bid = reservation - half_spread
        target_ask = reservation + half_spread

        max_buy = self.FAIR_VALUE + self.EXIT_EDGE if position < 0 else self.FAIR_VALUE - self.MAKER_MIN_EDGE
        min_sell = self.FAIR_VALUE - self.EXIT_EDGE if position > 0 else self.FAIR_VALUE + self.MAKER_MIN_EDGE

        buy_capacity = max(0, min(self.MAKER_SOFT_LIMIT - position,
                                  self.POSITION_LIMIT - position))
        sell_capacity = max(0, min(self.MAKER_SOFT_LIMIT + position,
                                   self.POSITION_LIMIT + position))

        size_skew = round(3 * position / float(self.MAKER_SOFT_LIMIT))
        buy_size = min(buy_capacity, max(1, min(self.MAKER_MAX_SIZE,
                                                self.MAKER_BASE_SIZE - size_skew)))
        sell_size = min(sell_capacity, max(1, min(self.MAKER_MAX_SIZE,
                                                 self.MAKER_BASE_SIZE + size_skew)))

        orders = []
        if buy_size > 0:
            price = best_bid
            improved = best_bid + self.QUOTE_IMPROVEMENT
            if improved < best_ask and improved <= target_bid and improved <= max_buy:
                price = improved
            if price <= target_bid and price <= max_buy and price < best_ask:
                orders.append(Order(self.PRODUCT, int(price), int(buy_size)))

        if sell_size > 0:
            price = best_ask
            improved = best_ask - self.QUOTE_IMPROVEMENT
            if improved > best_bid and improved >= target_ask and improved >= min_sell:
                price = improved
            if price >= target_ask and price >= min_sell and price > best_bid:
                orders.append(Order(self.PRODUCT, int(price), -int(sell_size)))
        return orders

    def run(self, state: TradingState):
        book = state.order_depths.get(self.PRODUCT)
        if book is None or not book.buy_orders or not book.sell_orders:
            return {self.PRODUCT: []}, 0, state.traderData or ""

        position = int(state.position.get(self.PRODUCT, 0))
        best_bid = max(book.buy_orders)
        best_ask = min(book.sell_orders)
        if best_bid >= best_ask:
            return {self.PRODUCT: []}, 0, state.traderData or ""

        try:
            data = json.loads(state.traderData or "{}")
            if not isinstance(data, dict):
                data = {}
        except (ValueError, TypeError):
            data = {}

        mid = (best_bid + best_ask) / 2.0
        variance = self._update_variance(state.timestamp, mid, data)
        peak = self._track_inventory_peak(position, state.timestamp, data)
        next_data = json.dumps(data)

        reversion_orders = self._mean_reversion_orders(book, position, variance, peak)
        if len(reversion_orders) > 0:

            return {self.PRODUCT: reversion_orders}, 0, next_data

        maker_orders = self._maker_orders(best_bid, best_ask, position, variance)
        return {self.PRODUCT: maker_orders}, 0, next_data

import json
from datamodel import Order, TradingState

SYMBOL = "PAPER_LANTERNS"
PAPER_POSITION_LIMIT = 50
MOMENTUM_LOOKBACK = 100
MIN_MOMENTUM_MOVE = 1.0
SPREAD_MULTIPLIER = 1.0

MAX_LEVELS_TO_SWEEP = 3

def load_memory(raw):
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
        return decoded if isinstance(decoded, dict) else {}
    except (TypeError, ValueError):
        return {}

def sweep_visible_book(symbol, opposite_orders, amount, is_buy):

    result = []
    remaining = max(0, int(amount))
    if remaining == 0:
        return result
    levels = sorted(opposite_orders, reverse=not is_buy)
    for price in levels[:MAX_LEVELS_TO_SWEEP]:
        if remaining <= 0:
            break
        available = abs(int(opposite_orders[price]))
        if available <= 0:
            continue
        quantity = min(remaining, available)
        result.append(Order(symbol, int(price), int(quantity if is_buy else -quantity)))
        remaining -= quantity
    return result

class PaperLanternsStrategy:
    def run(self, state: TradingState):
        memory = load_memory(state.traderData)
        orders = []
        book = state.order_depths.get(SYMBOL)
        if book is None or not book.buy_orders or not book.sell_orders:
            return {SYMBOL: orders}, 0, json.dumps(memory, separators=(",", ":"))

        best_bid = int(max(book.buy_orders))
        best_ask = int(min(book.sell_orders))
        if best_ask <= best_bid:
            return {SYMBOL: orders}, 0, json.dumps(memory, separators=(",", ":"))

        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        timestamp = int(state.timestamp)
        previous = memory.get("paper_lanterns_momentum", {})
        if not isinstance(previous, dict):
            previous = {}
        old_ts = previous.get("last_timestamp")
        if not isinstance(old_ts, int) or timestamp <= old_ts:
            previous = {}

        mids = previous.get("mids", [])
        if not isinstance(mids, list):
            mids = []
        mids = mids[-MOMENTUM_LOOKBACK:] + [mid]
        memory["paper_lanterns_momentum"] = {
            "last_timestamp": timestamp,
            "mids": mids,
        }
        saved = json.dumps(memory, separators=(",", ":"))
        if len(mids) <= MOMENTUM_LOOKBACK:
            return {SYMBOL: orders}, 0, saved

        momentum = mid - mids[0]
        threshold = max(MIN_MOMENTUM_MOVE, SPREAD_MULTIPLIER * spread)
        pos = int(state.position.get(SYMBOL, 0))

        if momentum >= threshold:

            capacity = max(0, PAPER_POSITION_LIMIT - pos)
            orders = sweep_visible_book(SYMBOL, book.sell_orders, capacity, True)
        elif momentum <= -threshold:

            capacity = max(0, PAPER_POSITION_LIMIT + pos)
            orders = sweep_visible_book(SYMBOL, book.buy_orders, capacity, False)

        return {SYMBOL: orders}, 0, saved

from math import gcd
from typing import Tuple

VOUCHER_MODE = "blended"

KELLY_ONLY_PRODUCTS = frozenset(("AMETHYST", "TELLURIUM"))

FLY_STRIKES = [6800, 7000, 7200, 7300, 7400, 7500, 7600, 7750, 7900, 8100]

FLY_PARAMS = {
    "position_limit": 50,
    "min_entry_edge": 1.0,
    "exit_value": 3.0,
    "max_open_per_triple": 25,
    "max_flies_per_tick": 25,
    "repair": True,
}

def build_triples(strikes=FLY_STRIKES):
    out = []
    for i in range(len(strikes) - 2):
        a, b, c = strikes[i], strikes[i + 1], strikes[i + 2]
        g = gcd(c - b, b - a)
        na, nc = (c - b) // g, (b - a) // g
        out.append((a, b, c, na, na + nc, nc))
    return out

class Ladder:

    def __init__(self, levels):
        self.lv = [[p, v] for p, v in levels if v > 0]

    def cost(self, q):

        tot, need = 0, q
        for p, v in self.lv:
            t = min(v, need)
            tot += t * p
            need -= t
            if need == 0:
                return tot
        return None

    def take(self, q):

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
        self.p = dict(FLY_PARAMS, **(params or {}))
        self.triples = build_triples()

    def run(self, state: TradingState, st: Dict) -> Tuple[Dict[str, List[Order]], Dict]:
        p = self.p
        open_n = st.get("open", {})
        depths = {k: state.order_depths.get(voucher_symbol(state, k)) for k in FLY_STRIKES}
        if any(d is None or not d.buy_orders or not d.sell_orders for d in depths.values()):
            return {}, st

        asks = {k: Ladder(sorted((px, -v) for px, v in depths[k].sell_orders.items())) for k in FLY_STRIKES}
        bids = {k: Ladder(sorted(((px, v) for px, v in depths[k].buy_orders.items()), reverse=True))
                for k in FLY_STRIKES}
        pos = {k: state.position.get(voucher_symbol(state, k), 0) for k in FLY_STRIKES}
        hi, lo = dict(pos), dict(pos)
        lim = p["position_limit"]
        orders: Dict[str, List[Order]] = {}

        def emit(k, slices, sign):
            for px, q in slices:
                orders.setdefault(voucher_symbol(state, k), []).append(Order(voucher_symbol(state, k), int(px), sign * int(q)))

        def buy(k, q):
            emit(k, asks[k].take(q), +1); hi[k] += q

        def sell(k, q):
            emit(k, bids[k].take(q), -1); lo[k] -= q

        if p["repair"]:
            expected = {k: 0 for k in FLY_STRIKES}
            for (a, b, c, na, nb, nc) in self.triples:
                n = int(open_n.get(f"{a}_{b}_{c}", 0))
                expected[a] += na * n; expected[b] -= nb * n; expected[c] += nc * n
            for k in FLY_STRIKES:
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

            done = 0
            while n_open > 0 and done < p["max_flies_per_tick"]:
                ra, rc, rb = bids[a].cost(na), bids[c].cost(nc), asks[b].cost(nb)
                if None in (ra, rc, rb) or ra + rc - rb < p["exit_value"]:
                    break
                if lo[a] - na < -lim or lo[c] - nc < -lim or hi[b] + nb > lim:
                    break
                sell(a, na); sell(c, nc); buy(b, nb)
                n_open -= 1; done += 1

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

        merged = {}
        for sym, lst in orders.items():
            buys = sorted([[o.price, o.quantity] for o in lst if o.quantity > 0])
            sells = sorted([[o.price, -o.quantity] for o in lst if o.quantity < 0], reverse=True)
            overlap = min(sum(q for _, q in buys), sum(q for _, q in sells))
            for side in (buys, sells):
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

KELLY_C = 0.25
KELLY_K = 114.7
KELLY_SKEW = 2.0
KELLY_A = 0.01
KELLY_WARMUP = 50

KELLY_H = 20
KELLY_N0 = 100.0
KELLY_PROBE = 3
KELLY_PROBE_UNITS = 300
KELLY_PROBE_ONLY = False

class KellyMM:

    def run(self, state, mem):
        result = {}
        for prod, od in state.order_depths.items():
            if prod not in KELLY_ONLY_PRODUCTS:
                continue
            if not od.buy_orders or not od.sell_orders:
                continue
            bb, ba = max(od.buy_orders), min(od.sell_orders)
            mid = (bb + ba) / 2
            m = mem.get(prod, {"mid": mid, "s2": 1.0, "n": 0, "pend": [],
                               "mq": 0.0, "q": 0.0})
            dm = mid - m["mid"]
            m["s2"] = (1 - KELLY_A) * m["s2"] + KELLY_A * dm * dm
            m["mid"], m["n"] = mid, m["n"] + 1
            for t in state.own_trades.get(prod, []):
                sgn = 1 if t.buyer == "SUBMISSION" else -1
                m["pend"].append([m["n"], t.price, sgn * t.quantity])
            keep = []
            for n0, px, qty in m["pend"]:
                if m["n"] - n0 >= KELLY_H:
                    m["mq"] += qty * (mid - px)
                    m["q"] += abs(qty)
                else:
                    keep.append([n0, px, qty])
            m["pend"] = keep
            mem[prod] = m
            lim = OFFICIAL_LIMITS[prod]
            if m["n"] < KELLY_WARMUP:
                continue
            e_hat = m["mq"] / (m["q"] + KELLY_N0)
            if KELLY_PROBE_ONLY:
                cap = KELLY_PROBE
            elif e_hat > 0:
                cap = max(KELLY_PROBE if m["q"] < KELLY_PROBE_UNITS else 0,
                          int(min(lim, KELLY_C * KELLY_K * e_hat / max(m["s2"], 1e-6))))
            else:
                cap = KELLY_PROBE if m["q"] < KELLY_PROBE_UNITS else 0
            pos = state.position.get(prod, 0)
            if cap <= 0 and pos == 0:
                continue
            capd = max(cap, 1)
            shift = -KELLY_SKEW * pos / capd
            bid = min(bb + 1, int(mid + shift - 0.5))
            ask = max(ba - 1, int(mid + shift + 1.5))
            if bid >= ask:
                continue
            orders = []
            buy_q = min(max(0, cap - pos), lim - pos)
            sell_q = min(max(0, cap + pos), lim + pos)
            if pos < 0 and cap == 0:
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

class ButterflyAdapter:

    def __init__(self):
        self.engine = ButterflyArb()

    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(saved, dict):
                saved = {}
        except (TypeError, ValueError):
            saved = {}
        orders, saved = self.engine.run(state, saved)
        return orders, 0, json.dumps(saved, separators=(",", ":"))

class KellyAdapter:

    def __init__(self):
        self.engine = KellyMM()

    def run(self, state: TradingState):
        try:
            saved = json.loads(state.traderData) if state.traderData else {}
            if not isinstance(saved, dict):
                saved = {}
        except (TypeError, ValueError):
            saved = {}
        orders, saved = self.engine.run(state, saved)
        return orders, 0, json.dumps(saved, separators=(",", ":"))

OFFICIAL_LIMITS = {
    "AMETHYST": 50,
    "PAPER_LANTERNS": 50,
    "TOFFEE_CAKE": 50,
    "TELLURIUM": 40,
    "JELLY": 50,
    "PEANUT_BUTTER": 50,
    "DRAGONFRUIT": 40,
    "GRANADILLA": 40,
    "MANGOSTEEN": 40,
    "FRUIT_BASKET": 40,
    "LYCHEE": 200,
    "WHITE_PEARLS": 20,
    "BLACK_PEARLS": 20,
    "GOLDEN_PEARLS": 20,
    "PINK_PEARLS": 20,
    "BLUE_PEARLS": 20,
}
for _strike in (6800, 7000, 7200, 7300, 7400, 7500, 7600, 7750, 7900, 8100):
    OFFICIAL_LIMITS[f"LYCHEE_VOUCHERS_{_strike}"] = 50
    OFFICIAL_LIMITS[f"VOUCHER_{_strike}"] = 50

class Trader:

    def __init__(self):
        if VOUCHER_MODE not in ("blended", "butterfly"):
            raise ValueError("VOUCHER_MODE must be 'blended' or 'butterfly'")

        self._strategies = {
            "pearls": PearlStrategy(),
            "fruit": FruitStrategy(),
            "toffee": ToffeeStrategy(),
            "paper": PaperLanternsStrategy(),
        }
        if VOUCHER_MODE == "blended":
            self._strategies["vouchers"] = VoucherStrategy()
        else:
            self._strategies["butterfly"] = ButterflyAdapter()
        self._strategies["kelly"] = KellyAdapter()

    @staticmethod
    def _load_state(raw: str) -> dict:
        try:
            data = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            return {}
        if not isinstance(data, dict) or data.get("version") != 1:
            return {}
        return data.get("components", {}) if isinstance(data.get("components"), dict) else {}

    @staticmethod
    def _cap_orders(symbol: str, orders: list[Order], position: int) -> list[Order]:

        if symbol not in OFFICIAL_LIMITS:
            raise ValueError(f"Missing official position limit for {symbol}")
        limit = OFFICIAL_LIMITS[symbol]
        buy_remaining = max(0, limit - position)
        sell_remaining = max(0, limit + position)
        clean = []
        for order in orders:
            if order.symbol != symbol:
                raise ValueError(f"{symbol} strategy submitted order for {order.symbol}")
            qty = int(order.quantity)
            if qty > 0:
                accepted = min(qty, buy_remaining)
                buy_remaining -= accepted
            elif qty < 0:
                accepted = -min(-qty, sell_remaining)
                sell_remaining += accepted
            else:
                continue
            if accepted:
                clean.append(Order(symbol, int(order.price), accepted))
        return clean

    def run(self, state: TradingState):
        storage = self._load_state(state.traderData)
        next_storage = {}
        combined: dict[str, list[Order]] = {}
        conversions = 0

        for name, strategy in self._strategies.items():

            scoped_state = copy.copy(state)
            scoped_state.traderData = storage.get(name, "")
            local_orders, local_conversions, local_memory = strategy.run(scoped_state)
            next_storage[name] = local_memory or ""
            conversions += int(local_conversions or 0)

            for symbol, instructions in (local_orders or {}).items():
                if symbol in combined:
                    raise ValueError(f"Two strategies both own {symbol}")
                safe_orders = self._cap_orders(
                    symbol, instructions, int(state.position.get(symbol, 0))
                )
                if safe_orders:
                    combined[symbol] = safe_orders

        packed = json.dumps(
            {"version": 1, "components": next_storage}, separators=(",", ":")
        )
        return combined, conversions, packed
