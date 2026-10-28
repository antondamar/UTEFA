"""PAPER_LANTERNS V8: Momentum with immediate, depth-aware execution.

A controlled *execution* experiment against the V2 (100-observation) momentum
baseline: V8 keeps the same signal and threshold but replaces passive entries
and five-unit reversal orders with marketable orders that consume visible
book liquidity. This can capture a trend sooner, at the cost of paying the
spread and potentially walking the book.

IMPORTANT:
- This is experimental. It has not been backtested on QuantiFi.
- The official PAPER_LANTERNS absolute position limit is 50.
- Quotes are only crossed when a full 100-observation momentum signal exists.
- Uses only public current/past data, no future information.
- Every submitted order counts toward the same directional position cap.
"""

import json
from datamodel import Order, TradingState

SYMBOL = "PAPER_LANTERNS"
POSITION_LIMIT = 50
MOMENTUM_LOOKBACK = 100
MIN_MOMENTUM_MOVE = 1.0
SPREAD_MULTIPLIER = 1.0

# If True, enter directly at visible opposite-side prices, up to 50 (or
# 100 on a full reversal) units, subject to current displayed liquidity.
# This is the single major execution change versus V2.
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
    """Create marketable orders at up to three *visible* price levels.

    Return only the quantities that the book currently displays; avoid
    quoting a huge quantity at a single level where there is little size.
    Buy orders have positive sizes; sell orders have negative sizes.
    """
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


class Trader:
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
            previous = {}  # Clear the history at a new trading day.

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
            # Buy enough to reach +50, including covering a short first.
            # For instance, starting from -50, maximum buy = 100 units.
            capacity = max(0, POSITION_LIMIT - pos)
            orders = sweep_visible_book(SYMBOL, book.sell_orders, capacity, True)
        elif momentum <= -threshold:
            # Sell enough to reach -50, including unwinding a long first.
            capacity = max(0, POSITION_LIMIT + pos)
            orders = sweep_visible_book(SYMBOL, book.buy_orders, capacity, False)

        return {SYMBOL: orders}, 0, saved
