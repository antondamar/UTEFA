"""TOFFEE_CAKE v9: V5 + dynamic mean-reversion entry sizing (V8 Test B).

This file reproduces the winning V8 ablation configuration:
  * Adaptive entry threshold: OFF. Fixed executable 30-coin entry edge.
  * Dynamic entry size: ON. 12 / 18 / 24 units based on price deviation.
  * Partial exits: OFF. Retain V5's full exit near 5000.
  * Market making: unchanged from V5, with inventory-aware passive quotes.

Fixed fair value = 5000; hard absolute position limit = 50.
The strategy is calibrated only on historical training days, not validated OOS.
Requires the competition's datamodel.py with Order and TradingState.
"""

import json
from datamodel import Order, TradingState


class Trader:
    PRODUCT = "TOFFEE_CAKE"

    # Competition: TOFFEE_CAKE official maximum absolute inventory = 50.
    POSITION_LIMIT = 50
    FAIR_VALUE = 5000
    ENTRY_EDGE = 30         # Aggressive buy <= 4970; sell >= 5030.
    EXIT_EDGE = 5           # Long exit bid >= 4995; short exit ask <= 5005.
    MR_MAX_PER_TICK = 18    # Per-tick cap, even when position capacity is larger.

    # More conviction -> more inventory, rather than always jumping to 50.
    TIER_1_EDGE = 30
    TIER_2_EDGE = 40
    TIER_3_EDGE = 55
    TIER_1_TARGET = 20
    TIER_2_TARGET = 35
    TIER_3_TARGET = 50

    # V9 = V8 ablation Test B. Preserve these settings for that result.
    ENABLE_ADAPTIVE_ENTRY = False
    ENABLE_DYNAMIC_SIZE = True
    ENABLE_PARTIAL_EXITS = False

    # Volatility is sqrt of the existing tick-to-tick variance EWMA.
    # Baseline sigma=2 -> same 30-coin entry edge as V5.
    ENTRY_VOL_SENSITIVITY = 3.0
    ENTRY_EDGE_FLOOR = 26
    ENTRY_EDGE_CEILING = 36

    # Entry size varies by conviction relative to the fixed 30-coin threshold.
    ENTRY_SIZE_LOW = 12
    ENTRY_SIZE_MEDIUM = 18
    ENTRY_SIZE_HIGH = 24

    # At 4990 bid (long) / 5010 ask (short), sell/cover no more than
    # 40% of the peak signed inventory; leave 60% for the full V5 exit.
    PARTIAL_EXIT_EDGE = 10
    PARTIAL_HOLD_FRACTION = 0.60

    # Market maker: leave room for the mean-reversion strategy.
    ENABLE_MAKER = True
    MAKER_SOFT_LIMIT = 16
    MAKER_BASE_SIZE = 5
    MAKER_MAX_SIZE = 8
    MAKER_BAND = 22
    MAKER_MIN_BOOK_SPREAD = 9
    MAKER_MIN_EDGE = 2
    QUOTE_IMPROVEMENT = 1

    # Avellaneda-Stoikov-inspired reservation value / half-spread.
    GAMMA = 0.015
    GAMMA_INVENTORY_BOOST = 0.6
    HORIZON = 15.0
    FAIR_PULL = 0.30
    # Approximate (2/gamma)*ln(1+gamma/k) for gamma=.015, k=.18.
    FILL_SPREAD_COMPONENT = 10.67236102313819

    # Tick-to-tick variance EWMA, persisted via traderData.
    VOL_ALPHA = 0.06
    INITIAL_VARIANCE = 4.0
    MIN_VARIANCE = 0.25
    MAX_VARIANCE = 16.0
    MAX_TICK_MOVE = 20.0

    def _entry_threshold(self, variance):
        """Require a larger deviation when short-term volatility is high."""
        if not self.ENABLE_ADAPTIVE_ENTRY:
            return self.ENTRY_EDGE
        volatility = variance ** 0.5
        threshold = self.ENTRY_EDGE + int(round(
            self.ENTRY_VOL_SENSITIVITY * (volatility - 2.0)
        ))
        return max(self.ENTRY_EDGE_FLOOR,
                   min(self.ENTRY_EDGE_CEILING, threshold))

    def _exposure_target(self, price_edge, entry_threshold):
        """V5's tier limits, shifted to the adaptive entry threshold."""
        if price_edge >= entry_threshold + 25:
            return min(self.TIER_3_TARGET, self.POSITION_LIMIT)
        if price_edge >= entry_threshold + 10:
            return min(self.TIER_2_TARGET, self.POSITION_LIMIT)
        if price_edge >= entry_threshold:
            return min(self.TIER_1_TARGET, self.POSITION_LIMIT)
        return 0

    def _entry_order_size(self, price_edge, entry_threshold):
        """Reserve large aggressive orders for large executable price edges."""
        if not self.ENABLE_DYNAMIC_SIZE:
            return self.MR_MAX_PER_TICK
        if price_edge >= entry_threshold + 25:
            return self.ENTRY_SIZE_HIGH
        if price_edge >= entry_threshold + 10:
            return self.ENTRY_SIZE_MEDIUM
        return self.ENTRY_SIZE_LOW

    def _track_inventory_peak(self, position, timestamp, data):
        """Remember peak inventory until flat/sign change, even between ticks.

        Without this record, repeated 40%-exits could accidentally close the
        whole position across successive ticks instead of only one tranche.
        """
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
        """V5 mean reversion with entry-size tiers; fixed entry/exit levels."""
        orders = []
        entry_threshold = self._entry_threshold(variance)

        # Do not change the full-exit size. V5 used 18 units per tick.
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

        # Buy cheap if flat/long; if short, buy back some or all inventory.
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

        # Sell rich if flat/short; if long, unwind some or all inventory.
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
                    variance = self.INITIAL_VARIANCE  # New simulator day.
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

        # Larger inventory increases inventory risk aversion.
        exposure = min(1.0, abs(position) / float(self.MAKER_SOFT_LIMIT))
        gamma = self.GAMMA * (1.0 + self.GAMMA_INVENTORY_BOOST * exposure)
        risk = gamma * variance * self.HORIZON
        reservation = mid + self.FAIR_PULL * (self.FAIR_VALUE - mid) - risk * position
        half_spread = (risk + self.FILL_SPREAD_COMPONENT) / 2.0
        target_bid = reservation - half_spread
        target_ask = reservation + half_spread

        # Keep quote prices attractive relative to fixed fair value; allow
        # favorable closing transactions for inventory already held.
        max_buy = self.FAIR_VALUE + self.EXIT_EDGE if position < 0 else self.FAIR_VALUE - self.MAKER_MIN_EDGE
        min_sell = self.FAIR_VALUE - self.EXIT_EDGE if position > 0 else self.FAIR_VALUE + self.MAKER_MIN_EDGE

        # Makers don't expand positions past +/-16; can reduce any MR inventory.
        buy_capacity = max(0, min(self.MAKER_SOFT_LIMIT - position,
                                  self.POSITION_LIMIT - position))
        sell_capacity = max(0, min(self.MAKER_SOFT_LIMIT + position,
                                   self.POSITION_LIMIT + position))
        # Inventory skew: long -> smaller buy, larger sell; short -> opposite.
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
            # Priority to mean reversion; never mix maker orders on this tick.
            return {self.PRODUCT: reversion_orders}, 0, next_data

        maker_orders = self._maker_orders(best_bid, best_ask, position, variance)
        return {self.PRODUCT: maker_orders}, 0, next_data
