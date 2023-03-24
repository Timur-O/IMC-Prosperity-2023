import numpy as np
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order, Position


def obv_sign(closing_diff: float):
    if closing_diff > 0:
        return 1
    elif closing_diff == 0:
        return 0
    else:
        return -1


class Trader:

    def __init__(self):
        self.limits: Dict[str, int] = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300
        }

        self.historicalBestAsk: Dict[str, List[int]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": []
        }

        self.historicalBestBid: Dict[str, List[int]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": []
        }

        self.historicalPrice: Dict[str, List[float]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": []
        }

        self.historicalOBV: Dict[str, List[float]] = {
            "PEARLS": [0],
            "BANANAS": [0],
            "COCONUTS": [0],
            "PINA_COLADAS": [0]
        }

        self.historicalPCtoCRatio: List[float] = []

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result = {}

        # Calculate and keep track of product values
        for product in state.order_depths.keys():
            # All Buy and Sell Orders for Product
            order_depth: OrderDepth = state.order_depths[product]
            num_sell = len(order_depth.sell_orders)
            num_buy = len(order_depth.buy_orders)

            # Keep Track of Historical Prices
            if num_sell > 0:
                sorted_asks = sorted(set(order_depth.sell_orders.keys()))
                self.historicalBestAsk[product].append(sorted_asks[0])

            if num_buy > 0:
                sorted_bids = sorted(set(order_depth.buy_orders.keys()), reverse=True)
                self.historicalBestBid[product].append(sorted_bids[0])

            if num_sell > 0 and num_buy > 0:
                avg_price = np.mean([self.historicalBestBid[product][-1], self.historicalBestAsk[product][-1]])
                self.historicalPrice[product].append(avg_price)

            # Keep Track of OBV
            current_volume = sum(order_depth.buy_orders.values()) + abs(sum(order_depth.sell_orders.values()))
            if len(self.historicalPrice[product]) > 1:
                price_diff = self.historicalPrice[product][-1] - self.historicalPrice[product][-2]
                newOBV = self.historicalOBV[product][-1] + current_volume * obv_sign(price_diff)
            else:
                price_diff = self.historicalPrice[product][-1]
                newOBV = current_volume * obv_sign(price_diff)

            self.historicalOBV[product].append(newOBV)

        # Pair Trading w/ Z-Score for Coconuts and Pina Coladas
        pina_colada_amount: int = 0
        coconut_amount: int = 0

        if ('COCONUTS' in state.order_depths.keys()) and ('PINA_COLADAS' in state.order_depths.keys()):
            time_to_consider: int = 20000
            z_score_entry_threshold: float = 2
            z_score_exit_threshold: float = 1

            # Get Current Positions
            try:
                coconut_position: Position = state.position['COCONUTS']
            except KeyError:
                coconut_position: Position = Position(0)

            try:
                pina_colada_position: Position = state.position['PINA_COLADAS']
            except KeyError:
                pina_colada_position: Position = Position(0)

            coconut_price: float = self.historicalPrice['COCONUTS'][-1]
            pina_colada_price: float = self.historicalPrice['PINA_COLADAS'][-1]

            coconut_limit: int = self.limits['COCONUTS']
            coconut_max_change_buy: int = coconut_limit - coconut_position
            coconut_max_change_sell: int = -coconut_limit - coconut_position  # -- => +
            pina_colada_limit: int = self.limits['PINA_COLADAS']
            pina_colada_max_change_buy: int = pina_colada_limit - pina_colada_position
            pina_colada_max_change_sell: int = -pina_colada_limit - pina_colada_position  # -- => +

            # Calculate and Keep Track of the Ratio
            current_ratio = pina_colada_price / coconut_price
            self.historicalPCtoCRatio.append(current_ratio)

            print("Z", current_ratio)

            if len(self.historicalPCtoCRatio) >= time_to_consider:
                historical_ratio_average = np.mean(self.historicalPCtoCRatio[-time_to_consider:])
                historical_ratio_std = np.std(self.historicalPCtoCRatio[-time_to_consider:])
                z_score_pc_c = (current_ratio - historical_ratio_average) / historical_ratio_std

                historical_pc_price = self.historicalPrice['PINA_COLADAS'][-time_to_consider:]
                historical_c_price = self.historicalPrice['COCONUTS'][-time_to_consider:]
                historical_mean_pc_price = np.mean(historical_pc_price)
                historical_mean_c_price = np.mean(historical_c_price)

                # Exit Position if Ratio is Normalized
                if -z_score_exit_threshold <= z_score_pc_c <= z_score_exit_threshold:
                    coconut_amount = -coconut_position
                    pina_colada_amount = -pina_colada_position
                # Enter Position if Ratio is Above Threshold
                elif z_score_pc_c > z_score_entry_threshold:
                    if pina_colada_price > historical_mean_pc_price:
                        # Pina Coladas Overpriced => Sell, Short
                        pina_colada_amount = max(pina_colada_max_change_sell, -round(coconut_max_change_buy * current_ratio))
                    elif coconut_price < historical_mean_c_price:
                        # Coconuts are Underpriced => Buy, Long
                        coconut_amount = min(coconut_max_change_buy, round(pina_colada_max_change_sell / current_ratio))
                # Enter Position if Ratio is Below Threshold
                elif z_score_pc_c < -z_score_entry_threshold:
                    # Pina Coladas are underpriced => Buy or Coconuts are overpriced => Sell
                    if pina_colada_price < historical_mean_pc_price:
                        # Pina Coladas Underpriced => Buy
                        pina_colada_amount = min(pina_colada_max_change_buy, round(coconut_max_change_sell * current_ratio))
                    elif coconut_price > historical_mean_c_price:
                        # Coconuts are Overpriced => Sell
                        coconut_amount = max(coconut_max_change_sell, -round(pina_colada_max_change_buy / current_ratio))

        # Iterate over all the available products contained in the order depths
        for product in state.order_depths.keys():
            # All Buy and Sell Orders for Product
            order_depth: OrderDepth = state.order_depths[product]
            num_sell = len(order_depth.sell_orders)
            num_buy = len(order_depth.buy_orders)

            # Get Current Position
            try:
                position: Position = state.position[product]
            except KeyError:
                position: Position = Position(0)

            # Calculate Current Limits for Product
            limit: int = self.limits[product]
            max_change_buy = limit - position
            max_change_sell = -limit - position  # -- => +

            sorted_asks = sorted(set(order_depth.sell_orders.keys()))
            sorted_bids = sorted(set(order_depth.buy_orders.keys()), reverse=True)

            # historicalBestAsk = self.historicalBestAsk[product]
            # historicalBestBid = self.historicalBestAsk[product]
            historicalPrice = self.historicalPrice[product]
            historicalOBV = self.historicalOBV[product]

            # Initialize the list of Orders to be sent
            orders: List[Order] = []

            # Simple Arbitrage for Pearls
            if product == 'PEARLS':
                # Pearls are worth ~10,000 in general
                acceptable_price_pearls = 10000.00

                # If any SELL orders and position limits allow buying
                if (num_sell > 0) and (position < limit):
                    order_counter = 0
                    while max_change_buy > 0:
                        best_ask = sorted_asks[order_counter]
                        best_ask_volume = max(order_depth.sell_orders[best_ask], -max_change_buy)

                        # If the lowest ask is less than the fair value
                        if best_ask < acceptable_price_pearls:
                            # Create a buy order to buy these cheap pearls
                            print("BUY", str(-best_ask_volume) + " PEARLS", best_ask)
                            orders.append(Order(product, best_ask, -best_ask_volume))
                            # Increment Counter
                            order_counter += 1
                            max_change_buy += best_ask_volume
                        else:
                            break

                # If any BUY orders and position limits allow selling
                if (num_buy > 0) and (position > -limit):
                    order_counter = 0
                    while max_change_sell < 0:
                        best_bid = sorted_bids[order_counter]
                        best_bid_volume = min(order_depth.buy_orders[best_bid], -max_change_sell)

                        # If the highest bid is more than the fair value
                        if best_bid > acceptable_price_pearls:
                            print("SELL", str(best_bid_volume) + " PEARLS", best_bid)
                            orders.append(Order(product, best_bid, -best_bid_volume))
                            # Increment Counter
                            order_counter += 1
                            max_change_sell += best_bid_volume
                        else:
                            break
            # Simple Price Direction Indication via Orders for Bananas
            elif product == 'BANANAS':
                history_length: int = 30000

                if (len(order_depth.buy_orders) > 0) and \
                   (len(order_depth.sell_orders) > 0) and \
                   (len(historicalOBV) >= history_length):
                    # Imbalance Ratio
                    imbalance_ratio = sum(order_depth.buy_orders.values()) / abs(sum(order_depth.sell_orders.values()))
                    imbalance_threshold = 1

                    # Gradient of OBV
                    obv_gradient = np.gradient(historicalOBV)[-1]
                    obv_average = np.mean(historicalOBV[-history_length])

                    if (imbalance_ratio > imbalance_threshold) and \
                       (obv_gradient > obv_average) and \
                       (max_change_buy > 0):
                        # Imbalance > 1 => Price Up, Current OBV Gradient > OBV Average => BUY
                        best_ask = min(order_depth.sell_orders.keys())
                        best_ask_volume = max(order_depth.sell_orders[best_ask], -max_change_buy)
                        print("BUY", str(-best_ask_volume) + " BANANAS", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))
                        # Increment Counter
                        max_change_buy += best_ask_volume

                    if (imbalance_ratio < imbalance_threshold) and \
                       (obv_gradient < obv_average) and \
                       (max_change_sell < 0):
                        # Imbalance < 1 => Price Down, Current OBV Gradient < OBV Average => SELL
                        best_bid = max(order_depth.buy_orders.keys())
                        best_bid_volume = min(order_depth.buy_orders[best_bid], -max_change_sell)
                        print("SELL", str(best_bid_volume) + " BANANAS", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                        # Increment Counter
                        max_change_sell += best_bid_volume
            # Pairs Trading - Executing Trades Based on Calculations Above
            elif (product == 'COCONUTS') and (coconut_amount != 0):
                if coconut_amount > 0:
                    # BUY
                    print("BUY", str(coconut_amount) + " COCONUTS", sorted_asks[-1])
                    orders.append(Order(product, sorted_asks[-1], coconut_amount))
                elif coconut_amount < 0:
                    # SELL
                    print("SELL", str(-coconut_amount) + " COCONUTS", sorted_bids[-1])
                    orders.append(Order(product, sorted_bids[-1], coconut_amount))
            # Pairs Trading - Executing Trades Based on Calculations Above
            elif (product == 'PINA_COLADAS') and (pina_colada_amount != 0):
                if pina_colada_amount > 0:
                    # BUY
                    print("BUY", str(pina_colada_amount) + " PINA_COLADAS", sorted_asks[-1])
                    orders.append(Order(product, sorted_asks[-1], pina_colada_amount))
                elif pina_colada_amount < 0:
                    # SELL
                    print("SELL", str(-pina_colada_amount) + " PINA_COLADAS", sorted_bids[-1])
                    orders.append(Order(product, sorted_bids[-1], pina_colada_amount))
            # Simple Bollinger Band Mean Reversion for Coconuts
            elif product == 'COCONUTS':
                K = 2
                N = 40000
                N_2 = N * 2

                moving_average = 0
                overall_direction = 0
                standard_deviation = 0

                if len(historicalPrice) >= N_2:
                    moving_average = np.mean(historicalPrice[-N_2:])
                    curr_prices = historicalPrice[-N:]
                    overall_direction = np.gradient(historicalPrice[-N:])[-1]
                    curr_prices_minus_mean = np.subtract(curr_prices, moving_average)
                    standard_deviation = np.std(curr_prices_minus_mean)

                # If any SELL orders and position limits allow buying
                if (num_sell > 0) and (position < limit) and (overall_direction > 0):
                    lower_band = moving_average - (K * standard_deviation)

                    best_ask = sorted_asks[0]
                    best_ask_volume = max(order_depth.sell_orders[best_ask], -max_change_buy)

                    # If the lowest ask is less than the fair value
                    if best_ask < lower_band:
                        # Create a buy order to buy these cheap pearls
                        print("BUY", str(-best_ask_volume) + " COCONUTS", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))
                        # Increment Counter
                        max_change_buy += best_ask_volume

                # If any BUY orders and position limits allow selling
                if (num_buy > 0) and (position > -limit) and (overall_direction < 0):
                    upper_band = moving_average + (K * standard_deviation)

                    best_bid = sorted_bids[0]
                    best_bid_volume = min(order_depth.buy_orders[best_bid], -max_change_sell)

                    # If the highest bid is more than the fair value
                    if best_bid > upper_band:
                        print("SELL", str(best_bid_volume) + " COCONUTS", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                        # Increment Counter
                        max_change_sell += best_bid_volume
            # Simple Bollinger Band Mean Reversion for Pina Coladas
            elif product == 'PINA_COLADAS':
                K = 2
                N = 50000
                N_2 = N * 2

                moving_average = 0
                overall_direction = 0
                standard_deviation = 0

                if len(historicalPrice) >= N_2:
                    moving_average = np.mean(historicalPrice[-N_2:])
                    curr_prices = historicalPrice[-N:]
                    overall_direction = np.gradient(historicalPrice[-N:])[-1]
                    curr_prices_minus_mean = np.subtract(curr_prices, moving_average)
                    standard_deviation = np.std(curr_prices_minus_mean)

                # If any SELL orders and position limits allow buying
                if (num_sell > 0) and (position < limit) and (overall_direction > 0):
                    lower_band = moving_average - (K * standard_deviation)

                    best_ask = sorted_asks[0]
                    best_ask_volume = max(order_depth.sell_orders[best_ask], -max_change_buy)

                    # If the lowest ask is less than the fair value
                    if best_ask < lower_band:
                        # Create a buy order to buy these cheap pearls
                        print("BUY", str(-best_ask_volume) + " PINA COLADAS", best_ask)
                        orders.append(Order(product, best_ask, -best_ask_volume))
                        # Increment Counter
                        max_change_buy += best_ask_volume

                # If any BUY orders and position limits allow selling
                if (num_buy > 0) and (position > -limit) and (overall_direction < 0):
                    upper_band = moving_average + (K * standard_deviation)

                    best_bid = sorted_bids[0]
                    best_bid_volume = min(order_depth.buy_orders[best_bid], -max_change_sell)

                    # If the highest bid is more than the fair value
                    if best_bid > upper_band:
                        print("SELL", str(best_bid_volume) + " PINA COLADAS", best_bid)
                        orders.append(Order(product, best_bid, -best_bid_volume))
                        # Increment Counter
                        max_change_sell += best_bid_volume
            # Default => Do nothing
            else:
                continue

            # Add all the above orders to the result dict
            result[product] = orders

        print("Position:", state.position)

        # Return the dict of orders
        return result
