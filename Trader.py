import numpy as np
from typing import Dict, List, Tuple

from datamodel import OrderDepth, TradingState, Order, Position, Time


def obv_sign(closing_diff: float) -> int:
    if closing_diff > 0:
        return 1
    elif closing_diff == 0:
        return 0
    else:
        return -1


def get_position(product: str, state: TradingState) -> Position:
    try:
        return state.position[product]
    except KeyError:
        return Position(0)


class Trader:

    # Initialize limits and historical dictionaries containing arrays
    def __init__(self):
        self.limits: Dict[str, int] = {
            "PEARLS": 20,
            "BANANAS": 20,
            "COCONUTS": 600,
            "PINA_COLADAS": 300,
            "BERRIES": 250,
            "DIVING_GEAR": 50,
            "BAGUETTE": 150,
            "DIP": 300,
            "UKULELE": 70,
            "PICNIC_BASKET": 70
        }

        self.historicalOBV: Dict[str, List[float]] = {
            "PEARLS": [0],
            "BANANAS": [0],
            "COCONUTS": [0],
            "PINA_COLADAS": [0],
            "BERRIES": [0],
            "DIVING_GEAR": [0],
            "BAGUETTE": [0],
            "DIP": [0],
            "UKULELE": [0],
            "PICNIC_BASKET": [0]
        }

        self.lastUsedOBVPrice: Dict[str, float] = {
            "PEARLS": 0.0,
            "BANANAS": 0.0,
            "COCONUTS": 0.0,
            "PINA_COLADAS": 0.0,
            "BERRIES": 0.0,
            "DIVING_GEAR": 0.0,
            "BAGUETTE": 0.0,
            "DIP": 0.0,
            "UKULELE": 0.0,
            "PICNIC_BASKET": 0.0
        }

        self.historicalObservations: Dict[str, List[int]] = {
            'DOLPHIN_SIGHTINGS': []
        }

        self.historicalBestAsk: Dict[str, List[int]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": [],
            "BERRIES": [],
            "DIVING_GEAR": [],
            "BAGUETTE": [],
            "DIP": [],
            "UKULELE": [],
            "PICNIC_BASKET": []
        }

        self.historicalBestBid: Dict[str, List[int]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": [],
            "BERRIES": [],
            "DIVING_GEAR": [],
            "BAGUETTE": [],
            "DIP": [],
            "UKULELE": [],
            "PICNIC_BASKET": []
        }

        self.historicalPrice: Dict[str, List[float]] = {
            "PEARLS": [],
            "BANANAS": [],
            "COCONUTS": [],
            "PINA_COLADAS": [],
            "BERRIES": [],
            "DIVING_GEAR": [],
            "BAGUETTE": [],
            "DIP": [],
            "UKULELE": [],
            "PICNIC_BASKET": []
        }

        self.historicalPairRatios: Dict[str, List[float]] = {}

    # Update historical records
    def update_history(self, product: str, state: TradingState) -> None:
        # All Buy and Sell Orders for Product
        order_depth: OrderDepth = state.order_depths[product]
        num_sell: int = len(order_depth.sell_orders)
        num_buy: int = len(order_depth.buy_orders)

        # Keep Track of Historical Prices
        if num_sell > 0:
            sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
            self.historicalBestAsk[product].append(sorted_asks[0])

        if num_buy > 0:
            sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)
            self.historicalBestBid[product].append(sorted_bids[0])

        if num_sell > 0 and num_buy > 0:
            avg_price: float = np.mean([self.historicalBestBid[product][-1], self.historicalBestAsk[product][-1]])
            self.historicalPrice[product].append(avg_price)

        # Keep Track of OBV
        current_volume: int = sum(order_depth.buy_orders.values()) + abs(sum(order_depth.sell_orders.values()))
        if len(self.historicalPrice[product]) > 1:
            if self.historicalPrice[product][-1] != self.lastUsedOBVPrice[product]:
                price_diff: float = self.historicalPrice[product][-1] - self.lastUsedOBVPrice[product]
                newOBV: float = self.historicalOBV[product][-1] + current_volume * obv_sign(price_diff)
                self.lastUsedOBVPrice[product] = self.historicalPrice[product][-1]
                self.historicalOBV[product].append(newOBV)
        elif len(self.historicalPrice[product]) == 1:
            price_diff: float = self.historicalPrice[product][-1]
            newOBV: float = current_volume * obv_sign(price_diff)
            self.historicalOBV[product].append(newOBV)

    # Calculate amount to buy/sell using arbitrage
    def calculate_arbitrage_amount(self, product: str, value: float, state: TradingState) -> int:
        # Get all values for the product
        order_depth: OrderDepth = state.order_depths[product]
        num_sell: int = len(order_depth.sell_orders)
        num_buy: int = len(order_depth.buy_orders)
        position: int = get_position(product, state)

        # Get and calculate limits
        limit: int = self.limits[product]
        max_buy: int = limit - position
        max_sell: int = -limit - position

        # Get general info
        sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
        sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)

        # Initialize result
        amount: int = 0

        if (num_sell > 0) and (position < limit):
            order_counter: int = 0
            while (max_buy > 0) and (order_counter < num_sell):
                # BUY
                curr_ask: int = sorted_asks[order_counter]
                curr_volume: int = max(order_depth.sell_orders[curr_ask], -max_buy)

                # If the lowest ask is less than the fair value
                if curr_ask < value:
                    amount += -curr_volume

                    # Increment Counter
                    order_counter += 1
                    max_buy += curr_volume
                else:
                    break

        # If any BUY orders and position limits allow selling
        if (num_buy > 0) and (position > -limit):
            order_counter: int = 0
            while (max_sell < 0) and (order_counter < num_buy):
                curr_bid: int = sorted_bids[order_counter]
                curr_volume: int = min(order_depth.buy_orders[curr_bid], -max_sell)

                # If the highest bid is more than the fair value
                if curr_bid > value:
                    amount += -curr_volume

                    # Increment Counter
                    order_counter += 1
                    max_sell += curr_volume
                else:
                    break

        # Return the calculated amount
        return amount

    # Calculate the amount to buy/sell using simple price direction indication from the order book
    def calculate_price_direction_amount(self, product: str,
                                         imbalance_threshold: float,
                                         amount_of_history: int,
                                         state: TradingState) -> int:
        # Get all values for the product
        order_depth: OrderDepth = state.order_depths[product]
        num_sell: int = len(order_depth.sell_orders)
        num_buy: int = len(order_depth.buy_orders)
        historicalOBV: List[float] = self.historicalOBV[product]
        position: int = get_position(product, state)

        # Get and calculate limits
        limit: int = self.limits[product]
        max_buy: int = limit - position
        max_sell: int = -limit - position

        # Get general info
        sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
        sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)

        # Initialize result
        amount: int = 0

        # Ensure there is enough historical data and are enough buy/sell orders
        if (len(historicalOBV) >= amount_of_history) and (num_buy > 0) and (num_sell > 0):
            # Imbalance Ratio
            imbalance_ratio: float = sum(order_depth.buy_orders.values()) / abs(sum(order_depth.sell_orders.values()))

            # Gradient of OBV
            obv_gradient: float = np.gradient(historicalOBV)[-1]
            obv_average: float = np.mean(historicalOBV[-amount_of_history:])

            # Imbalance > 1 => Price Up, Current OBV Gradient > OBV Average => BUY
            if (imbalance_ratio > imbalance_threshold) and (obv_gradient > obv_average) and (max_buy > 0):
                order_counter: int = 0
                while (max_buy > 0) and (order_counter < num_sell):
                    curr_ask: int = sorted_asks[order_counter]
                    curr_volume: int = max(order_depth.sell_orders[curr_ask], -max_buy)

                    amount += -curr_volume

                    # Increment Counter
                    order_counter += 1
                    max_buy += curr_volume

            # Imbalance < 1 => Price Down, Current OBV Gradient < OBV Average => SELL
            if (imbalance_ratio < imbalance_threshold) and (obv_gradient < obv_average) and (max_sell < 0):
                order_counter: int = 0
                while (max_sell < 0) and (order_counter < num_buy):
                    curr_bid: int = sorted_bids[order_counter]
                    curr_volume: int = min(order_depth.buy_orders[curr_bid], -max_sell)

                    amount += -curr_volume

                    # Increment Counter
                    order_counter += 1
                    max_sell += curr_volume

        # Return the calculated amount
        return amount

    # Calculate the amounts to buy/sell using pair trading
    def calculate_pair_trading_amounts(self,
                                       product1: str,
                                       product2: str,
                                       entry_threshold: float,
                                       exit_threshold: float,
                                       amount_of_history: int,
                                       state: TradingState) -> Tuple[int, int]:
        # Initialize historical ratio list if it doesn't exist yet
        both_products = product1 + product2
        if both_products not in self.historicalPairRatios.keys():
            self.historicalPairRatios[both_products] = []

        if (product1 in state.order_depths.keys()) and (product2 in state.order_depths.keys()):
            # Get Current Positions
            product1_position: Position = get_position(product1, state)
            product2_position: Position = get_position(product2, state)

            # Get Current Prices
            product1_price: float = self.historicalPrice[product1][-1]
            product2_price: float = self.historicalPrice[product2][-1]
            product1_ask_price: int = self.historicalBestAsk[product1][-1]
            product1_bid_price: int = self.historicalBestBid[product1][-1]
            product2_ask_price: int = self.historicalBestAsk[product2][-1]
            product2_bid_price: int = self.historicalBestBid[product2][-1]

            # Get Current Volumes
            product1_order_depth: OrderDepth = state.order_depths[product1]
            product2_order_depth: OrderDepth = state.order_depths[product2]

            product1_bid_volume: int = product1_order_depth.buy_orders[product1_bid_price]
            product2_bid_volume: int = product2_order_depth.buy_orders[product2_bid_price]
            product1_ask_volume: int = product1_order_depth.sell_orders[product1_ask_price]
            product2_ask_volume: int = product2_order_depth.sell_orders[product2_ask_price]

            # Get and Calculate Limits
            product1_limit: int = self.limits[product1]
            product2_limit: int = self.limits[product2]
            product1_max_buy: int = product1_limit - product1_position
            product2_max_buy: int = product2_limit - product2_position
            product1_max_sell: int = -product1_limit - product1_position
            product2_max_sell: int = -product2_limit - product2_position

            product1_sell_limit: int = max(product1_max_sell, -product1_bid_volume)
            product2_sell_limit: int = max(product2_max_sell, -product2_bid_volume)
            product1_buy_limit: int = min(product1_max_buy, -product1_ask_volume)
            product2_buy_limit: int = min(product2_max_buy, -product2_ask_volume)

            # Calculate and Keep Track of the Ratio
            current_ratio: float = product2_price / product1_price
            self.historicalPairRatios[both_products].append(current_ratio)

            # Ensure that there is enough historical data
            if len(self.historicalPairRatios[both_products]) >= amount_of_history:
                historical_ratios: list = self.historicalPairRatios[both_products][-amount_of_history:]

                # Calculate historical values
                ratio_average: float = np.mean(historical_ratios)
                ratio_std: float = np.std(historical_ratios)

                # Calculate z-score
                z_score: float = (current_ratio - ratio_average) / ratio_std

                # Exit Any Existing Position if Ratio has Normalized
                if -exit_threshold <= z_score <= exit_threshold:
                    return -product1_position, -product2_position
                # Enter a New Position if Ratio Passes Entry Threshold (Upper)
                elif z_score > entry_threshold:
                    # Product 2 Overpriced (i.e. sell) or Product 1 Underpriced (i.e. buy)
                    product2_amount = max(product2_sell_limit, -round(product1_buy_limit * current_ratio))  # Sell
                    product1_amount = min(product1_buy_limit, round(product2_sell_limit / current_ratio))   # Buy
                    return product1_amount, product2_amount
                # Enter a New Position if Ratio Passes Entry Threshold (Lower)
                elif z_score < -entry_threshold:
                    # Product 2 Underpriced (i.e. buy) or Product 1 Overpriced (i.e. sell)
                    product2_amount = min(product2_buy_limit, round(product1_sell_limit / current_ratio))   # Buy
                    product1_amount = max(product1_sell_limit, -round(product2_buy_limit * current_ratio))  # Sell
                    return product1_amount, product2_amount
            # If Z-Score Within Thresholds Or Not Enough History => Don't Buy or Sell
            return 0, 0

    # Calculate the amount to buy/sell using mean reversion on a tracked product
    def calculate_mean_reversion_amount(self,
                                        tracked_product: str,
                                        traded_product: str,
                                        number_of_deviations: int,
                                        amount_of_time_for_present: int,
                                        amount_of_time_for_history: int,
                                        state: TradingState) -> int:
        # Get all values for the traded product
        order_depth: OrderDepth = state.order_depths[traded_product]
        num_sell: int = len(order_depth.sell_orders)
        num_buy: int = len(order_depth.buy_orders)
        position: int = get_position(traded_product, state)

        # Get and calculate limits
        limit: int = self.limits[traded_product]
        max_buy: int = limit - position
        max_sell: int = -limit - position

        # Get details for tracked product
        historicalObservations: List[float] = self.historicalObservations[tracked_product]

        # Get general info
        sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
        sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)

        # Initialize result
        amount: int = 0

        # Ensure there is enough historical data
        if len(historicalObservations) >= amount_of_time_for_history:
            moving_average: float = np.mean(historicalObservations[-amount_of_time_for_history:])
            curr_observations: List[float] = historicalObservations[-amount_of_time_for_present:]
            standard_deviation: float = np.std(np.subtract(curr_observations, moving_average))

            if (num_sell > 0) and (position < limit):
                lower_band = moving_average - (number_of_deviations * standard_deviation)
                order_counter: int = 0
                while (max_buy > 0) and (order_counter < num_sell):
                    curr_ask: int = sorted_asks[order_counter]
                    curr_volume: int = max(order_depth.sell_orders[curr_ask], -max_buy)

                    if curr_ask < lower_band:
                        amount += -curr_volume

                        # Increment Counter
                        order_counter += 1
                        max_buy += curr_volume
                    else:
                        break

            if (num_buy > 0) and (position < limit):
                upper_band = moving_average + (number_of_deviations * standard_deviation)

                order_counter: int = 0
                while (max_sell < 0) and (order_counter < num_buy):
                    curr_bid: int = sorted_bids[order_counter]
                    curr_volume: int = min(order_depth.buy_orders[curr_bid], -max_sell)

                    if curr_bid > upper_band:
                        amount += -curr_volume

                        # Increment Counter
                        order_counter += 1
                        max_sell += curr_volume
                    else:
                        break

        return amount

    # Calculate the amount to buy/sell using simple gradient calculations on a tracked product
    def calculate_gradient_based_tracking_amount(self,
                                                 tracked_product: str,
                                                 traded_product: str,
                                                 buy_threshold: float,
                                                 sell_threshold: float,
                                                 amount_of_time_for_history: int,
                                                 state: TradingState) -> int:
        # Calculate gradient
        if len(self.historicalObservations[tracked_product]) > amount_of_time_for_history:
            gradient = np.gradient(self.historicalObservations[tracked_product][-amount_of_time_for_history:])
            curr_gradient = gradient[-1]
        else:
            return 0

        # Get all values for the product
        order_depth: OrderDepth = state.order_depths[traded_product]
        num_sell: int = len(order_depth.sell_orders)
        num_buy: int = len(order_depth.buy_orders)
        position: int = get_position(traded_product, state)

        # Get and calculate limits
        limit: int = self.limits[traded_product]
        max_buy: int = limit - position
        max_sell: int = -limit - position

        # Get general info
        sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
        sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)

        # Initialize result
        amount: int = 0

        if curr_gradient > buy_threshold:
            # Buy
            order_counter: int = 0
            while (max_buy > 0) and (order_counter < num_sell):
                curr_ask: int = sorted_asks[order_counter]
                curr_volume: int = max(order_depth.sell_orders[curr_ask], -max_buy)

                amount += -curr_volume

                # Increment Counter
                order_counter += 1
                max_buy += curr_volume
        elif curr_gradient < sell_threshold:
            # Sell
            order_counter: int = 0
            while (max_sell < 0) and (order_counter < num_buy):
                curr_bid: int = sorted_bids[order_counter]
                curr_volume: int = min(order_depth.buy_orders[curr_bid], -max_sell)

                amount += -curr_volume

                # Increment Counter
                order_counter += 1
                max_sell += curr_volume

        return amount

    # Calculate the amount to buy/sell using time
    def calculate_time_based_amount(self, product: str, buy_time: Time, sell_time: Time, state: TradingState):
        # Initialize the amount
        amount: int = 0

        # Get and calculate limits
        position: int = get_position(product, state)
        limit: int = self.limits[product]
        max_buy: int = limit - position
        max_sell: int = -limit - position

        # Buy if its buy time
        if state.timestamp == buy_time:
            amount = max_buy
        # Sell if its sell time
        elif state.timestamp == sell_time:
            amount = max_sell

        return amount

    # Calculate the amount to buy/sell using statistical arbitrage
    def calculate_statistical_arbitrage_amount(self,
                                               product1: str,
                                               product2: str,
                                               product3: str,
                                               product4: str,
                                               distance_from_usual_sum: int,
                                               state: TradingState
                                               ) -> Tuple[int, int, int, int]:
        # Get all values for the products
        order_depth1: OrderDepth = state.order_depths[product1]
        order_depth2: OrderDepth = state.order_depths[product2]
        order_depth3: OrderDepth = state.order_depths[product3]
        order_depth4: OrderDepth = state.order_depths[product4]

        curr_price1: float = self.historicalPrice[product1][-1]
        curr_price2: float = self.historicalPrice[product2][-1]
        curr_price3: float = self.historicalPrice[product3][-1]
        curr_price4: float = self.historicalPrice[product4][-1]

        # Get general info
        sorted_asks1: List[int] = sorted(set(order_depth1.sell_orders.keys()))
        sorted_bids1: List[int] = sorted(set(order_depth1.buy_orders.keys()), reverse=True)
        sorted_asks2: List[int] = sorted(set(order_depth2.sell_orders.keys()))
        sorted_bids2: List[int] = sorted(set(order_depth2.buy_orders.keys()), reverse=True)
        sorted_asks3: List[int] = sorted(set(order_depth3.sell_orders.keys()))
        sorted_bids3: List[int] = sorted(set(order_depth3.buy_orders.keys()), reverse=True)
        sorted_asks4: List[int] = sorted(set(order_depth4.sell_orders.keys()))
        sorted_bids4: List[int] = sorted(set(order_depth4.buy_orders.keys()), reverse=True)

        # Get and calculate limits
        position1: int = get_position(product1, state)
        position2: int = get_position(product2, state)
        position3: int = get_position(product3, state)
        position4: int = get_position(product4, state)

        limit1: int = self.limits[product1]
        limit2: int = self.limits[product2]
        limit3: int = self.limits[product3]
        limit4: int = self.limits[product4]

        max_buy1: int = limit1 - position1
        max_buy2: int = limit2 - position2
        max_buy3: int = limit3 - position3
        max_buy4: int = limit4 - position4

        max_sell1: int = -limit1 - position1
        max_sell2: int = -limit2 - position2
        max_sell3: int = -limit3 - position3
        max_sell4: int = -limit4 - position4

        product1_max_buy = min(max_buy1, sorted_bids1[0])
        product2_max_buy = min(max_buy2, sorted_bids2[0])
        product3_max_buy = min(max_buy3, sorted_bids3[0])
        product4_max_buy = min(max_buy4, sorted_bids4[0])

        product1_max_sell = max(max_sell1, sorted_asks1[0])
        product2_max_sell = max(max_sell2, sorted_asks2[0])
        product3_max_sell = max(max_sell3, sorted_asks3[0])
        product4_max_sell = max(max_sell4, sorted_asks4[0])

        sum_value: float = (2 * curr_price1) + (4 * curr_price2) + curr_price3

        print("Sum", sum_value)
        print("Curr Price 4", curr_price4)

        if sum_value < (curr_price4 - distance_from_usual_sum):
            # Sell Product 4
            product4_target_buy = min(product4_max_buy,
                                      round(product1_max_sell / 2),
                                      round(product2_max_sell / 4),
                                      product3_max_sell)
            # Buy 2 * Product 1, 4 * Product 2, 1 * Product 3
            product1_target_sell = 2 * product4_target_buy
            product2_target_sell = 4 * product4_target_buy
            product3_target_sell = product4_target_buy

            return product1_target_sell, product2_target_sell, product3_target_sell, product4_target_buy
        elif sum_value > (curr_price4 + distance_from_usual_sum):
            # Buy Product 4
            product4_target_sell = min(product4_max_sell,
                                       round(product1_max_buy / 2),
                                       round(product2_max_buy / 4),
                                       product3_max_buy)
            # Sell 2 * Product 1, 4 * Product 2, 1 * Product 3
            product1_target_buy = 2 * product4_target_sell
            product2_target_buy = 4 * product4_target_sell
            product3_target_buy = product4_target_sell

            return product1_target_buy, product2_target_buy, product3_target_buy, product4_target_sell
        else:
            # Exit Position
            return -position1, -position2, -position3, -position4


    def run(self,
            state: TradingState,
            value_to_test_1: float = None,
            value_to_test_2: float = None,
            value_to_test_3: float = None
            ) -> Dict[str, List[Order]]:
        """
        Only method required. It takes all buy and sell orders for all symbols as an input,
        and outputs a list of orders to be sent
        """
        # Initialize the method output dict as an empty dict
        result: Dict[str, List[Order]] = {}

        # Initialize amounts dictionary
        amounts: Dict[str, int] = {
            "PEARLS": 0,
            "BANANAS": 0,
            "COCONUTS": 0,
            "PINA_COLADAS": 0,
            "BERRIES": 0,
            "DIVING_GEAR": 0,
            "BAGUETTE": 0,
            "DIP": 0,
            "UKULELE": 0,
            "PICNIC_BASKET": 0
        }

        # Update Dolphin Sightings
        self.historicalObservations['DOLPHIN_SIGHTINGS'].append(state.observations['DOLPHIN_SIGHTINGS'])

        # Calculate and keep track of product values
        for product in state.order_depths.keys():
            self.update_history(product, state)

        # Simple Arbitrage for Pearls
        amounts['PEARLS'] = self.calculate_arbitrage_amount('PEARLS', 10000.00, state)

        # Simple Price Direction Indication via Order Book for Bananas
        amounts['BANANAS'] = self.calculate_price_direction_amount('BANANAS', 2, 0, state)

        # Pair Trading w/ Z-Score for Coconuts and Pina Coladas
        amounts['COCONUTS'], amounts['PINA_COLADAS'] = self.calculate_pair_trading_amounts('COCONUTS',
                                                                                           'PINA_COLADAS',
                                                                                           1,
                                                                                           2,
                                                                                           600,
                                                                                           state)

        # Time-Based Trading for Mayberries
        amounts['BERRIES'] = self.calculate_time_based_amount('BERRIES', 132000, 504000, state)

        # Mean Reversion Based Tracking on Dolphin Sightings to Calculate Price of Diving Gear
        amounts['DIVING_GEAR'] = self.calculate_mean_reversion_amount('DOLPHIN_SIGHTINGS',
                                                                      'DIVING_GEAR',
                                                                      2,
                                                                      50,
                                                                      150,
                                                                      state)

        amounts['BAGUETTE'], amounts['DIP'], amounts['UKULELE'], amounts['PICNIC_BASKET'] = \
            self.calculate_statistical_arbitrage_amount('BAGUETTE',
                                                        'DIP',
                                                        'UKULELE',
                                                        'PICNIC_BASKET',
                                                        100,
                                                        state)

        # Make purchases for each product based on previously calculated amounts
        for product in state.order_depths.keys():
            if product == 'DOLPHIN_SIGHTINGS':
                continue

            # Get general info about current market
            order_depth: OrderDepth = state.order_depths[product]
            num_sell: int = len(order_depth.sell_orders)
            num_buy: int = len(order_depth.buy_orders)
            sorted_asks: List[int] = sorted(set(order_depth.sell_orders.keys()))
            sorted_bids: List[int] = sorted(set(order_depth.buy_orders.keys()), reverse=True)

            # Get product specific values
            amount: int = amounts[product]

            # Initialize the list of Orders to be sent
            orders: List[Order] = []

            if amount != 0:
                order_counter: int = 0
                while (amount > 0) and order_counter < num_sell:
                    # BUY
                    curr_ask: int = sorted_asks[-order_counter]
                    curr_volume: int = max(-amount, order_depth.sell_orders[curr_ask])

                    print("BUY", str(-curr_volume) + " " + product, curr_ask)
                    orders.append(Order(product, curr_ask, -curr_volume))

                    amount -= -curr_volume
                    order_counter += 1

                order_counter: int = 0
                while (amount < 0) and (order_counter < num_buy):
                    # SELL
                    curr_bid: int = sorted_bids[-order_counter]
                    curr_volume: int = min(-amount, order_depth.buy_orders[curr_bid])

                    print("SELL", str(curr_volume) + " " + product, curr_bid)
                    orders.append(Order(product, curr_bid, -curr_volume))

                    amount += curr_volume

                    order_counter += 1
            else:
                continue

            # Add all the above orders to the result dict
            result[product] = orders

        print("Position:", state.position)

        # Return the dict of orders
        return result
