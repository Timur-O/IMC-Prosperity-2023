"""
    Based on:
        * https://github.com/n-0/backtest-imc-prosperity-2023
        * https://github.com/BRCSmith/backtest-imc-prosperity-2023
"""
import copy
import os
import random
import statistics
import uuid
from datetime import datetime

import numpy as np
import pandas as pd

from Trader import Trader
from datamodel import *

# Time-steps used in training files
TIME_DELTA = 100

# Please put all! the price and log files into
# the same directory or adjust the code accordingly
TRAINING_DATA_PREFIX = "./data"

ALL_SYMBOLS = [
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'DOLPHIN_SIGHTINGS',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET'
]

POSITIONABLE_SYMBOLS = [
    'PEARLS',
    'BANANAS',
    'COCONUTS',
    'PINA_COLADAS',
    'DIVING_GEAR',
    'BERRIES',
    'BAGUETTE',
    'DIP',
    'UKULELE',
    'PICNIC_BASKET'
]

first_round = ['PEARLS', 'BANANAS']
snd_round = first_round + ['COCONUTS', 'PINA_COLADAS']
third_round = snd_round + ['DIVING_GEAR', 'DOLPHIN_SIGHTINGS', 'BERRIES']
fourth_round = third_round + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round = fourth_round

SYMBOLS_BY_ROUND = {
    1: first_round,
    2: snd_round,
    3: third_round,
    4: fourth_round,
    5: fifth_round,
}

first_round_pst = ['PEARLS', 'BANANAS']
snd_round_pst = first_round_pst + ['COCONUTS', 'PINA_COLADAS']
third_round_pst = snd_round_pst + ['DIVING_GEAR', 'BERRIES']
fourth_round_pst = third_round_pst + ['BAGUETTE', 'DIP', 'UKULELE', 'PICNIC_BASKET']
fifth_round_pst = fourth_round_pst

SYMBOLS_BY_ROUND_POSITIONABLE = {
    1: first_round_pst,
    2: snd_round_pst,
    3: third_round_pst,
    4: fourth_round_pst,
    5: fifth_round_pst,
}


def process_prices(df_prices, round_to_sim, time_limit) -> dict[int, TradingState]:
    states = {}
    for _, row in df_prices.iterrows():
        time: int = int(row["timestamp"])
        if time > time_limit:
            break
        product: str = row["product"]
        if states.get(time) is None:
            position: Dict[Product, Position] = {}
            own_trades: Dict[Symbol, List[Trade]] = {}
            market_trades: Dict[Symbol, List[Trade]] = {}
            observations: Dict[Product, Observation] = {}
            listings = {}
            depths = {}
            states[time] = TradingState(time, listings, depths, own_trades, market_trades, position, observations)

        if product not in states[time].position and product in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
            states[time].position[product] = 0
            states[time].own_trades[product] = []
            states[time].market_trades[product] = []

        states[time].listings[product] = Listing(product, product, "1")

        if product == "DOLPHIN_SIGHTINGS":
            states[time].observations["DOLPHIN_SIGHTINGS"] = row['mid_price']
        else:
            depth = OrderDepth()
            if row["bid_price_1"] > 0:
                depth.buy_orders[row["bid_price_1"]] = int(row["bid_volume_1"])
            if row["bid_price_2"] > 0:
                depth.buy_orders[row["bid_price_2"]] = int(row["bid_volume_2"])
            if row["bid_price_3"] > 0:
                depth.buy_orders[row["bid_price_3"]] = int(row["bid_volume_3"])
            if row["ask_price_1"] > 0:
                depth.sell_orders[row["ask_price_1"]] = -int(row["ask_volume_1"])
            if row["ask_price_2"] > 0:
                depth.sell_orders[row["ask_price_2"]] = -int(row["ask_volume_2"])
            if row["ask_price_3"] > 0:
                depth.sell_orders[row["ask_price_3"]] = -int(row["ask_volume_3"])
            states[time].order_depths[product] = depth

    return states


def process_trades(df_trades, states: dict[int, TradingState], time_limit, name=True):
    for _, trade in df_trades.iterrows():
        time: int = trade['timestamp']
        if time > time_limit:
            break
        symbol = trade['symbol']
        if symbol not in states[time].market_trades:
            states[time].market_trades[symbol] = []
        if name:
            t = Trade(
                symbol,
                trade['price'],
                trade['quantity'],
                str(trade['buyer']),
                str(trade['seller']),
                time)
        else:
            t = Trade(
                symbol,
                trade['price'],
                trade['quantity'],
                '',  # str(trade['buyer']),
                '',  # str(trade['seller']),
                time)
        states[time].market_trades[symbol].append(t)
    return states


current_limits = {
    'PEARLS': 20,
    'BANANAS': 20,
    'COCONUTS': 600,
    'PINA_COLADAS': 300,
    'DIVING_GEAR': 50,
    'BERRIES': 250,
    'BAGUETTE': 150,
    'DIP': 300,
    'UKULELE': 70,
    'PICNIC_BASKET': 70,
}


def calc_mid(states: dict[int, TradingState], round_to_sim: int, time: int, max_time: int) -> dict[str, float]:
    medians_by_symbol = {}
    non_empty_time = time
    for p_symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
        hit_zero = False
        while len(states[non_empty_time].order_depths[p_symbol].sell_orders.keys()) == 0 or len(
                states[non_empty_time].order_depths[p_symbol].buy_orders.keys()) == 0:
            # little hack
            if time == 0 or hit_zero and time != max_time:
                hit_zero = True
                non_empty_time += TIME_DELTA
            else:
                non_empty_time -= TIME_DELTA
        min_ask = min(states[non_empty_time].order_depths[p_symbol].sell_orders.keys())
        max_bid = max(states[non_empty_time].order_depths[p_symbol].buy_orders.keys())
        median_price = statistics.median([min_ask, max_bid])
        medians_by_symbol[p_symbol] = median_price
    return medians_by_symbol


# Setting a high time_limit can be harder to visualize
# print_position prints the position before! every Trader.run
def simulate_alternative(round_to_sim: int,
                         day_to_sim: int,
                         trader,
                         time_limit=999900,
                         names=True,
                         halfway=False,
                         monkeys=False,
                         monkey_names=['Max', 'Camilla'],
                         value_to_test_1: float = None,
                         value_to_test_2: float = None,
                         value_to_test_3: float = None):
    prices_path = os.path.join(TRAINING_DATA_PREFIX, f'prices_round_{round_to_sim}_day_{day_to_sim}.csv')
    trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round_to_sim}_day_{day_to_sim}_wn.csv')
    if not names:
        trades_path = os.path.join(TRAINING_DATA_PREFIX, f'trades_round_{round_to_sim}_day_{day_to_sim}_nn.csv')
    df_prices = pd.read_csv(prices_path, sep=';')
    df_trades = pd.read_csv(trades_path, sep=';', dtype={'seller': str, 'buyer': str})

    states = process_prices(df_prices, round_to_sim, time_limit)
    states = process_trades(df_trades, states, time_limit, names)
    ref_symbols = list(states[0].position.keys())
    max_time = max(list(states.keys()))

    # handling these four is rather tricky
    profits_by_symbol: dict[int, dict[str, float]] = {0: dict(zip(ref_symbols, [0.0] * len(ref_symbols)))}
    balance_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}
    credit_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}
    unrealized_by_symbol: dict[int, dict[str, float]] = {0: copy.deepcopy(profits_by_symbol[0])}

    states, trader, profits_by_symbol, balance_by_symbol = trades_position_pnl_run(states,
                                                                                   trader,
                                                                                   round_to_sim,
                                                                                   halfway,
                                                                                   value_to_test_1,
                                                                                   value_to_test_2,
                                                                                   value_to_test_3,
                                                                                   max_time,
                                                                                   profits_by_symbol,
                                                                                   balance_by_symbol,
                                                                                   credit_by_symbol,
                                                                                   unrealized_by_symbol)

    profits_by_symbol = create_log_file(round_to_sim, day_to_sim, states, profits_by_symbol, balance_by_symbol, trader)

    if monkeys:
        profit_balance, trades_monkey = monkey_positions(monkey_names, states, round_to_sim, max_time)
        print(f'PNL + BALANCE monkeys {profit_balance[max_time]}')
        print(f'Trades monkeys {trades_monkey[max_time]}')

    return profits_by_symbol


def trades_position_pnl_run(states: dict[int, TradingState],
                            trader,
                            round_to_sim: int,
                            halfway: bool,
                            value_to_test_1: float,
                            value_to_test_2: float,
                            value_to_test_3: float,
                            max_time: int,
                            profits_by_symbol: dict[int, dict[str, float]],
                            balance_by_symbol: dict[int, dict[str, float]],
                            credit_by_symbol: dict[int, dict[str, float]],
                            unrealized_by_symbol: dict[int, dict[str, float]]):
    for time, state in states.items():
        position = copy.deepcopy(state.position)
        orders = trader.run(state, value_to_test_1, value_to_test_2, value_to_test_3)
        trades = clear_order_book(orders, state.order_depths, time, halfway)
        mids = calc_mid(states, round_to_sim, time, max_time)
        if profits_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
            profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
        if credit_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
            credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
        if balance_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
            balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
        if unrealized_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
            unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
            for p_symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                unrealized_by_symbol[time + TIME_DELTA][p_symbol] = mids[p_symbol] * position[p_symbol]
        valid_trades = []
        failed_symbol = []
        grouped_by_symbol = {}
        if len(trades) > 0:
            for trade in trades:
                if trade.symbol in failed_symbol:
                    continue
                n_position = position[trade.symbol] + trade.quantity
                if abs(n_position) > current_limits[trade.symbol]:
                    print('ILLEGAL TRADE, WOULD EXCEED POSITION LIMIT, KILLING ALL REMAINING ORDERS')
                    trade_vars = vars(trade)
                    trade_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                    print(f'Stopped at the following trade: {trade_str}')
                    print(f"All trades that were sent:")
                    for trade_inner in trades:
                        trade_vars = vars(trade_inner)
                        trades_str = ', '.join("%s: %s" % item for item in trade_vars.items())
                        print(trades_str)
                    failed_symbol.append(trade.symbol)
                valid_trades.append(trade)
        FLEX_TIME_DELTA = TIME_DELTA
        if time == max_time:
            FLEX_TIME_DELTA = 0
        for valid_trade in valid_trades:
            position[valid_trade.symbol] += valid_trade.quantity
            if grouped_by_symbol.get(valid_trade.symbol) is None:
                grouped_by_symbol[valid_trade.symbol] = []
            grouped_by_symbol[valid_trade.symbol].append(valid_trade)
            credit_by_symbol[time + FLEX_TIME_DELTA][
                valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
        if states.get(time + FLEX_TIME_DELTA) is not None:
            states[time + FLEX_TIME_DELTA].own_trades = grouped_by_symbol
            for p_symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                unrealized_by_symbol[time + FLEX_TIME_DELTA][p_symbol] = mids[p_symbol] * position[p_symbol]
                if position[p_symbol] == 0 and states[time].position[p_symbol] != 0:
                    profits_by_symbol[time + FLEX_TIME_DELTA][p_symbol] += \
                        credit_by_symbol[time + FLEX_TIME_DELTA][p_symbol]
                    credit_by_symbol[time + FLEX_TIME_DELTA][p_symbol] = 0
                    balance_by_symbol[time + FLEX_TIME_DELTA][p_symbol] = 0
                else:
                    balance_by_symbol[time + FLEX_TIME_DELTA][p_symbol] = \
                        credit_by_symbol[time + FLEX_TIME_DELTA][p_symbol] + \
                        unrealized_by_symbol[time + FLEX_TIME_DELTA][p_symbol]

        if time == max_time:
            print("End of simulation reached. All positions left are liquidated")
            # I have the feeling this already has been done, and only repeats the same values as before
            for o_symbol in position.keys():
                profits_by_symbol[time + FLEX_TIME_DELTA][o_symbol] += \
                    credit_by_symbol[time + FLEX_TIME_DELTA][o_symbol] + \
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][o_symbol]
                balance_by_symbol[time + FLEX_TIME_DELTA][o_symbol] = 0
        if states.get(time + FLEX_TIME_DELTA) is not None:
            states[time + FLEX_TIME_DELTA].position = copy.deepcopy(position)

    return states, trader, profits_by_symbol, balance_by_symbol


def monkey_positions(monkey_names: list[str],
                     states: dict[int, TradingState],
                     round_to_sim: int,
                     max_time: int):
    profits_by_symbol: dict[int, dict[str, dict[str, float]]] = {0: {}}
    balance_by_symbol: dict[int, dict[str, dict[str, float]]] = {0: {}}
    credit_by_symbol: dict[int, dict[str, dict[str, float]]] = {0: {}}
    unrealized_by_symbol: dict[int, dict[str, dict[str, float]]] = {0: {}}
    prev_monkey_positions: dict[str, dict[str, int]] = {}
    curr_monkey_positions: dict[str, dict[str, int]] = {}
    trades_by_round: dict[int, dict[str, list[Trade]]] = {0: dict(zip(monkey_names,
                                                                      [[] for _ in range(len(monkey_names))]
                                                                      ))}
    profit_balance: dict[int, dict[str, dict[str, float]]] = {0: {}}

    for monkey in monkey_names:
        ref_symbols = list(states[0].position.keys())
        profits_by_symbol[0][monkey] = dict(zip(ref_symbols, [0.0] * len(ref_symbols)))
        balance_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        credit_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        unrealized_by_symbol[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        profit_balance[0][monkey] = copy.deepcopy(profits_by_symbol[0][monkey])
        curr_monkey_positions[monkey] = dict(
            zip(SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim], [0] * len(SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim])))
        prev_monkey_positions[monkey] = copy.deepcopy(curr_monkey_positions[monkey])

    for time, state in states.items():
        already_calculated = False
        for monkey in monkey_names:
            position = copy.deepcopy(curr_monkey_positions[monkey])
            mids = calc_mid(states, round_to_sim, time, max_time)
            if trades_by_round.get(time + TIME_DELTA) is None:
                trades_by_round[time + TIME_DELTA] = copy.deepcopy(trades_by_round[time])

            for p_symbol in POSITIONABLE_SYMBOLS:
                if already_calculated:
                    break
                if state.market_trades.get(p_symbol):
                    for market_trade in state.market_trades[p_symbol]:
                        if trades_by_round[time].get(market_trade.buyer) is not None:
                            trades_by_round[time][market_trade.buyer].append(
                                Trade(p_symbol, market_trade.price, market_trade.quantity))
                        if trades_by_round[time].get(market_trade.seller) is not None:
                            trades_by_round[time][market_trade.seller].append(
                                Trade(p_symbol, market_trade.price, -market_trade.quantity))
            already_calculated = True

            if profit_balance.get(time + TIME_DELTA) is None and time != max_time:
                profit_balance[time + TIME_DELTA] = copy.deepcopy(profit_balance[time])
            if profits_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
                profits_by_symbol[time + TIME_DELTA] = copy.deepcopy(profits_by_symbol[time])
            if credit_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
                credit_by_symbol[time + TIME_DELTA] = copy.deepcopy(credit_by_symbol[time])
            if balance_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
                balance_by_symbol[time + TIME_DELTA] = copy.deepcopy(balance_by_symbol[time])
            if unrealized_by_symbol.get(time + TIME_DELTA) is None and time != max_time:
                unrealized_by_symbol[time + TIME_DELTA] = copy.deepcopy(unrealized_by_symbol[time])
                for p_symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                    unrealized_by_symbol[time + TIME_DELTA][monkey][p_symbol] = mids[p_symbol] * position[p_symbol]
            valid_trades = []
            if trades_by_round[time].get(monkey) is not None:
                valid_trades = trades_by_round[time][monkey]
            FLEX_TIME_DELTA = TIME_DELTA
            if time == max_time:
                FLEX_TIME_DELTA = 0
            for valid_trade in valid_trades:
                position[valid_trade.symbol] += valid_trade.quantity
                credit_by_symbol[time + FLEX_TIME_DELTA][monkey][
                    valid_trade.symbol] += -valid_trade.price * valid_trade.quantity
            if states.get(time + FLEX_TIME_DELTA) is not None:
                for p_symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                    unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] = mids[p_symbol] * position[p_symbol]
                    if position[p_symbol] == 0 and prev_monkey_positions[monkey][p_symbol] != 0:
                        profits_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] += \
                            credit_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol]
                        credit_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] = 0
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] = 0
                    else:
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] = \
                            credit_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] + \
                            unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol]
                    profit_balance[time + FLEX_TIME_DELTA][monkey][p_symbol] = \
                        profits_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol] + \
                        balance_by_symbol[time + FLEX_TIME_DELTA][monkey][p_symbol]
            prev_monkey_positions[monkey] = copy.deepcopy(curr_monkey_positions[monkey])
            curr_monkey_positions[monkey] = position
            if time == max_time:
                print("End of monkey simulation reached.")
                # I have the feeling this already has been done, and only repeats the same values as before
                for o_symbol in position.keys():
                    profits_by_symbol[time + FLEX_TIME_DELTA][monkey][o_symbol] += \
                        credit_by_symbol[time + FLEX_TIME_DELTA][monkey][o_symbol] + \
                        unrealized_by_symbol[time + FLEX_TIME_DELTA][monkey][o_symbol]
                    balance_by_symbol[time + FLEX_TIME_DELTA][monkey][o_symbol] = 0
    return profit_balance, trades_by_round


def cleanup_order_volumes(org_orders: List[Order]) -> List[Order]:
    orders = []
    for order_1 in org_orders:
        final_order = copy.copy(order_1)
        for order_2 in org_orders:
            if order_1.price == order_2.price and order_1.quantity == order_2.quantity:
                continue
            if order_1.price == order_2.price:
                final_order.quantity += order_2.quantity
        orders.append(final_order)
    return orders


def clear_order_book(trader_orders: dict[str, List[Order]], order_depth: dict[str, OrderDepth], time: int,
                     halfway: bool) -> list[Trade]:
    trades = []
    for symbol in trader_orders.keys():
        if order_depth.get(symbol) is not None:
            symbol_order_depth = copy.deepcopy(order_depth[symbol])
            t_orders = cleanup_order_volumes(trader_orders[symbol])
            for order in t_orders:
                if order.quantity < 0:
                    if halfway:
                        bids = symbol_order_depth.buy_orders.keys()
                        asks = symbol_order_depth.sell_orders.keys()
                        max_bid = max(bids)
                        min_ask = min(asks)
                        if order.price <= statistics.median([max_bid, min_ask]):
                            trades.append(Trade(symbol, order.price, order.quantity, "BOT", "YOU", time))
                        else:
                            print(f'No matches for order {order} at time {time}')
                            print(f'Order depth is {order_depth[order.symbol].__dict__}')
                    else:
                        potential_matches = list(
                            filter(lambda o: o[0] == order.price, symbol_order_depth.buy_orders.items()))
                        if len(potential_matches) > 0:
                            match = potential_matches[0]
                            if abs(match[1]) > abs(order.quantity):
                                final_volume = order.quantity
                            else:
                                # this should be negative
                                final_volume = -match[1]
                            trades.append(Trade(symbol, order.price, final_volume, "BOT", "YOU", time))
                        else:
                            print(f'No matches for order {order} at time {time}')
                            print(f'Order depth is {order_depth[order.symbol].__dict__}')
                if order.quantity > 0:
                    if halfway:
                        bids = symbol_order_depth.buy_orders.keys()
                        asks = symbol_order_depth.sell_orders.keys()
                        max_bid = max(bids)
                        min_ask = min(asks)
                        if order.price >= statistics.median([max_bid, min_ask]):
                            trades.append(Trade(symbol, order.price, order.quantity, "YOU", "BOT", time))
                        else:
                            print(f'No matches for order {order} at time {time}')
                            print(f'Order depth is {order_depth[order.symbol].__dict__}')
                    else:
                        potential_matches = list(
                            filter(lambda o: o[0] == order.price, symbol_order_depth.sell_orders.items()))
                        if len(potential_matches) > 0:
                            match = potential_matches[0]
                            # Match[1] will be negative so needs to be changed to work here
                            if abs(match[1]) > abs(order.quantity):
                                final_volume = order.quantity
                            else:
                                final_volume = abs(match[1])
                            trades.append(Trade(symbol, order.price, final_volume, "YOU", "BOT", time))
                        else:
                            print(f'No matches for order {order} at time {time}')
                            print(f'Order depth is {order_depth[order.symbol].__dict__}')
    return trades


csv_header = "day;timestamp;product;bid_price_1;bid_volume_1;bid_price_2;bid_volume_2;bid_price_3;bid_volume_3;" \
             "ask_price_1;ask_volume_1;ask_price_2;ask_volume_2;ask_price_3;ask_volume_3;mid_price;profit_and_loss\n"
log_header = [
    'Sandbox logs:\n',
    '0 OpenBLAS WARNING - could not determine the L2 cache size on this system, assuming 256k\n',
    'START RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085 Version: $LATEST\n',
    'END RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085\n',
    'REPORT RequestId: 8ab36ff8-b4e6-42d4-b012-e6ad69c42085	Duration: 18.73 ms	Billed Duration: 19 ms	'
    'Memory Size: 128 MB	Max Memory Used: 94 MB	Init Duration: 1574.09 ms\n',
]


def create_log_file(round_to_sim: int,
                    day_to_sim: int,
                    states: dict[int, TradingState],
                    profits_by_symbol: dict[int, dict[str, float]],
                    balance_by_symbol: dict[int, dict[str, float]],
                    trader: Trader):
    file_name = uuid.uuid4()
    time_stamp = datetime.timestamp(datetime.now())
    max_time = max(list(states.keys()))
    log_path = os.path.join('logs', f'{time_stamp}_{file_name}.log')
    with open(log_path, 'w', encoding="utf-8", newline='\n') as f:
        f.writelines(log_header)
        f.write('\n')
        for time, state in states.items():
            if hasattr(trader, 'logger'):
                if hasattr(trader.logger, 'local_logs') is not None:
                    if trader.logger.local_logs.get(time) is not None:
                        f.write(f'{time} {trader.logger.local_logs[time]}\n')
                        continue
            if time != 0:
                f.write(f'{time}\n')

        f.write(f'\n\n')
        f.write('Submission logs:\n\n\n')
        f.write('Activities log:\n')
        f.write(csv_header)
        for time, state in states.items():
            for symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                f.write(f'{day_to_sim};{time};{symbol};')
                bids_length = len(state.order_depths[symbol].buy_orders)
                bids = list(state.order_depths[symbol].buy_orders.items())
                bids_prices = list(state.order_depths[symbol].buy_orders.keys())
                bids_prices.sort()
                asks_length = len(state.order_depths[symbol].sell_orders)
                asks_prices = list(state.order_depths[symbol].sell_orders.keys())
                asks_prices.sort()
                asks = list(state.order_depths[symbol].sell_orders.items())
                if bids_length >= 3:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};{bids[2][0]};{bids[2][1]};')
                elif bids_length == 2:
                    f.write(f'{bids[0][0]};{bids[0][1]};{bids[1][0]};{bids[1][1]};;;')
                elif bids_length == 1:
                    f.write(f'{bids[0][0]};{bids[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if asks_length >= 3:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};{asks[2][0]};{asks[2][1]};')
                elif asks_length == 2:
                    f.write(f'{asks[0][0]};{asks[0][1]};{asks[1][0]};{asks[1][1]};;;')
                elif asks_length == 1:
                    f.write(f'{asks[0][0]};{asks[0][1]};;;;;')
                else:
                    f.write(f';;;;;;')
                if len(asks_prices) == 0 or max(bids_prices) == 0:
                    if symbol == 'DOLPHIN_SIGHTINGS':
                        dolphin_sightings = state.observations['DOLPHIN_SIGHTINGS']
                        f.write(f'{dolphin_sightings};{0.0}\n')
                    else:
                        f.write(f'{0};{0.0}\n')
                else:
                    actual_profit = 0.0
                    if symbol in SYMBOLS_BY_ROUND_POSITIONABLE[round_to_sim]:
                        actual_profit = profits_by_symbol[time][symbol] + balance_by_symbol[time][symbol]
                    min_ask = min(asks_prices)
                    max_bid = max(bids_prices)
                    median_price = statistics.median([min_ask, max_bid])
                    f.write(f'{median_price};{actual_profit}\n')
                    if time == max_time:
                        if profits_by_symbol[time].get(symbol) is None:
                            print(f'Final profit for {symbol} = {actual_profit}')
                if time == time_inp:
                    print(f'Final profit for {symbol} = {profits_by_symbol[time][symbol]}')

        print(f"\nSimulation on round {round_to_sim} day {day_to_sim} for time {max_time} complete")

        return profits_by_symbol


def value_optimization_tester_one(trader_to_use: Trader,
                                  product: str,
                                  param1_min: float,
                                  param1_max: float,
                                  param1_step: float):
    profits = []
    params = []

    for param1 in np.arange(param1_min, param1_max + param1_step, param1_step):
        curr_params = [param1]
        params.append(curr_params)

        curr_profits = []
        for day in [1, 2, 3]:
            result = simulate_alternative(rnd_inp,
                                          day,
                                          trader_to_use,
                                          time_inp,
                                          names_inp,
                                          False,
                                          False,
                                          curr_params[0])

            if product == 'CPC':
                curr_profit = result[time_inp]['COCONUTS'] + result[time_inp]['PINA_COLADAS']
            elif product == 'BDUP':
                curr_profit = result[time_inp]['BAGUETTE'] + result[time_inp]['DIP'] + \
                              result[time_inp]['UKULELE'] + result[time_inp]['PICNIC_BASKET']
            else:
                curr_profit = result[time_inp][product]

            curr_profits.append(curr_profit)

        avg_profit = np.mean(curr_profits)
        profits.append(avg_profit)

    best_params = params[np.argmax(profits)]
    best_profit = np.max(profits)

    print("Best Params: ", best_params)
    print("Max Profit: ", best_profit)

    return best_params, best_profit


def value_optimization_tester_two(trader_to_use: Trader,
                                  product: str,
                                  param1_min: float,
                                  param1_max: float,
                                  param1_step: float,
                                  param2_min: float,
                                  param2_max: float,
                                  param2_step: float):
    profits = []
    params = []

    for param1 in np.arange(param1_min, param1_max + param1_step, param1_step):
        for param2 in np.arange(param2_min, param2_max + param2_step, param2_step):
            curr_params = [param1, param2]
            params.append(curr_params)

            curr_profits = []
            for day in [1, 2, 3]:
                result = simulate_alternative(rnd_inp, day,
                                              trader_to_use,
                                              time_inp,
                                              names_inp,
                                              False,
                                              False,
                                              curr_params[0],
                                              curr_params[1])

                if product == 'CPC':
                    curr_profit = result[time_inp]['COCONUTS'] + result[time_inp]['PINA_COLADAS']
                elif product == 'BDUP':
                    curr_profit = result[time_inp]['BAGUETTE'] + result[time_inp]['DIP'] + \
                                  result[time_inp]['UKULELE'] + result[time_inp]['PICNIC_BASKET']
                else:
                    curr_profit = result[time_inp][product]

                curr_profits.append(curr_profit)

            avg_profit = np.mean(curr_profits)
            profits.append(avg_profit)

    best_params = params[np.argmax(profits)]
    best_profit = np.max(profits)

    print("Best Params: ", best_params)
    print("Max Profit: ", best_profit)

    return best_params, best_profit


def value_optimization_tester_three(trader_to_use: Trader,
                                    product: str,
                                    param1_min: float,
                                    param1_max: float,
                                    param1_step: float,
                                    param2_min: float,
                                    param2_max: float,
                                    param2_step: float,
                                    param3_min: float,
                                    param3_max: float,
                                    param3_step: float):
    profits = []
    params = []

    for param1 in np.arange(param1_min, param1_max + param1_step, param1_step):
        for param2 in np.arange(param2_min, param2_max + param2_step, param2_step):
            for param3 in np.arange(param3_min, param3_max + param3_step, param3_step):
                curr_params = [param1, param2, param3]
                params.append(curr_params)

                curr_profits = []
                for day in [1, 2, 3]:
                    result = simulate_alternative(rnd_inp, day,
                                                  trader_to_use,
                                                  time_inp,
                                                  names_inp,
                                                  False,
                                                  False,
                                                  curr_params[0],
                                                  curr_params[1],
                                                  curr_params[2])

                    if product == 'CPC':
                        curr_profit = result[time_inp]['COCONUTS'] + result[time_inp]['PINA_COLADAS']
                    elif product == 'BDUP':
                        curr_profit = result[time_inp]['BAGUETTE'] + result[time_inp]['DIP'] + \
                                      result[time_inp]['UKULELE'] + result[time_inp]['PICNIC_BASKET']
                    else:
                        curr_profit = result[time_inp][product]

                    curr_profits.append(curr_profit)

                avg_profit = np.mean(curr_profits)
                profits.append(avg_profit)

    best_params = params[np.argmax(profits)]
    best_profit = np.max(profits)

    print("Best Params: ", best_params)
    print("Max Profit: ", best_profit)

    return best_params, best_profit


# Adjust accordingly the round and day to your needs
if __name__ == "__main__":
    test_multiple_values = bool(input("Do you want to test multiple values (blank for False): ") or False)

    trader_val = Trader()
    time_inp = int(input("Input a timestamp to end (blank for 999000): ") or 999000)
    rnd_inp = int(input("Input a round (blank for 4): ") or 4)
    day_inp = int(input("Input a day (blank for random): ") or random.randint(1, 3))
    names_inp = (not bool(input("With bot names (blank for True): "))) or True
    halfway_inp = bool(input("Matching orders halfway (blank for False): ")) or False

    if test_multiple_values:
        prod_inp = str(input("Input a product: "))
        num_inp = int(input("Input a number of values to test (blank for 3): ") or 3)

        min_inp_1 = float(input("Input a minimum value 1: "))
        max_inp_1 = float(input("Input a maximum value 1: "))
        change_inp_1 = float(input("How much should the value 1 change by (blank for 1): ") or 1.0)

        if num_inp == 1:
            value_optimization_tester_one(trader_val,
                                          prod_inp,
                                          min_inp_1,
                                          max_inp_1,
                                          change_inp_1)
        elif num_inp >= 2:
            min_inp_2 = float(input("Input a minimum value 2: "))
            max_inp_2 = float(input("Input a maximum value 2: "))
            change_inp_2 = float(input("How much should the value 2 change by (blank for 1): ") or 1.0)

            if num_inp == 2:
                value_optimization_tester_two(trader_val,
                                              prod_inp,
                                              min_inp_1,
                                              max_inp_1,
                                              change_inp_1,
                                              min_inp_2,
                                              max_inp_2,
                                              change_inp_2)
            else:
                min_inp_3 = float(input("Input a minimum value 3: "))
                max_inp_3 = float(input("Input a maximum value 3: "))
                change_inp_3 = float(input("How much should the value 3 change by (blank for 1): ") or 1.0)
                value_optimization_tester_three(trader_val,
                                                prod_inp,
                                                min_inp_1,
                                                max_inp_1,
                                                change_inp_1,
                                                min_inp_2,
                                                max_inp_2,
                                                change_inp_2,
                                                min_inp_3,
                                                max_inp_3,
                                                change_inp_3)
    else:
        print(f"Running simulation on round {rnd_inp} day {day_inp} for time {time_inp}")
        print("Remember to change the trader import")
        simulate_alternative(rnd_inp, day_inp, trader_val, time_inp, names_inp, halfway_inp)
