## Ideas:
* Avellaneda & Stoikov Market-Making Strategy 

#### ALL: (Requires Back-Testing)
  * Find optimal history amount
  * Find optimal thresholds

#### Pair Trading:
  * Make sure the actual amount available matches returned values for _amount variables
  * Make sure the amount doesn't mess with the ratio
---
## Implemented:
#### PEARLS:
   * Simple Arbitrage (Buy Below 10,000 and Sell Above 10,000)

#### BANANAS:
   * Determine Price Direction Based on Order Book:
        * Imbalance Ratio (Buy Vol / Sell Vol)
            * If > 1 => BUY
            * If < 1 => SELL
        * On-Balance Volume (OBV)
            * Current Gradient > Average => BUY
            * Current Gradient < Average => SELL

#### COCONUTS and PINA COLADAS:
   * Pairs Trading

#### BERRIES:
   * Time-Based Trading

#### DIVING GEAR:
  * Track DOLPHIN SIGHTINGS, trade based on that:
    * Bollinger Bands Mean Reversion to decide when to buy and sell