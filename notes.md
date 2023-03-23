## Ideas:

### COCONUTS AND PINA COLADAS
* Pairs Trading
---
## Implemented:
* PEARLS:
    * Simple Arbitrage (Buy Below 10,000 and Sell Above 10,000)
* BANANAS:
    * Determine Price Direction Based on Order Book:
        * Imbalance Ratio (Buy Vol / Sell Vol)
            * If > 1 => BUY
            * If < 1 => SELL
        * On-Balance Volume (OBV)
            * Current Gradient > Average => BUY
            * Current Gradient < Average => SELL
* COCONUTS and PINA COLADAS:
    * Bollinger Bands Mean Reversion