# IMC Prosperity

---
## Ideas:
* Avellaneda & Stoikov Market-Making Strategy
* Counter-Party Tracking / Insights

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

#### Picnic Products:
   * 1x PICNIC_BASKET =
      * 2x BAGUETTE
      * 4x DIP
      * 1x UKULELE
   * Statistical Arbitrage:
     * Buying Combined Cheaper => Sell Each Separately
     * Buying Separate Cheaper => Sell Combined
---
## Backtesting Hyper-Parameter Optimization Results:

#### BANANAS:
- Round 4, Day 1:
  * Best set of hyper-parameters:  [1.3188359288875715, 9710]
  * Best objective function value:  -1080.0
- Round 4, Day 2:
  * Best set of hyper-parameters:  [1.9382348151126094, 4067]
  * Best objective function value:  -676.0
- Round 4, Day 3:
  * Best set of hyper-parameters:  [0.1386060490952874, 861]
  * Best objective function value:  -1010.0

#### CPC:
- Round 4, Day 1:
  * Best set of hyper-parameters:  [0.5802234262316439, 1.2285591499542938, 961]
  * Best objective function value:  -342728.0
- Round 4, Day 2:
  * Best set of hyper-parameters:  [0.1, 0.644863499540242, 500]
  * Best objective function value:  -399616.0
- Round 4, Day 3:
  * Best set of hyper-parameters:  [0.1, 0.36020125424758764, 500]
  * Best objective function value:  -373980.0

#### DIVING_GEAR:
- Round 4, Day 1:
  * Best set of hyper-parameters:  [1, 4245, 3424]
  * Best objective function value:  -14252.0
- Round 4, Day 2:
  * Best set of hyper-parameters:  [2, 3050, 6532]
  * Best objective function value:  -48454.0
- Round 4, Day 3:
  * Best set of hyper-parameters:  [3, 2786, 3759]
  * Best objective function value:  -8075.0

#### BDUP:
- Round 4, Day 1:
  * Best set of hyper-parameters:  [78113]
  * Best objective function value:  0.0
- Round 4, Day 2:
  * Best set of hyper-parameters:  [419]
  * Best objective function value:  -439566.0
- Round 4, Day 3:
  * Best set of hyper-parameters:  [396]
  * Best objective function value:  -303060.0