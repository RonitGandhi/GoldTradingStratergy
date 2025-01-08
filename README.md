# **Gold Trading Strategies README**

## **Overview**
This project explores and evaluates three trading strategies for gold using Pytrends data, technical indicators, and hybrid approaches. The aim is to optimize trading decisions, improve profitability, and compare these strategies against the traditional **Buy & Hold** benchmark.

---

## **Strategies**

1. **Strategy 1: Pytrends Data Only**
   - Relies solely on trend data (Google Trends) to generate buy and sell signals.
   - **Pros**: Simple and easy to implement.
   - **Cons**: Higher transaction costs due to frequent trades and lower profitability.

2. **Strategy 2: Pytrends + Technical Indicators**
   - Combines Pytrends data with indicators such as RSI, MACD, Bollinger Bands, and SMA/EMA.
   - **Pros**: Improved signals, better returns, and reduced trades.
   - **Cons**: Still underperforms compared to **Buy & Hold** in total returns.

3. **Strategy 3: Hybrid (Buy & Hold + Pytrends)**
   - A mix of **Buy & Hold** and active trading using Pytrends and technical indicators.
   - **Pros**: Best risk-adjusted returns (highest Sharpe ratio) and strong cumulative performance.
   - **Cons**: Marginally less profitable than **Buy & Hold**.

4. **Buy & Hold Benchmark**
   - Passive strategy involving holding gold throughout the period.
   - **Pros**: Highest profitability and simplicity.
   - **Cons**: Full exposure to market volatility.

---

## **Key Metrics**

- **Final Portfolio Value**: Total value at the end of the trading period.
- **Cumulative Return (%)**: Percentage growth of the portfolio.
- **Sharpe Ratio**: Risk-adjusted performance metric.
- **Total Trades**: Number of buy/sell transactions.

---

## **Results**

| **Metric**                | **Strategy 1**    | **Strategy 2**    | **Strategy 3**    | **Buy & Hold**    |
|---------------------------|-------------------|-------------------|-------------------|-------------------|
| **Final Portfolio Value** | $13,856.10       | $15,783.80       | $17,287.07       | $20,131.69       |
| **Cumulative Return (%)** | 38.56%           | 57.84%           | 72.87%           | 101.32%          |
| **Sharpe Ratio**          | 0.48             | 0.65             | 1.28             | 0.70             |
| **Total Trades**          | 27               | 11               | 30               | N/A              |

---

## **How to Use**

1. **Data Preparation**:
   - Collect price data for gold and trend data (e.g., Pytrends).
   - Merge the datasets and calculate technical indicators (RSI, MACD, SMA, etc.).

2. **Run Strategies**:
   - Use provided Python scripts to execute Strategies 1, 2, and 3.
   - Adjust thresholds (e.g., RSI levels, moving averages) to test performance.

3. **Evaluate Performance**:
   - Analyze the metrics (returns, Sharpe ratio, number of trades).
   - Compare against the **Buy & Hold** benchmark.

4. **Visualization**:
   - Examine buy/sell signals plotted on gold prices.
   - View portfolio value and cumulative return comparisons.

---

## **Limitations**
- Assumes no transaction costs or slippage.
- Based on historical data; performance may differ in live markets.
- Thresholds may need periodic adjustment for changing market dynamics.

---

## **Future Work**
- Automate optimization of thresholds for better profitability.
- Explore additional indicators (e.g., momentum, volume).
- Incorporate transaction costs for realistic performance assessment.

---

## **Contact**
For questions or contributions, feel free to reach out.
