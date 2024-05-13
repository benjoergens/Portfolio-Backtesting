# Portfolio-Backtesting

This project uses Tiingo's historical prices API to pull 5 years of px data for ~14k index ETFs and stocks.

Sorting them into 12 sectors (GICS + an index ETF sector), it then backtests a portfolio that allocates equally to each sector, and equally to each ticker within each sector. The portfolio re-balances on the first business day of each month (60 re-balancings).

The potfolio's 5yr backtest is then plotted and analyzed using QuantStats (protfolio_report.html), and a csv file listing 5yr cumulative returns for each ticker in the porfolio is compiled.

Additional details about the poject methodology:
- Prices pulled from Tiingo are adjusted close
- Penny stocks (any stocks with values <$5 for threshold of 22 days) are excluded due to high vol. distortions
