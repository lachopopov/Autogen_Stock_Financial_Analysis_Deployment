import yfinance as yf
import json
from typing import Dict, Any, Optional

class FinanceTools:
    """Handles financial data fetching and analysis tools."""
    
    @staticmethod
    def finance_data_fetch(ticker: str, period: str = "1mo") -> str:
        """
        Fetches financial data for a given ticker.
        
        Args:
            ticker: Stock ticker symbol.
            period: Historical period (e.g., '1mo').
        
        Returns:
            JSON string with data or error.
        """
        if not ticker or not isinstance(ticker, str):
            return json.dumps({"error": "Invalid ticker provided."})
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if hist.empty:
                return json.dumps({"error": f"No historical data for {ticker}."})
            converted_hist = {key: {str(k): v for k, v in value.items()} for key, value in hist.to_dict().items()}
            info = stock.info
            
            summary = {
                "name": info.get("shortName", ticker),
                "symbol": ticker,
                "currentPrice": info.get("currentPrice"),
                "currency": info.get("currency", "USD"),
                "summary": info.get("longBusinessSummary", ""),
                "marketCap": info.get("marketCap"),
                "peRatio": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "dividend": info.get("dividendRate"),
                "recentClosePrices": converted_hist.get("Close", {})
            }
            return json.dumps({"name": "finance_data_fetch", "data": summary})
        except Exception as e:
            return json.dumps({"error": f"Failed to fetch data for {ticker}: {str(e)}"})

    @staticmethod
    def technical_analysis_tool(ticker: str) -> str:
        """
        Performs technical analysis on a ticker.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            JSON string with analysis or error.
        """
        if not ticker or not isinstance(ticker, str):
            return json.dumps({"error": "Invalid ticker provided."})
        try:
            data = yf.Ticker(ticker).history(period="3mo")
            if data.empty:
                return json.dumps({"error": f"No data for technical analysis on {ticker}."})
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['RSI'] = 100 - (100 / (1 + rs))

            summary = {
                "SMA_20": data['SMA_20'].dropna().iloc[-1] if not data['SMA_20'].dropna().empty else None,
                "EMA_20": data['EMA_20'].dropna().iloc[-1] if not data['EMA_20'].dropna().empty else None,
                "RSI": data['RSI'].dropna().iloc[-1] if not data['RSI'].dropna().empty else None,
                "Last_Close": data['Close'].iloc[-1]
            }
            return json.dumps({"name": "technical_analysis_tool", "data": summary})
        except Exception as e:
            return json.dumps({"error": f"Technical analysis failed for {ticker}: {str(e)}"})

    @staticmethod
    def risk_assessment_tool(ticker: str) -> str:
        """
        Assesses risk for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            JSON string with assessment or error.
        """
        if not ticker or not isinstance(ticker, str):
            return json.dumps({"error": "Invalid ticker provided."})
        try:
            info = yf.Ticker(ticker).info
            beta = info.get("beta")
            summary = {
                "Beta": beta,
                "MarketCap": info.get("marketCap"),
                "DividendYield": info.get("dividendYield"),
                "Volatility": info.get("52WeekChange"),
                "RiskRating": "High" if beta and beta > 1.2 else "Moderate" if beta and beta > 0.8 else "Low"
            }
            return json.dumps({"name": "risk_assessment_tool", "data": summary})
        except Exception as e:
            return json.dumps({"error": f"Risk assessment failed for {ticker}: {str(e)}"})

    @staticmethod
    def strategy_signal_tool(ticker: str) -> str:
        """
        Generates strategy signals for a ticker.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            JSON string with signals or error.
        """
        if not ticker or not isinstance(ticker, str):
            return json.dumps({"error": "Invalid ticker provided."})
        try:
            data = yf.Ticker(ticker).history(period="6mo", interval="1d")
            if data.empty:
                return json.dumps({"error": f"No data for strategy signals on {ticker}."})
            close = data['Close']
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))

            summary = {
                "MACD": macd.dropna().iloc[-1] if not macd.dropna().empty else None,
                "MACD_Signal": signal.dropna().iloc[-1] if not signal.dropna().empty else None,
                "RSI": rsi.dropna().iloc[-1] if not rsi.dropna().empty else None,
                "Last_Close": close.iloc[-1]
            }
            return json.dumps({"name": "strategy_signal_tool", "data": summary})
        except Exception as e:
            return json.dumps({"error": f"Strategy signal analysis failed for {ticker}: {str(e)}"})

    @staticmethod
    def get_stock_metrics(ticker: str) -> str:
        """
        Gets key stock metrics.
        
        Args:
            ticker: Stock ticker symbol.
        
        Returns:
            JSON string with metrics or error.
        """
        if not ticker or not isinstance(ticker, str):
            return json.dumps({"Error": "Invalid ticker provided."})
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            metrics = {
                "Current Price": f"${info.get('currentPrice', 'N/A'):.2f}" if info.get('currentPrice') else "N/A",
                "Market Cap": f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A",
                "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A",
                "Beta": f"{info.get('beta', 'N/A'):.2f}" if info.get('beta') else "N/A"
            }
            return json.dumps(metrics)
        except Exception as e:
            return json.dumps({"Error": f"Could not fetch metrics: {str(e)}"})