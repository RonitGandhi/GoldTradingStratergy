# import pandas as pd
# import yfinance as yf
# from pytrends.request import TrendReq
# import matplotlib.pyplot as plt
# import seaborn as sns

# def fetch_gold_etf_data(ticker='GLD', start_date='2018-01-01', end_date='2024-01-01'):
#     """
#     Fetch historical ETF price data from Yahoo Finance
    
#     :param ticker: ETF ticker symbol (default is SPDR Gold Shares)
#     :param start_date: Start date for data collection
#     :param end_date: End date for data collection
#     :return: DataFrame with date and adjusted close prices
#     """
#     etf_data = yf.download(ticker, start=start_date, end=end_date)
    
#     # Convert index to datetime without timezone and reset index
#     etf_data.index = pd.to_datetime(etf_data.index).tz_localize(None)
#     return etf_data[['Adj Close']].reset_index()

# def fetch_google_trends_data(keyword='gold', start_date='2018-01-01', end_date='2024-01-01'):
#     """
#     Fetch Google Trends interest data
    
#     :param keyword: Search term to analyze
#     :param start_date: Start date for data collection
#     :param end_date: End date for data collection
#     :return: DataFrame with date and search interest
#     """
#     # Initialize pytrends
#     pytrends = TrendReq(hl='en-US', tz=360)
    
#     # Convert dates to proper format
#     pytrends.build_payload([keyword], timeframe=f'{start_date} {end_date}')
    
#     # Get interest over time
#     trends_data = pytrends.interest_over_time()
    
#     # Reset index to make date a column and convert to datetime without timezone
#     trends_data = trends_data.reset_index()
#     trends_data['date'] = pd.to_datetime(trends_data['date']).dt.tz_localize(None)
    
#     return trends_data

# def analyze_correlation(etf_data, trends_data):
#     """
#     Merge ETF and Google Trends data and compute correlation
    
#     :param etf_data: DataFrame with ETF prices
#     :param trends_data: DataFrame with Google Trends data
#     :return: Correlation coefficient and scatter plot
#     """
#     # Rename columns for clarity
#     etf_data.columns = ['Date', 'ETF_Price']
#     trends_data.columns = ['Date', 'Search_Interest', 'isPartial']
    
#     # Merge datasets on date
#     merged_data = pd.merge(etf_data, trends_data[['Date', 'Search_Interest']], on='Date', how='inner')
    
#     # Compute correlation
#     correlation = merged_data['ETF_Price'].corr(merged_data['Search_Interest'])
    
#     # Create scatter plot
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=merged_data, x='Search_Interest', y='ETF_Price')
#     plt.title(f'Gold ETF Prices vs Search Interest\nCorrelation: {correlation:.4f}')
#     plt.xlabel('Google Trends Search Interest')
#     plt.ylabel('ETF Adjusted Close Price')
#     plt.tight_layout()
#     plt.savefig('gold_correlation_plot.png')
#     plt.close()
    
#     return correlation, merged_data

# def main():
#     # Fetch data
#     etf_data = fetch_gold_etf_data()
#     trends_data = fetch_google_trends_data()
    
#     # Analyze correlation
#     correlation, merged_data = analyze_correlation(etf_data, trends_data)
    
#     print(f"Correlation between Gold ETF Prices and Search Interest: {correlation:.4f}")
    
#     # Optional: Save merged data for further analysis
#     merged_data.to_csv('gold_etf_trends_data.csv', index=False)

# if __name__ == '__main__':
#     main()


    ###############################################################################
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

class GoldCorrelationAnalysis:
    def __init__(self, ticker='GLD', keyword='gold', start_date='2018-01-01', end_date='2024-01-01'):
        """
        Initialize the correlation analysis with ETF and Google Trends data
        """
        self.ticker = ticker
        self.keyword = keyword
        self.start_date = start_date
        self.end_date = end_date
        
        # Fetch data
        self.etf_data = self._fetch_gold_etf_data()
        self.trends_data = self._fetch_google_trends_data()
        
        # Merged dataset
        self.merged_data = None
    
    def _fetch_gold_etf_data(self):
        """
        Fetch historical ETF price data from Yahoo Finance
        
        :return: DataFrame with date and various price metrics
        """
        try:
            # Download data
            etf_data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
            
            # Handle multi-level column index from yfinance
            if isinstance(etf_data.columns, pd.MultiIndex):
                # Flatten column names
                etf_data.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in etf_data.columns]
            
            # Convert index to datetime without timezone
            etf_data.index = pd.to_datetime(etf_data.index).tz_localize(None)
            
            # Reset index
            df = etf_data.reset_index()
            
            # Rename columns for consistency
            rename_dict = {
                'index': 'Date',
                'Adj Close_GLD': 'Close_Price'
            }
            df = df.rename(columns=rename_dict)
            
            # Calculate daily returns
            df['Daily_Return'] = df['Close_Price'].pct_change()
            
            return df
        
        except Exception as e:
            print(f"Error fetching ETF data: {e}")
            raise
    
    def _fetch_google_trends_data(self):
        """
        Fetch Google Trends interest data
        
        :return: DataFrame with date and search interest
        """
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Convert dates to proper format
        pytrends.build_payload([self.keyword], timeframe=f'{self.start_date} {self.end_date}')
        
        # Get interest over time
        trends_data = pytrends.interest_over_time()
        
        # Reset index and convert timezone
        trends_data = trends_data.reset_index()
        trends_data['date'] = pd.to_datetime(trends_data['date']).dt.tz_localize(None)
        
        return trends_data
    
    def merge_datasets(self):
        """
        Merge ETF and Google Trends data
        
        :return: Merged DataFrame
        """
        # Rename columns for consistency
        etf_data = self.etf_data.copy()
        trends_data = self.trends_data.copy()
        
        trends_data.columns = ['Date', 'Search_Interest', 'isPartial']
        
        # Merge datasets
        self.merged_data = pd.merge(etf_data, trends_data[['Date', 'Search_Interest']], 
                                    on='Date', how='inner')
        
        return self.merged_data
    
    def perform_correlation_analysis(self):
        """
        Perform comprehensive correlation analysis
        
        :return: Dictionary of correlation metrics
        """
        if self.merged_data is None:
            self.merge_datasets()
        
        # Prepare correlation metrics
        correlation_metrics = {}
        
        # Pearson Correlation
        correlation_metrics['pearson_correlation'] = {
            'price_vs_search': self.merged_data['Close_Price'].corr(self.merged_data['Search_Interest']),
            'return_vs_search': self.merged_data['Daily_Return'].corr(self.merged_data['Search_Interest'])
        }
        
        # Spearman Rank Correlation (less sensitive to outliers)
        correlation_metrics['spearman_correlation'] = {
            'price_vs_search': self.merged_data['Close_Price'].corr(self.merged_data['Search_Interest'], method='spearman'),
            'return_vs_search': self.merged_data['Daily_Return'].corr(self.merged_data['Search_Interest'], method='spearman')
        }
        
        # Statistical Significance (p-value)
        price_correlation, price_pvalue = stats.pearsonr(self.merged_data['Close_Price'], self.merged_data['Search_Interest'])
        return_correlation, return_pvalue = stats.pearsonr(self.merged_data['Daily_Return'].dropna(), 
                                                           self.merged_data['Search_Interest'].dropna())
        
        correlation_metrics['statistical_significance'] = {
            'price_correlation_pvalue': price_pvalue,
            'return_correlation_pvalue': return_pvalue
        }
        
        return correlation_metrics
    
    def visualize_correlations(self):
        """
        Create visualizations for correlation analysis
        """
        if self.merged_data is None:
            self.merge_datasets()
        
        # Create a multi-panel figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{self.ticker} Price vs Google Trends Search Interest', fontsize=16)
        
        # Scatter plot: Price vs Search Interest
        sns.scatterplot(data=self.merged_data, x='Search_Interest', y='Close_Price', ax=axes[0, 0])
        axes[0, 0].set_title('Price vs Search Interest')
        
        # Scatter plot: Returns vs Search Interest
        sns.scatterplot(data=self.merged_data, x='Search_Interest', y='Daily_Return', ax=axes[0, 1])
        axes[0, 1].set_title('Returns vs Search Interest')
        
        # Line plot: Price over time
        ax1 = axes[1, 0]
        ax1.plot(self.merged_data['Date'], self.merged_data['Close_Price'], color='blue', label='Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Close Price', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Overlay search interest on secondary y-axis
        ax2 = ax1.twinx()
        ax2.plot(self.merged_data['Date'], self.merged_data['Search_Interest'], color='red', label='Search Interest')
        ax2.set_ylabel('Search Interest', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Price and Search Interest Over Time')
        
        # Correlation heatmap
        correlation_matrix = self.merged_data[['Close_Price', 'Daily_Return', 'Search_Interest']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[1, 1], center=0)
        axes[1, 1].set_title('Correlation Heatmap')
        
        plt.tight_layout()
        plt.savefig('gold_correlation_analysis.png')
        plt.close()
    
    def run_full_analysis(self):
        """
        Run complete correlation analysis and generate outputs
        """
        # Merge datasets
        self.merge_datasets()
        
        # Perform correlation analysis
        correlation_results = self.perform_correlation_analysis()
        
        # Create visualizations
        self.visualize_correlations()
        
        # Print detailed results
        print("Correlation Analysis Results:")
        print("\nPearson Correlation:")
        print(f"Price vs Search Interest: {correlation_results['pearson_correlation']['price_vs_search']:.4f}")
        print(f"Returns vs Search Interest: {correlation_results['pearson_correlation']['return_vs_search']:.4f}")
        
        print("\nSpearman Rank Correlation:")
        print(f"Price vs Search Interest: {correlation_results['spearman_correlation']['price_vs_search']:.4f}")
        print(f"Returns vs Search Interest: {correlation_results['spearman_correlation']['return_vs_search']:.4f}")
        
        print("\nStatistical Significance (p-values):")
        print(f"Price Correlation p-value: {correlation_results['statistical_significance']['price_correlation_pvalue']:.4f}")
        print(f"Returns Correlation p-value: {correlation_results['statistical_significance']['return_correlation_pvalue']:.4f}")
        
        # Save merged data for further analysis
        self.merged_data.to_csv('gold_etf_trends_detailed_data.csv', index=False)
        
        return correlation_results

def main():
    # Create analysis instance
    analysis = GoldCorrelationAnalysis(
        ticker='GLD',  # SPDR Gold Shares ETF
        keyword='gold',
        start_date='2020-01-01',
        end_date='2024-08-31'
    )
    
    # Run full analysis
    results = analysis.run_full_analysis()

if __name__ == '__main__':
    main()