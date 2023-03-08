import pandas as pd

class CompareMetrics():

    def __init__(self, portfolio = 'sp500prices.pkl'):
      self.portfolio = pd.read_pickle(portfolio)
      self.myMetric = self.__my_metric()
      self.sharpeRatio = self.__sharpe_ratio()
      self.compareMetrics = self.__compare_metrics()
      self.correlationCoefficient = self.__get_correlation_coefficient()

    """ Part 1: Define your metric """
    # TODO
    def __my_metric(self) -> float:
       pass

    """ Part 2: Define the Sharpe Ratio """
    # TODO
    def __sharpe_ratio(self) -> float:
      pass 

    """ Part 3: Evaluate Metric Correlation """
    # TODO
    def __compare_metrics(self) -> pd.DataFrame:
       pass

    # TODO
    def __get_correlation_coefficient(self) -> float:
       pass
       
if __name__ == '__main__':

   sp500 = CompareMetrics() 
   # TODO: enter the name of the metric you chose
   myMetric = ""

   print("1. " + myMetric + " of the portfolio is: " + str(sp500.myMetric))
   print("2. Sharpe Ratio of the portfolio is: " + str(sp500.sharpeRatio))
   print("3. Metric comparison is: " + '\n' + str(sp500.compareMetrics))
   print("4. Correlation Coefficient between " + myMetric + " and Sharpe Ratio is: " + str(sp500.correlationCoefficient))