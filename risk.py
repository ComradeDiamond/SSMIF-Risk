import pandas as pd
import math

class CompareMetrics():

   def __init__(self, portfolio = 'sp500prices.pkl'):
      self.portfolio = pd.read_pickle(portfolio)
      self.daily_returns = self.__calc_daily_returns()
      self.daily_values = self.__calc_daily_values()
      self.myMetric = self.__my_metric()
      self.sharpeRatio = self.__sharpe_ratio()
      self.compareMetrics = self.__compare_metrics()
      self.correlationCoefficient = self.__get_correlation_coefficient()

   """
   Calculates the daily returns of a stock
   This is done so we can just keep it somewhere and not have to recalculate it every time we run __my_metric and __sharpe_ratio
   @returns {float[]} An array of daily returns. This shares the same index as the dataframe iloc indeces
   """
   def __calc_daily_returns(self):
      
      dailyReturns = []
       # A dict that contains the last portfolio stock prices we used because there are NANs. Stores it by their seriesIdx
      lastValues = dict()           

      # First, we iterate through each day that the portfolio contains
      for idx in range(len(self.portfolio.index)):
         row = self.portfolio.iloc[idx]

         # Keep track of the total rate of changes (or total expected returns) for the day. We'll use this to calculate expected return
         totalROC = 0
         
         # Since some rows have NAN, we use numCounted to track how many values we actually "counted". This way, we just completely ignore NAN
         numCounted = 0

         # The seriesIdx corresponds to the value of different stocks on the portfolio date
         for seriesIdx in range(len(row.index)):
            prevValue = lastValues.get(seriesIdx)

            # If we don't have a base price yet
            if (prevValue == None):
               if (not math.isnan(row.iloc[seriesIdx])):
                  lastValues[seriesIdx] = row.iloc[seriesIdx]
                  numCounted += 1
            else:
               if (not math.isnan(row.iloc[seriesIdx])):
                  # If we have an init price, we can calculate the rate of change (percent change)
                  lastValues[seriesIdx] = row.iloc[seriesIdx]
                  totalROC += ((row.iloc[seriesIdx] - prevValue) / prevValue)
                  numCounted += 1
         # Since all portfolio items are equal weight, we can the totalROC / numCounted to get expected value of portfolio
         dailyReturns.append(totalROC / numCounted)
      
      return dailyReturns

   """
   @returns {float[]} The daily value of the portfolio (all stocks combined)
   The iloc pandaframe index matches with this float[]
   """
   def __calc_daily_values(self):
      return self.portfolio.sum(axis=1).tolist()

   """
   Part 1: Define your metric 
   This function calculates the burke ratio of the portfolio in __init__
   Formula for burke ratio:
   @param daysin {int} Calculates the (hopefully annual sharpe ratio) $daysin days from the most recent day in sp500prices
   @param adjustedScale {int} The time period to use for the Burke ratio, INCLUDING THE DATE IN $daysin
   It's set to 252 by default as there's 252 trading days in a year
   https://breakingdownfinance.com/finance-topics/performance-measurement/burke-ratio/
   """
   def __my_metric(self, daysin = 0, adjustedScale = 252) -> float:
      adjustedScale = 252
      riskFreeReturnRate = 0.0426 / 252 * adjustedScale     # Based on the 10 year treasury rate
      returns = 0 # Placeholder 0

      # First, let's find the index of the "date" we're implementing the burke ratio on.
      burkeRatioIdx = len(self.portfolio.index) - 1 - daysin
      targetDate = self.portfolio.iloc[burkeRatioIdx]

      # We have enough data to calculate portfolio returns directly
      if (adjustedScale <= burkeRatioIdx + 1):
         # Get the index of the initial date we're going to perform Burke ratio on
         initDateIdx = burkeRatioIdx - adjustedScale + 1
         initialDate = self.portfolio.iloc[initDateIdx]

         # Iterate through every stock in the portfolio
         numCounted = 0          # Since NAN exists in some data
         cumReturns = 0          # Cumulative returns
         for seriesIdx in range(len(self.portfolio.iloc[initDateIdx].index)):
            if (not (math.isnan(targetDate.iloc[seriesIdx]) or math.isnan(initialDate.iloc[seriesIdx]))):
               numCounted += 1
               cumReturns += (targetDate.iloc[seriesIdx] - initialDate.iloc[seriesIdx]) / initialDate.iloc[seriesIdx]
         
         returns = cumReturns / numCounted         # All stocks are weighed the same, so we can divide by the percent return we aggregated
      else:
         # We don't have enough data to calculate portfolio returns directly.
         # Calculate the average daily portfolio return and then try to annualize it
         
         relevantDailyRets = self.daily_returns[0: burkeRatioIdx + 1]      

         # Now we can find the mean
         avgDailyRet = sum(relevantDailyRets) / len(relevantDailyRets)

         # Calculate the expected period returns. 
         returns = ((1 + avgDailyRet) ** adjustedScale) - 1

      # Second step - subtract the returns from risk free rate
      rpMinusrf = returns - riskFreeReturnRate

      # Third step - calculate drawdown of each element. It's the percent difference between peak value and current value.

      # This is the relevant sum and max of all portfolio values
      sumValues = self.daily_values[0: burkeRatioIdx + 1]
      maximum = max(sumValues)

      # Now calculate the square of the drawdowns.
      drawdowns = [(xi - maximum) / maximum for xi in sumValues]
      drawdowns2 = [xi ** 2 for xi in drawdowns]

      # Now, find the sum and square root of drawdowns2
      denom = math.sqrt(sum(drawdowns2))

      # Now you have the burke ratio
      return rpMinusrf / denom

   """
   Part 2: Define the sharpe ratio
   This function calculates the annualized sharpe ration of the portfolio in __init__
   Formula for sharpe:
   1. Find the mean of all daily returns (Returns as in percent change today vs original price)
   2. Divide it by standard deviation of these returns 
   @param daysin {int} Calculates the (hopefully annual sharpe ratio) $daysin days from the most recent day in sp500prices
   @param adjustedScale {int} To scale things, we're going to calculate the daily sharpe ratio and then scale it by sqrt(252) for annual and sqrt(12) for monthly
   # This helped too: https://www.youtube.com/watch?app=desktop&v=_jGi7m8QlkM&ab_channel=Trader%27sCode + the comment that said you needed to divide the risk free rate
   """
   def __sharpe_ratio(self, daysin = 0, adjustedScale=252) -> float:

      # Based on the 10 year treasury rate. Gives RFRR for the adjustedscale period
      # This is because subtracting the daily RFRR every iteration of the mean calculation is the same as just subtracting the period RFRR at the end
      riskFreeReturnRate = 0.0426 / 252 * adjustedScale    
   
      # 1) Find the mean of all daily returns

      # We already calculated daily returns. Now, we just need to select the ones that's relevant to us.
      # Thankfully, daysin exists
      relevantDailyRets = self.daily_returns[0: len(self.daily_returns) - daysin]

      # Now we can find the mean of these returns to get average daily returns and subsequently, average returns in the period
      avgDailyRet = sum(relevantDailyRets) / len(relevantDailyRets)
      avgPeriodRet = ((1 + avgDailyRet) ** adjustedScale) - 1

      # 2) Divide it by the standard deviation of these dailyReturns. Multiply it for the stdev of the period
      sumDiff2 = sum([(xi - avgDailyRet) ** 2 for xi in relevantDailyRets])
      stdev = math.sqrt(sumDiff2 / len(relevantDailyRets)) * adjustedScale

      sharpe = (avgPeriodRet - riskFreeReturnRate) / stdev

      return sharpe

   """
   Part 3: Evaluate Metric Correlation
   Changes the $daysin number in order to find compare the Sharpe and Burke ratios against different dates/time periods
   TLDR; generate a side by side of annualized Sharpe and Burke ratios
   """
   def __compare_metrics(self) -> pd.DataFrame:
      numGenerate = range(1, 800, 15)     

      data = {
         "Sharpe": [],
         "Burke": []
      }

      for i in numGenerate:
         data["Sharpe"].append(self.__sharpe_ratio(daysin=i))
         data["Burke"].append(self.__my_metric(daysin=i))

      return pd.DataFrame(data, index=numGenerate)

   """
   Calculates the correlation coefficient between the Burke Ratio and the Sharpe Ratio.
   This puts the Sharpe Ratio on the "x" and the Burke on the "y"
   Returns it as a float
   """
   def __get_correlation_coefficient(self) -> float:
      xValues = self.compareMetrics["Sharpe"]
      yValues = self.compareMetrics["Burke"]

      print(xValues.tolist())
      print(yValues.tolist())

      averageX = xValues.mean()
      averageY = yValues.mean()

      sumdXdY = 0
      sumdX2 = 0
      sumdY2 = 0
      for i in range(len(xValues.index)):
         sumdXdY += (xValues.iloc[i] - averageX) * (yValues.iloc[i] - averageY)
         sumdX2 += (xValues.iloc[i] - averageX) ** 2
         sumdY2 += (yValues.iloc[i] - averageY) ** 2
      
      r = sumdXdY / math.sqrt(sumdX2 * sumdY2)
      return r
       
if __name__ == '__main__':

   sp500 = CompareMetrics() 
   myMetric = "Burke Ratio"

   print("1. " + myMetric + " of the portfolio is: " + str(sp500.myMetric))
   print("2. Sharpe Ratio of the portfolio is: " + str(sp500.sharpeRatio))
   print("3. Metric comparison is: " + '\n' + str(sp500.compareMetrics))
   print("4. Correlation Coefficient between " + myMetric + " and Sharpe Ratio is: " + str(sp500.correlationCoefficient))