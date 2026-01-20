Beverton-Holt Model Investigation:
- [`Beverton-Holt.py`](Beverton-Holt.py)
  
The Beverton-Holt Model determines the population of the next generation to equal (the proliferation rate X current population)/(1+ (Population/M)) where M = carrying capacity/ proliferation rate -1
This is a typical logistic growth model where as the population aproaches carrying capacity growth slows. In order to adjust for finance I used the following calculation.
I set the financial carrying capacity to equal Market cap + (Market cap X 1-RSI/100) except when RSI is greater then the RSI limits for which then the carrying capacity equals market cap - (market cap X RSI/100) (results yielded that RSI limit was an unnescesary extreme case response). This uses RSI as a proxy for whether the stock is under or over sold, preferably a novel indicator which is more effective would be used as this use of RSI introduces some possible inaccuracy.
I then for the rest of the equation set the proliferation rate to equal a rolling geometric average mean over a certain time period and the current population to equate to the current market cap.
So the reultant full equation is as follows:

((1+rolling mean geometric returns)^H x current market cap)/ 1+ market cap / ((financial carrying capacity)/geo mean daily returns) 

where H equals the horizon in number of days for when the code predicts the price.


Ricker Model Investigation:
- [`Ricker.py`](Ricker.py)

The ricker model outlines the predicted population at t+1 to be the current population X e^proliferation rate X (1-(population/carrying capacity))
I simply substituted my previous definitions for a this such that the equation is:
Market cap t+1 = Market cap X e^rolling geometric mean X (1-(market cap/financial carrying capacity))
The utility of this is it is more capable at predicting price decreases even with historical positive returns compared to the Beverton-Holt model.
Despite this it performed much worse.


Hassell Model Investigation:
- [`Hassell.py`](Hassell.py)

The Hassell model dictates that population for t+1 = (proliferation rate X current population)/(1+current population/M)^C
This is the same as the beverton-holt model except the denominator is raised to the power of C. C controls the density dependence where lower C is more optimistic for future growth as higher C indicates factors that slow population growth when population grows have more of an effect.
I used 4 different definitions for C:
  1. I set C as a given constant
  2. C = (universe market cap - specific stock's market cap)/Universe market cap
         - This means as the stock has a greater share of the universe the competiive                 indice is lowered since there is less competition as it controls more of the               market
  3. C = 1-CPI
         - As inflation rates are greater there is greater population increases as there is            decreased monetary competition
  4. C = 1-Buffet Indicator
         - This indicates more money in stock market means lower competition so lower C

All methods try to quantify monetary competition using this biological context as a vector to do so. As access to money increases C decreases creating more optimistic growth predictions.

Self Made Logistic Growth Model:
- [`K_adjusted_R.py`](K_adjusted_R.py)
