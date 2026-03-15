# Probability & Distributions
# Load Libraries
library(tidyverse)
library(mosaic)
library(ggformula)
library(openintro)


# Download the data file called 'CafeData2.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio

# what are your variable names?
names(CafeData2)

favstats(~totalSale, data = CafeData2) # summary measures for total sale
gf_histogram(~waitTime, data = CafeData2) # what is the shape 
mean(~totalSale, data = CafeData2)
sd(~totalSale, data = CafeData2)

# Finding some probabilities or percentages
xpnorm(32, mean = 30.7949, sd = 5.1785) # what percentage are less than $25?
xpnorm(32, mean = 30.7949, sd = 5.1785, lower.tail = FALSE) # what percentage are more than $32?

xqnorm(.25,mean = 30.7949, sd = 5.1785) # what would be the estimated Q1 value?
xqnorm(.30,mean = 30.7949, sd = 5.1785) # 30% of customer spend less than _____ dollars...
xqnorm(.10,mean = 30.7949, sd = 5.1785, lower.tail = FALSE) # 10% of customers spend more than what value?

xcnorm(.50,mean = 30.7949, sd = 5.1785) # what would be the estimated Q1 and Q3 values?
xcnorm(.75,mean = 30.7949, sd = 5.1785) # 75% of customers spend between ___ and ___ dollars...


favstats(~waitTime, data = CafeData2) # summary measures for total sale

xcnorm(.70,mean = 6.819333, sd = 2.222843)
xqnorm(.65,mean = 6.819333, sd = 2.222843)
xpnorm(6.8,mean = 6.819333, sd = 2.222843, lower.tail = FALSE)

xpnorm(118000, mean = 116000, sd = 16500, lower.tail = FALSE) 

xpnorm(9,mean = 6.819333, sd = 2.222843, , lower.tail = FALSE)

# your work goes below
