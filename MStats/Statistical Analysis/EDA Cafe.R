# Exploratory Dta Analysis (EDA)
# Load Libraries
library(tidyverse)
library(mosaic)
library(ggformula)
library(openintro)

# Download the data file called 'CafeData1.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio


# what are the variable names in this data set?
names(CafeData1)

# Take a peak at the data set...
View(CafeData1)

# Check out the size of the  data set ...
dim(CafeData1)

# Exploring categorical variables
# What types of coffee do people buy?
tally(~coffeeType, data = CafeData1) # counts by category
tally(~coffeeType, data = CafeData1, format = 'percent') # percents by category
tally(~coffeeType, data = CafeData1) %>% prop.table() %>% addmargins() %>% round(digits = 2) # Sum rows and columns, round to  digits

# Visualize coffee type as a count and a percentage.
gf_bar(~coffeeType, data = CafeData1)
gf_percents(~coffeeType, data = CafeData1)

gf_percents(~fct_infreq(coffeeType), data = CafeData1) # sorted by size

# Does type of coffee ordered vary by service period?
tally(coffeeType ~ period, data = CafeData1) # counts by category
tally(coffeeType ~ period, data = CafeData1, format = 'percent') # percents by category
tally(coffeeType ~ period, data = CafeData1) %>% prop.table() %>% addmargins() %>% round(digits = 2) # Sum rows and columns, round to  digits

gf_bar(~ coffeeType, data = CafeData1, fill = ~period) # fill argument splits the bars up by that categories in that variable
gf_bar(~ coffeeType, data = CafeData1, fill = ~period, position = "fill") # standardizes the bars to all be 100%
gf_bar(~ coffeeType, data = CafeData1, fill = ~period, position = "dodge") # creates staggered bars

# Exploring numerical variables
# What do customers spend on coffee, other food, or in total?
favstats(~coffeeSale, data = CafeData1)
favstats(~otherSale, data = CafeData1)
favstats(~totalSale, data = CafeData1)

# Does coffee price differ based on type of coffee? 
favstats(~coffeeSale | coffeeType, data = CafeData1)
favstats(~otherSale | coffeeType, data = CafeData1)
favstats(~totalSale | coffeeType, data = CafeData1)

# Visualize coffee, food, and total sales using histograms and boxplots
gf_histogram(~coffeeSale, data = CafeData1) # default bins = 30
gf_histogram(~coffeeSale, data = CafeData1, bins = 6) # smooth the shape with a bin argument

gf_boxplot(coffeeSale ~ "", data = CafeData1)
gf_boxplot(coffeeSale ~ coffeeType, data = CafeData1)

gf_histogram(~otherSale, data = CafeData1) # default bins = 30
gf_histogram(~otherSale, data = CafeData1, bins = 6) # smooth the shape with a bin argument

gf_boxplot(otherSale ~ "", data = CafeData1)
gf_boxplot(otherSale ~ coffeeType, data = CafeData1)

gf_histogram(~totalSale, data = CafeData1) # default bins = 30
gf_histogram(~totalSale, data = CafeData1, bins = 12) # smooth the shape with a bin argument

gf_boxplot(totalSale ~ "", data = CafeData1)
gf_boxplot(totalSale ~ coffeeType, data = CafeData1)


## Your Work Goes Below
