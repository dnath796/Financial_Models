#Multiple Linear Regression
# Load Libraries
library(tidyverse)
library(mosaic)
library(ggformula)
library(openintro)

options(scipen=99) # if you like, use this to turn off scientific notation
options(scipen=0) # if you like, restores scientific notation to  default: .0001 = 1e-04


# Download the data file called 'cafeData56.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio
CafeData56 <- read_csv("CafeData56.csv")

# what are your variable names? how big is the data frame?
names(CafeData56)
View(CafeData56)
dim(CafeData56)

# SLR refresh
tempmodel <- lm(Sales ~ Temperature, data = CafeData56)
gf_point(Sales ~ Temperature, data = CafeData56, alpha = .5) %>% gf_lm()
msummary(tempmodel)
mplot(tempmodel, which = 1:2, add.smooth = FALSE)

# MLR models
tempadvmodel <- lm(Sales ~ Temperature + Advertising_Spend, data = CafeData56)
gf_point(Sales ~ Temperature, data = CafeData56, alpha = .5) %>% gf_lm()
gf_point(Sales ~ Advertising_Spend, data = CafeData56, alpha = .5) %>% gf_lm()
msummary(tempadvmodel)
mplot(tempadvmodel, which = 1:2, add.smooth = FALSE)

temprainmodel <- lm(Sales ~ Temperature + Rain, data = CafeData56)
gf_point(Sales ~ Temperature, color = ~Rain, data = CafeData56, alpha = .5)
msummary(temprainmodel)
mplot(temprainmodel, which = 1:2, add.smooth = FALSE)

TRAmodel <- lm(Sales ~ Temperature + Rain + Advertising_Spend, data = CafeData56)
msummary(TRAmodel)
mplot(TRAmodel, which = 1:2, add.smooth = FALSE)


# Correlation Matrix & Pairs Plots
cor(select(CafeData56, Temperature, Advertising_Spend, Sales), use = "complete")
cor(select(CafeData56, Temperature, Advertising_Spend, Sales), use = "complete") %>% round(digits = 2)

pairs(~Temperature + Advertising_Spend + Sales, data = CafeData56)


# your work goes below

custadvmodel <- lm(Sales ~ Advertising_Spend + Number_of_Customers, data = CafeData56)
gf_point(Sales ~ Advertising_Spend, data = CafeData56, alpha = .5) %>% gf_lm()
gf_point(Sales ~ Number_of_Customers, data = CafeData56, alpha = .5) %>% gf_lm()
msummary(custadvmodel)
mplot(tempadvmodel, which = 1:2, add.smooth = FALSE)


drinksmodel <- lm(Sales ~ Advertising_Spend + Number_of_Customers + Special, data = CafeData56)
gf_point(Sales ~ Advertising_Spend, data = CafeData56, alpha = .5) %>% gf_lm()
gf_point(Sales ~ Number_of_Customers, data = CafeData56, alpha = .5) %>% gf_lm()
gf_point(Sales ~ Special, data = CafeData56, alpha = .5) %>% gf_lm()

msummary(drinksmodel)
mplot(tempadvmodel, which = 1:2, add.smooth = FALSE)


cor(select(CafeData56, Temperature, Sales,Advertising_Spend, Number_of_Customers ), use = "complete")  %>% round(digits = 2)
