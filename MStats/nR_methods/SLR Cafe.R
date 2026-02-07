#Simple Linear Regression
# Load Libraries
library(tidyverse)
library(mosaic)
library(ggformula)
library(openintro)

options(scipen=99) # if you like, use this to turn off scientific notation
options(scipen=0) # if you like, restores scientific notation to  default: .0001 = 1e-04


# Download the data file called 'cafeData56.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio
CafeData56 <- read_csv("~/Desktop/MStats_Spring26/MStats/CafeData56.csv")

# what are your variable names? how big is the data frame?
names(CafeData56)
View(CafeData56)
dim(CafeData56)

# Correlation
# correlation measure when it is equal to 1 higher,  stronger much linear.
cor(Sales ~ Temperature, data = CafeData56)

# visualize it in a scatter plot
gf_point(Sales ~ Temperature, data = CafeData56, alpha = .5)

cor(Sales ~ Number_of_Customers, data = CafeData56)
gf_point(Sales ~ Number_of_Customers, data = CafeData56, alpha = .5)

# SLR
# building a model by assigning it to a character, Lm () linear regression model
tempmodel <- lm(Sales ~ Temperature, data = CafeData56)
# visualize the model using gf_point. negative, positive significant or not
gf_point(Sales ~ Temperature, data = CafeData56, alpha = .5) %>% gf_lm()
# summarize is runned on model not on the data, 
msummary(tempmodel)

# sales = intercept +- coefficent *temperature,  it is siginificant if p is less, strenght is r2

#model, which plots 1 and 2, residual vs fitted q-q. curvature. 
mplot(tempmodel, which = 1:2, add.smooth = FALSE)

custmodel <- lm(Sales ~ Number_of_Customers, data = CafeData56)
gf_point(Sales ~ Number_of_Customers, data = CafeData56, alpha = .5) %>% gf_lm()
msummary(custmodel)
mplot(custmodel, which = 1:2, add.smooth = FALSE) 

# your work goes below

cor(Advertising_Spend ~ Temperature , data = CafeData56)
gf_point(Temperature ~ Number_of_Customers, data = CafeData56, alpha = .5)


advmodel <- lm(Sales ~ Advertising_Spend, data = CafeData56)
gf_point(Sales ~ Advertising_Spend, data = CafeData56, alpha = .5) %>% gf_lm()
msummary(advmodel)
mplot(advmodel, which = 1:2, add.smooth = FALSE) 
