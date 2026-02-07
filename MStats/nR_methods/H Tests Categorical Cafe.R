#Hypothesis Tests Categorical Data
# Load Libraries
library(tidyverse)
library(mosaic)
library(ggformula)
library(openintro)

options(scipen=99) # if you like, use this to turn off scientific notation
options(scipen=0) # if you like, restores scientific notation to  default: .0001 = 1e-04

# Download the data file called 'CafeData4.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio
CafeData4 <- read_csv("~/Desktop/MStats_Spring26/MStats/CafeData4.csv")

# what are your variable names? how big is the data frame?
names(CafeData4)
dim(CafeData4)

# How frequently do people come to the cafe?  
tally(~visitFrequency, data = CafeData4, format="percent")
prop(~visitFrequency, data = CafeData4, success = "Monthly")

prop(~visitFrequency, data = CafeData4, success = "First-time")

tally(~howHeardAboutCafe, data = CafeData4, format="percent")

prop(~howHeardAboutCafe, data = CafeData4, success ="Flyers")
tally(~age, data = CafeData4, format = "percent")


# More than 25% of customers visit at least once a month - One sample z-test of the proportion 
prop.test(~visitFrequency, data = CafeData4, success = "Monthly", correct=FALSE, p = .25,  alternative = "greater")

prop.test(~visitFrequency, data = CafeData4, success = "Daily", correct=FALSE, p = .20,  alternative = "less")

prop.test(~howHeardAboutCafe, data = CafeData4, success = "Newspaper", correct=FALSE, p = .25,  alternative = "greater")


# What percentage of all customers are daily customers? 1 sample Confidence Interval
binom.test(~socialMediaPlatform, data=CafeData4, success = "Twitter", ci.method = "Wald", conf.level = .90)

binom.test(~visitFrequency, data=CafeData4, success = "Weekly", ci.method = "Wald", conf.level = .95)


# What percentage of customers have a satisfaction rating at or above 7?
binom.test(~averageSpendPerVisit <= 20, data=CafeData4, success = TRUE, ci.method = "Wald", conf.level = .95)

# What percentage of customers have a satisfaction rating of 10?
binom.test(~customerSatisfactionRating == 10, data=CafeData4, success = TRUE, ci.method = "Wald", conf.level = .95)

# Is there a difference between percent of referrals if someone attends a special event? Two-sample z-test of proportions
tally(referredOthers~attendedSpecialEvents, data = CafeData4) %>% prop.table(margin = 2)
prop.test(referredOthers~attendedSpecialEvents, data = CafeData4, correct=FALSE, alternative = "two.sided")


# is there a relationship between customer occupation and how often they visit?  Chi-square test of independence
tally(occupation ~ visitFrequency, data = CafeData4)
xchisq.test(occupation ~ visitFrequency, data = CafeData4)

# Is there a relationship between referral behavior and gender? Chi-sq test of independence
tally(responseToPromotions ~ socialMediaPlatform, data = CafeData4)
xchisq.test(responseToPromotions ~ socialMediaPlatform, data = CafeData4)

# What about a relationship between referral behavior and rating scores? chi-square test of independence 
tally(referredOthers ~ customerSatisfactionRating > 7, data = CafeData4)
xchisq.test(referredOthers ~ customerSatisfactionRating > 7, data = CafeData4)

# What about a relationship between referral behavior and rating scores? chi-square test of independence 
tally(attendedSpecialEvents ~ numberOfReferralsMade > 2, data = CafeData4)
xchisq.test(attendedSpecialEvents ~ numberOfReferralsMade < 2, data = CafeData4)


\# your work goes below