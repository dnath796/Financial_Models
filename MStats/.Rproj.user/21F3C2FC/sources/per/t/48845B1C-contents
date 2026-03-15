# Hypothesis Tests Numerical Data
# Load Libraries


options(scipen=99) # if you like, use this to turn off scientific notation
options(scipen=0) # if you like, restores scientific notation to  default: .0001 = 1e-04


# Download the data file called 'CafeData3.csv' to your MStats folder on your laptop 
# use the Import Dataset function to open this data set in your RStudio
CafeData3 <- read_csv("E:/RLabs/Managerial Statistics/CafeData3.csv")


# what are your variable names? how big is the data frame?
names(CafeData3)
dim(CafeData3)

# Explore the Measures 
favstats(~waitTime, data = CafeData3) # summary measures for wait time
gf_histogram(~waitTime, data = CafeData3) # visualize wait time 

# People wait about 5 minutes - one sample mean t-test
t.test(~waitTime, data = CafeData3, mu = 5, alt = "two.sided")

# People wait less than 6 minutes - one sample mean t-test
t.test(~waitTime, data = CafeData3, mu = 6, alt = "less")

# Based on the data, how long do people wait? one sample CI
t.test(~waitTime, data = CafeData3, conf.level = .90)


# Are Wait Times Longer when Understaffed?
t.test(waitTime ~ staffing, data = CafeData3, mu = 0, alt = "greater") # check group order!
t.test(waitTime ~ staffing, data = CafeData3, mu = 0, alt = "less")


# Are Wait times different during lunch than other serving periods?
t.test(waitTime ~ timeOfDay, data = CafeData3, mu = 0, alt = "two.sided") # need two groups only
tally(~timeOfDay, data = CafeData3)

t.test(waitTime ~ timeOfDay == "Lunch", data = CafeData3, mu = 0, alt = "two.sided") # Lunch group versus all others

# How big is the difference between Lunch and all other Wait Times?
t.test(waitTime ~ timeOfDay == "Lunch", data = CafeData3, conf.level = .99)


# your work goes below

t.test(serviceTime ~ timeOfDay == "Breakfast", data = CafeData3, conf.level = .80)

t.test(~tableTime, data = CafeData3, conf.level = .95)

t.test(~serviceTime, data = CafeData3, conf.level = .95)

t.test(~serviceTime, data = CafeData3, mu = 12.5, alt = "greater")



