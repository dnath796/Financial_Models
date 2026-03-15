#Final Exam

# Load Libraries

options(scipen=99) # if you like, use this to turn off scientific notation
options(scipen=0) # if you like, restores scientific notation to  default settings: 0.0001 = 1e-04

# you will be using data sets that are in the Mosaic and OpenIntro Packages  
# Verify that you have these packages loaded properly by following the instructions below

# Execute each code line below and verify you get the designated results
dim(Whickham)  # 1314 cases in 3 columns of data
head(Whickham) # first data row should be Alive, Yes, 23
dim(oscars)  # 184 cases in 11 columns of data
tail(oscars) # last data row should show Rami Malek as The 2019 Best Actor

names(RailTrail)
# Your Final Exam Code Goes Below


favstats(~ volume | dayType, data = RailTrail)


gf_histogram(~ volume, data = RailTrail, bins = 10) 

names(movies)
tally(~ rating , data = movies)
tally(rating ~ genre , data = movies, format = 'percent')%>% round(digits = 2)
tally(genre ~ rating , data = movies, format = 'percent')%>% round(digits = 2)


gf_bar(~ rating, data = movies, fill = ~genre, position = "fill") # standardizes the bars to all be 100%


xpnorm(118000, mean = 116000, sd = 16500, lower.tail = FALSE)
xpnorm(95000, mean = 116000, sd = 16500)

xqnorm(.35, mean = 116000, sd = 16500, lower.tail = FALSE)%>% round(digits = 2)
xqnorm(.40, mean = 116000, sd = 16500)

xcnorm(.40, mean = 116000, sd = 16500)


dim(RailTrail)

t.test(~volume, data = RailTrail, mu = 380, alt = "two.sided")
 favstats(~volume, data = RailTrail)
 
 t.test(~volume, data = RailTrail, mu = 345, alt = "greater")
 
 t.test(volume ~ dayType, data = RailTrail, conf.level = .95)

 precip
 t.test(precip ~ dayType, data = RailTrail, conf.level = .95)
 
 names(reddit_finance)

 
 prop(~ pan_ret_date_chg, data = reddit_finance, success = "No change")
  
 prop.test(~pan_ret_date_chg, data = reddit_finance, success = "No change", correct=FALSE, p = .74,  alternative = "greater")
 
 prop(reddit_finance[["pan_ret_date_chg"]], success = "No_change", correct = FALSE, p = .75, alternative = "greater")

 
 150000
 t.test(~retire_exp, data=reddit_finance, mu = 150000, alt = "greater")
 
 
 binom.test(~retire_exp, data=reddit_finance, success = "150000", ci.method = "Wald", conf.level = .95)
 
 RailTrail$volume_high <- RailTrail$volume > 300
 
 tally(dayType ~ volume > 300, data = RailTrail)
 xchisq.test(dayType ~ volume > 300, data = RailTrail)
  
 names(starbucks)
 dim(starbucks) 
 cor(carb ~ fat, data = starbucks )

 gf_point(fiber ~ fat, data = starbucks, alpha = .5)
 
 cor (fiber ~ fat, data = starbucks )
 
 starmodel <- lm(calories ~ protein, data = starbucks )
 gf_point(calories ~ protein, data = starbucks , alpha = .5) %>% gf_lm()
 msummary(starmodel) 

 
 
 custadvmodel <- lm(calories ~ fat + carb + fiber, data = starbucks)
 gf_point(calories ~ fat, data = starbucks, alpha = .5) %>% gf_lm()
 gf_point(Sales ~ Number_of_Customers, data = CafeData56, alpha = .5) %>% gf_lm()
 msummary(custadvmodel)
 mplot(custadvmodel, which = 1:2, add.smooth = FALSE) 
 
 
 custadvmodel <- lm(calories ~ fat + carb, data = starbucks)
 msummary(custadvmodel)

cor (select(starbucks, calories , fat , carb , fiber, protein ), use = "complete")  %>% round(digits = 4)
 