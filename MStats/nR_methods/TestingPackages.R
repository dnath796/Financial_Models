# Test your packages
# Put your cursor on the line of code as described in the quiz 
# and use Ctrl/Cmd + Enter to run the appropriate line of code.

# Test tidyverse
library(tidyverse)
filter(mpg, cty >= 20)

# Test mosaic
library(mosaic)
tally(~manufacturer, data = mpg)

# Test ggformula
library(ggformula)
gf_bar(~class, data = mpg)

# Test openintro
library(openintro)
tail(solar)

# Test Data
# Download the data file called 'Colors.csv' to your MStats folder on your laptop 
# do not open the file in any other application like Excel or Numbers; simply download the file, then move it to your MStats Folder).
# Navigate to the lower right hand pane called Files
# Click on the File Name, and select the Import Dataset...option
# The Import Text Data panel will open. 
# You should see a preview of the data set showing three columns of data
# Select the Import button in the lower right corner of that panel.
