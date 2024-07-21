import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.cbook
import zipfile
import bz2
import warnings
warnings.filterwarnings('ignore',category=matplotlib.MatplotlibDeprecationWarning)
data = pd.read_csv("C:\\Users\\LENOVO\\OneDrive\\Desktop\\shri\\Indicators.bz2")
print("Data Shape: ",data.shape)
print("Sample Data : \n",data.head())
print("Columns: \n",data.columns)

countries = data['CountryName'].unique().tolist()
print('Number of countries: ',len(countries))

countryCodes = data['CountryCode'].unique().tolist()
print('Number of country codes: ',len(countryCodes))

indicators = data['IndicatorName'].unique().tolist()
print("Number of indicators: ",len(indicators))

years = data['Year'].unique().tolist()
print('Number of years: ',len(years))

print(min(years), max(years))

# Data Visualization
#Let us pick a country and an indicator to explore CO2 Emissions per capita and the USA.
#To select CO2 emissions for the United States, We will take the intersection
# of two masks, one with all the rows that contains the string,
# "C02 emissions" and the other which contains all the rows containing the string, "USA".

hist_indicator = 'CO2 emissions \(metric'
hist_country = 'USA'
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['CountryCode'].str.contains(hist_country)
stage = data[mask1 & mask2]

# stage dataset contain indicators matching the USA for country code & CO2 emissions over time.
print(stage.shape)
stage.head()
print("Indicator Name: ",stage['IndicatorName'].iloc[0])
print('Country Code: ',stage['CountryCode'].iloc[1])

#Let us see how emissions have changed over time using MatplotLib
years = stage['Year'].values
co2 = stage['Value'].values
# Plot the histogram
plt.bar(years,co2)
#plt.show()

#It is seen that emissions per capita have dropped a bit over time,
# but let us make this graph a bit more appealing before we continue to explore it.
#Let us create a line plot
plt.plot(years, co2)
# Label the axes
plt.xlabel('Year')
plt.ylabel(stage['IndicatorName'].iloc[0])
# label the Figure
plt.axis([1959, 2011,0,25])
#plt.show()

#Using Histograms to explore the distribution of values
#We could also visualize this data as a histogram to better explore
# the ranges of values in CO2 production per year.
# If we want to just include those within one standard deviation fo the mean, you could do the following
# lower = stage['Value'].mean() - stage['Value'].std()
# upper = stage['Value'].mean() + stage['Value'].std()
# hist_data = [x for x in stage[:10000]['Value'] if x>lower and x<upper ]
# Otherwise, let's look at all the data

hist_data = stage['Value'].values
print(hist_data)
print(len(hist_data))

# Histogram of the data
plt.hist(hist_data, 10,density=False,facecolor='green') # 10 is the number of the bins
plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of Years')
plt.title('Histogram Example')
plt.grid(True)
#plt.show()

#USA has many years where it produced between 19-20 metric tons per capita with outliers on either side.
#But how do the USA's numbers relate to those of other countries?
# select CO2 emissions for all countries in 2011

hist_indicator = 'CO2 emissions \(metric'
hist_year = 2011
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['Year'].isin([hist_year])

# apply our mask
co2_2011 = data[mask1 & mask2]
co2_2011.head()
# #For how many countries do we have CO2 per capita emissions data in 2011

print(len(co2_2011))

# Let us plot a histogram of the emmissions per capita by country
# subplots returns a touple with the figure, axis attributes.
fig, ax = plt.subplots()
ax.annotate("USA",xy=(18,5),xycoords='data',xytext=(18,30),textcoords='data',
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3"))
plt.hist(co2_2011['Value'],10,density=False,facecolor='green')
plt.xlabel(stage['IndicatorName'].iloc[0])
plt.ylabel('# of countries')
plt.title('Histogram of CO2 Emission Per Capita')
#plt.show()

#USA, at ~18 CO2 emissions (metric tons per capital) is quite high among allcountries.
#3. Matplotlib: Basic Plotting Part 2
#Relationship between GPD and CO2 Emissions in USA
# Select GDP Per capita emissions for the United States

hist_indicator = 'GDP per capita \(constant 2005'
hist_country = "USA"
mask1 = data['IndicatorName'].str.contains(hist_indicator)
mask2 = data['CountryCode'].str.contains(hist_country)

# Stage is just those indicators matching the USA for country code and CO2 emissions over time.
gdp_stage = data[mask1 & mask2]

# plot gdp_stage vs stage
print('GDP: ',gdp_stage.head())
stage.head()
# switch to a line plot
plt.plot(gdp_stage['Year'].values, gdp_stage['Value'].values)
# Label the axes
plt.xlabel('Year')
plt.ylabel(gdp_stage['IndicatorName'].iloc[0])
# label the figure
plt.title("GDP per capita USA")
#plt.show()

#Although we have seen a decline in the CO2 emissions per capita,
# it does not seem to translate to a decline in GDP per capita
#ScatterPlot for comparing GDP against CO2 emissions (per capita)
#First, we will need to make sure we are looking at the same time frames.

print("GDP Min Year = ",gdp_stage['Year'].min(),"max: ",gdp_stage['Year'].max())
print("CO2 Min Year: ",stage['Year'].min(),'max: ',stage['Year'].max())

#We have 3 extra years of GDP data, so let's trim those off so the scatterplot
# has equal length arrays to compare (this is actually required by scatterplot)

gdp_stage_trunc = gdp_stage[gdp_stage['Year'] < 2012]
print(len(gdp_stage_trunc))
print(len(stage))

fig, axis = plt.subplots()
# Grid lines, Xticks, Xlabel, Ylabel
axis.yaxis.grid(True)
axis.set_title('CO2 Emission vs GDP (per capita)')
axis.set_xlabel(gdp_stage_trunc['IndicatorName'].iloc[0])
axis.set_ylabel(stage['IndicatorName'].iloc[0])
X = gdp_stage_trunc['Value']
Y = stage['Value']
axis.scatter(X, Y)
#plt.show()

#This does not look like a strong relationship. We can test this by looking at correlation.
print("Correlation between GDP and CO2 Emission",np.corrcoef(gdp_stage_trunc['Value'],stage['Value']))
