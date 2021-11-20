import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import statsmodels.api as sm
import seaborn as sea
from scipy import stats
from scipy.stats import norm


# read both energy datasets and merge them in to one
df1 = pd.read_csv('electricity_generation_source.csv')
df1.head(5)
print(df1.head)
df1.info()
df2 = pd.read_csv('energy_consumption_source.csv')
energy_df = pd.concat([df1,df2], axis=1)
energy_df.info()

# final merged energy data set
energy_df.to_csv('energy_data.csv')


# read merged energy dataset
energy_df = pd.read_csv('energy_data.csv')
print(energy_df.dtypes)
print(energy_df.head)

print(energy_df.corr())

# calculate and print descriptive statistics
print(energy_df.describe())
stats_df = energy_df.describe()
stats_df.to_csv('stats_data.csv')

# code for printing all data in a column
pd.set_option("display.max_rows", None, "display.max_columns", None)

# below are data frames for world energy sources
country_gen = energy_df['Entity-Gen']
country_con = energy_df['Entity-Con']
year_gen = energy_df['Year-Gen']
year_con = energy_df['Year-Con']
world_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Oil electricity per capita (kWh)" ]
world_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Oil per capita (kWh)" ]
w_year_gen = energy_df.loc[energy_df["Entity-Gen"]== 'World', "Year-Gen"]
w_year_con = energy_df.loc[energy_df["Entity-Con"]== 'World', "Year-Con"]
w_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Coal electricity per capita (kWh)" ]
w_gas_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Gas electricity per capita (kWh)" ]
w_oil_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Oil electricity per capita (kWh)" ]
w_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Nuclear electricity per capita (kWh)" ]
w_hydro_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Hydro electricity per capita (kWh)" ]
w_wind_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Wind electricity per capita (kWh)" ]
w_solar_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Solar electricity per capita (kWh)" ]
w_other_gen = energy_df.loc[energy_df["Entity-Gen"] == 'World', "Other renewable electricity per capita (kWh)" ]
w_coal_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Coal per capita (kWh)" ]
w_gas_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Gas per capita (kWh)" ]
w_oil_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Oil per capita (kWh)" ]
w_nuclear_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Nuclear per capita (kWh)" ]
w_hydro_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Hydro per capita (kWh)" ]
w_wind_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Wind per capita (kWh)" ]
w_solar_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Solar per capita (kWh)" ]
w_other_con = energy_df.loc[energy_df["Entity-Con"] == 'World', "Other renewables per capita (kWh)" ]

# below code is for plotting box plots for 5 variables
sea.boxplot(w_coal_con)
sea.boxplot(w_gas_con)
sea.boxplot(w_oil_con)
sea.boxplot(w_nuclear_con)
sea.boxplot(w_hydro_con)

# below code is for creating scatter plot for world oil and gas
plt.scatter(w_oil_con, w_gas_con)

# data frames of fossil fuels for top 5 energy producing countries
us_wind_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Wind electricity per capita (kWh)"]
us_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Coal electricity per capita (kWh)"]
us_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Coal electricity per capita (kWh)"]
china_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'China', "Coal electricity per capita (kWh)"]
russia_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Russia', "Coal electricity per capita (kWh)"]
sa_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Saudi Arabia', "Coal electricity per capita (kWh)"]
canada_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Coal electricity per capita (kWh)"]
us_coal_gen_year = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Year-Gen"]
china_coal_gen_year = energy_df.loc[energy_df["Entity-Gen"] == 'China', "Year-Gen"]
russia_coal_gen_year = energy_df.loc[energy_df["Entity-Gen"] == 'Russia', "Year-Gen"]
sa_coal_gen_year = energy_df.loc[energy_df["Entity-Gen"] == 'Saudi Arabia', "Year-Gen"]
canada_coal_gen_year = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Year-Gen"]

# below are plots for comparing energy sources from the top 5 energy producing contries
plt.plot(us_coal_gen_year, us_coal_gen, label="US")
plt.plot(china_coal_gen_year, china_coal_gen, label="China")
plt.plot(russia_coal_gen_year, russia_coal_gen, label="Russia")
plt.plot(sa_coal_gen_year, sa_coal_gen, label="Saudi Arabia")
plt.plot(canada_coal_gen_year, canada_coal_gen, label="Canada")
plt.legend()
plt.title('Electricity from Coal in Top 5 Energy Countries')
plt.xlabel('Year')
plt.ylabel('kwh')


# data frames of green energy for top 5 energy producing countries
us_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Nuclear electricity per capita (kWh)"]
china_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'China', "Nuclear electricity per capita (kWh)"]
russia_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Russia', "Nuclear electricity per capita (kWh)"]
sa_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Saudi Arabia', "Nuclear electricity per capita (kWh)"]
canada_nuclear_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Nuclear electricity per capita (kWh)"]

# below are plots for comparing green energy sources of the top 5 energy producing contries
plt.plot(us_coal_gen_year, us_nuclear_gen, label="US")
plt.plot(china_coal_gen_year, china_nuclear_gen, label="China")
plt.plot(russia_coal_gen_year, russia_nuclear_gen, label="Russia")
plt.plot(sa_coal_gen_year, sa_nuclear_gen, label="Saudi Arabia")
plt.plot(canada_coal_gen_year, canada_nuclear_gen, label="Canada")
plt.legend()
plt.title('Electricity from Nuclear in Top 5 Energy Countries')
plt.xlabel('Year')
plt.ylabel('kwh')


# below are US and China coal data frames
us_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Coal electricity per capita (kWh)"]
china_coal_gen = energy_df.loc[energy_df["Entity-Gen"] == 'China', "Coal electricity per capita (kWh)"]

# below is a scatter plot comparing china coal and world coal
plt.scatter(china_coal_gen, w_coal_gen)
plt.xlabel('China Coal')
plt.ylabel('World Coal')
plt.title('Coal Electricity Generation')

# r squared, pvalue and t-test for china coal and world coal
r2, pv = scipy.stats.pearsonr(china_coal_gen, w_coal_gen)
print(r2)
print(pv)
dif_means = stats.ttest_ind(china_coal_gen, w_coal_gen)
print(dif_means)

# r squared, pvalue and t-test for us coal and world coal
r, p = scipy.stats.pearsonr(us_coal_gen, w_coal_gen)
print(r)
print(p)
dif_means2 = stats.ttest_ind(us_coal_gen, w_coal_gen)
print(dif_means2)


# regression analysis of world oil consumption
x = w_year_con
y = w_oil_con
model = sm.OLS(y, x).fit()
predictions = model.predict(x)
model.summary()
print(model.summary())
sea.regplot(x,y)
print(predictions)



# regression analysis of world oil electricity generation
x = w_year_gen
y = w_oil_gen
model = sm.OLS(y,x).fit()
predictions = model.predict(x)
model.summary()
print(model.summary())
sea.regplot(x,y)


# scatter plot for us coal electricity and world coal electricity
plt.scatter(us_coal_gen, w_coal_gen)
plt.xlabel('US Coal')
plt.ylabel('World Coal')
plt.title('Coal Electricity Generation')

# strong positive correlation plot between world and china coal generation
plt.scatter(china_coal_gen, w_coal_gen)

# pdf and cdf of world oil
count1, bins_count1 = np.histogram(w_oil_gen, bins='auto')
w_oil_pdf = count1/sum(count1)
w_oil_cdf = np.cumsum(w_oil_pdf)
plt.plot(bins_count1[1:], w_oil_cdf, label='Oil')

# pd and cdf of world wind
count2, bins_count2 = np.histogram(w_wind_gen, bins='auto')
w_wind_pdf = count2/sum(count2)
w_wind_cdf = np.cumsum(w_wind_pdf)
plt.plot(bins_count1[1:], w_wind_cdf, label='Wind')
plt.title('CDF Comparison')
plt.ylabel('CDF')
plt.xlabel('kwh')
plt.legend()
plt.plot(w_year_gen, world_gen, color='r')
plt.ylabel('kwh')
plt.xlabel('Year')


# plot world electricity generation from all fuel sources
plt.plot(w_year_gen, w_coal_gen, label="Coal")
plt.plot(w_year_gen, w_gas_gen, label="Gas")
plt.plot(w_year_gen, w_oil_gen, label="Oil")
plt.plot(w_year_gen, w_nuclear_gen, label="Nuclear")
plt.plot(w_year_gen, w_hydro_gen, label="Hydro")
plt.plot(w_year_gen, w_wind_gen, label="Wind")
plt.plot(w_year_gen, w_solar_gen, label="Solar")
plt.plot(w_year_gen, w_other_gen, label="Other")
plt.legend()



# plot world energy consumption of all fuel sources
plt.plot(w_year_con, w_coal_con, label="Coal")
plt.plot(w_year_con, w_gas_con, label="Gas")
plt.plot(w_year_con, w_oil_con, label="Oil")
plt.plot(w_year_con, w_nuclear_con, label="Nuclear")
plt.plot(w_year_con, w_hydro_con, label="Hydro")
plt.plot(w_year_con, w_wind_con, label="Wind")
plt.plot(w_year_con, w_solar_con, label="Solar")
plt.plot(w_year_con, w_other_con, label="Other")
plt.legend()


# data frames for electricity source variables
coal_gen = energy_df['Coal electricity per capita (kWh)']
gas_gen = energy_df['Gas electricity per capita (kWh)']
oil_gen = energy_df['Oil electricity per capita (kWh)']
nuclear_gen = energy_df['Nuclear electricity per capita (kWh)']
hydro_gen = energy_df['Hydro electricity per capita (kWh)']
wind_gen = energy_df['Wind electricity per capita (kWh)']
solar_gen = energy_df['Solar electricity per capita (kWh)']
other_gen = energy_df['Other renewable electricity per capita (kWh)']

# data frames for consumption source variables
coal_con = energy_df['Coal per capita (kWh)']
gas_con = energy_df['Oil per capita (kWh)']
oil_con = energy_df['Gas per capita (kWh)']
nuclear_con = energy_df['Nuclear per capita (kWh)']
hydro_con = energy_df['Hydro per capita (kWh)']
wind_con = energy_df['Wind per capita (kWh)']
solar_con = energy_df['Solar per capita (kWh)']
other_con = energy_df['Other renewables per capita (kWh)']


# us oil electricity generation data frame and plots
us_oil_gen = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Oil electricity per capita (kWh)" ]
print(us_oil.head)
sea.distplot(us_oil_gen, fit=norm, kde=False)
us_oil_gen_years = energy_df.loc[energy_df["Entity-Gen"] == 'United States', "Year-Gen"]
plt.plot(us_oil_gen_years, us_oil_gen, color='r')

# us oil consumption data frame and plots
us_oil_con = energy_df.loc[energy_df["Entity-Con"] == 'United States', "Oil per capita (kWh)"]
us_oil_con_years = energy_df.loc[energy_df["Entity-Con"] == 'United States', "Year-Con"]
plt.plot(us_oil_con_years, us_oil_con, color='b', label='US Oil')
plt.ylabel('kwh')

# canada oil dataframe and plots
canada_oil_gen = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Oil electricity per capita (kWh)" ]
print(us_oil.head)
sea.distplot(us_oil, fit=norm, kde=False)
canada_oil_gen_years = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Year-Gen"]
plt.plot(us_oil_gen_years, us_oil_gen, color='r')
canada_oil_con = energy_df.loc[energy_df["Entity-Con"] == 'Canada', "Oil per capita (kWh)"]
canada_oil_con_years = energy_df.loc[energy_df["Entity-Con"] == 'Canada', "Year-Con"]
plt.plot(canada_oil_con_years, canada_oil_con, color='r', label='Canadian Oil')
plt.xlabel('Year')
plt.title('Energy Consumption from Oil')
plt.legend()


# us oil histogram and cdf
count, bins_count = np.histogram(us_oil, bins='auto')
us_pdf = count/sum(count)
us_cdf = np.cumsum(us_pdf)
plt.plot(bins_count[1:], us_cdf, label='us oil')


# us oil PMF
fig, ax = plt.subplots()
plt.title('Blue is US Oil & Red is Canadian Oil')
sea.histplot(us_oil, stat='probability', bins="auto", color='b')
ax2 = ax.twinx()


# canadian oil dataframe
canada_oil = energy_df.loc[energy_df["Entity-Gen"] == 'Canada', "Oil electricity per capita (kWh)" ]
print(canada_oil.head)



# canadian oil histogram and cdf
count2, bins_count2 = np.histogram(canada_oil, bins='auto')
canada_pdf = count2/sum(count2)
canada_cdf = np.cumsum(canada_pdf)
plt.plot(bins_count[1:], canada_cdf, label='canadian oil')
plt.title('CDF Comparison')
plt.ylabel('CDF')
plt.xlabel('kwh')
plt.legend()


# canadian oil PMF
sea.histplot(canada_oil, stat='probability', bins="auto", color='r')
sea.plt.show()

# below are histograms for variables
plt.hist(year_gen)
plt.title('Years for Energy Generation Data')

plt.hist(year_con)
plt.title('Years for Energy Consumption Data')

plt.hist(coal_gen)
plt.title('Coal Generation Histogram')

plt.hist(gas_gen)
plt.title('Gas Generation Histogram')

plt.hist(oil_gen)
plt.title('Oil Generation Histogram')

plt.hist(nuclear_gen)
plt.title('Nuclear Generation Histogram')

plt.hist(hydro_gen)
plt.title('Hydro Generation Histogram')

plt.hist(wind_gen)
plt.title('Wind Generation Histogram')

plt.hist(solar_gen)
plt.title('Solar Generation Histogram')

plt.hist(other_gen)
plt.title('Other Renewables Generation Histogram')

plt.hist(coal_con)
plt.title('Coal Consumption Histogram')

plt.hist(gas_con)
plt.title('Gas Consumption Histogram')

plt.hist(oil_con)
plt.title('Oil Consumption Histogram')

plt.hist(nuclear_con)
plt.title('Nuclear Consumption Histogram')

plt.hist(hydro_con)
plt.title('Hydro Consumption Histogram')

plt.hist(wind_con)
plt.title('Wind Consumption Histogram')

plt.hist(solar_con)
plt.title('Solar Consumption Histogram')

plt.hist(other_con)
plt.title('Other Renewables Consumption Histogram')

plt.show()