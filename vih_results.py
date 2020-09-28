import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import time
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import base64
aids_country= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/homosexual%20aids.csv')
aids_country
aids_country.rename(columns={'Cumulative total**': 'total'}, inplace=True)
aids_country
aids_country_homosexual= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/AIDS_diagnoses_men_infected_by_sex_between_men_by_country_and_year_of_diagnosis_2006%20to%202015.csv')
aids_country_homosexual.rename(columns={'Cumulative total**': 'total'}, inplace=True)
aids_country_homosexual
aids_country_heterosexual= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/AIDS_diagnoses_heterosexual_contact_by%20country_and_year_of_diagnosis_2006%20to%202015.csv')
aids_country_heterosexual
aids_test= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/number_of_HIV_tests_performed_by_country_and_year_2006%20to%202015.csv')
aids_test
aids_test_country= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/HIV_AIDS_deaths_by_geographic_area_country_and_year_of_death_2006%20to%202015.csv')
aids_test_country
aids_mother_to_child= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/AIDS_diagnoses_mother_to_child_transmission_by_country_year_of_diagnosis_2006%20to%202015.csv')
aids_mother_to_child
aids_drug_injection= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/AIDS_diagnoses_injecting_drug_use_by_country_and_year_of_diagnosis_2006%20to%202015.csv')
aids_drug_injection
aids_country_homosexual_2006 = aids_country_homosexual['2006'].dropna()
aids_country_homosexual_2006
aids_country_homosexual_2007 = aids_country_homosexual['2007'].dropna()
aids_country_homosexual_2007
aids_country_homosexual_2008 = aids_country_homosexual['2008'].dropna()
aids_country_homosexual_2008
aids_country_homosexual_2009 = aids_country_homosexual['2009'].dropna()
aids_country_homosexual_2009
aids_country_homosexual_2010 = aids_country_homosexual['2010'].dropna()
aids_country_homosexual_2010
aids_country_homosexual_2011 = aids_country_homosexual['2011'].dropna()
aids_country_homosexual_2011
aids_country_homosexual_2012 = aids_country_homosexual['2012'].dropna()
aids_country_homosexual_2012
aids_country_homosexual_2013 = aids_country_homosexual['2013'].dropna()
aids_country_homosexual_2013
aids_country_homosexual_2014 = aids_country_homosexual['2014'].dropna()
aids_country_homosexual_2014
aids_country_homosexual_2015 = aids_country_homosexual['2015'].dropna()
aids_country_homosexual_2015
aids_country_heterosexual_2006 = aids_country_heterosexual['2006'].dropna()
aids_country_heterosexual_2006
aids_country_heterosexual_2007 = aids_country_heterosexual['2007'].dropna()
aids_country_heterosexual_2007
aids_country_heterosexual_2008 = aids_country_heterosexual['2008'].dropna()
aids_country_heterosexual_2008
aids_country_heterosexual_2009 = aids_country_heterosexual['2009'].dropna()
aids_country_heterosexual_2009
aids_country_heterosexual_2010 = aids_country_heterosexual['2010'].dropna()
aids_country_heterosexual_2010
aids_country_heterosexual_2011 = aids_country_heterosexual['2011'].dropna()
aids_country_heterosexual_2011
aids_country_heterosexual_2012 = aids_country_heterosexual['2012'].dropna()
aids_country_heterosexual_2012
aids_country_heterosexual_2013 = aids_country_heterosexual['2013'].dropna()
aids_country_heterosexual_2013
aids_country_heterosexual_2014 = aids_country_heterosexual['2014'].dropna()
aids_country_heterosexual_2014
aids_country_heterosexual_2015 = aids_country_heterosexual['2015'].dropna()
aids_country_heterosexual_2015
aids_country_homosexual_cumulative_total = aids_country_homosexual['total'].dropna()
aids_country_homosexual_cumulative_total
aids_country_heterosexual_cumulative_total = aids_country_heterosexual['Cumulative total**'].dropna()
aids_country_heterosexual_cumulative_total

aids_mother_to_child_cumulative_total = aids_mother_to_child['Cumulative total**'].dropna()
aids_mother_to_child_cumulative_total

aids_drug_injection_total = aids_drug_injection['Cumulative total**'].dropna()
aids_drug_injection_total

df1=pd.DataFrame({'año': 'Homosexual','total': aids_country_homosexual_cumulative_total.sum()}, index=[0])
df2=pd.DataFrame({'año': 'Heterosexual','total': aids_country_heterosexual_cumulative_total.sum()}, index=[0])
df3=pd.DataFrame({'año': 'Paternidad','total': aids_mother_to_child_cumulative_total.sum()}, index=[0])
df4=pd.DataFrame({'año': 'Inyeccion','total': aids_drug_injection_total.sum()}, index=[0])
plt.figure(figsize=(16, 10))
plt.xticks(size=10)
plt.yticks(size=10)
plt.grid(True)
#df1['hue']=1
#df2['hue']=2
res=pd.concat([df1,df2, df3, df4])
sns.barplot(x='año',y='total',data=res)
plt.title('Total de confirmados entre el 2006 al 2015', size=30)
plt.savefig('comparacion-contagios-vih.png')
##plt.show()

with open("comparacion-contagios-vih.png", "rb") as image_file:
    encoded_string1 = base64.b64encode(image_file.read())

print(encoded_string1)

plt.figure(figsize=(16, 10))
plt.title('Casos por sexo homosexual entre hombres', size=30)
df1=pd.DataFrame({'año': '2006','cantidad de infectados': aids_country_homosexual_2006.sum()}, index=[0])
df2=pd.DataFrame({'año': '2007','cantidad de infectados': aids_country_homosexual_2007.sum()}, index=[0])
df3=pd.DataFrame({'año': '2008','cantidad de infectados': aids_country_homosexual_2008.sum()}, index=[0])
df4=pd.DataFrame({'año': '2009','cantidad de infectados': aids_country_homosexual_2009.sum()}, index=[0])
df5=pd.DataFrame({'año': '2010','cantidad de infectados': aids_country_homosexual_2010.sum()}, index=[0])
df6=pd.DataFrame({'año': '2011','cantidad de infectados': aids_country_homosexual_2011.sum()}, index=[0])
df7=pd.DataFrame({'año': '2012','cantidad de infectados': aids_country_homosexual_2012.sum()}, index=[0])
df8=pd.DataFrame({'año': '2013','cantidad de infectados': aids_country_homosexual_2013.sum()}, index=[0])
df9=pd.DataFrame({'año': '2014','cantidad de infectados': aids_country_homosexual_2014.sum()}, index=[0])
df10=pd.DataFrame({'año': '2015','cantidad de infectados': aids_country_homosexual_2015.sum()}, index=[0])
#df1['hue']=1
#df2['hue']=2
res=pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10])
sns.barplot(x='año',y='cantidad de infectados',data=res)
plt.savefig('comparacion-contagios-gays-vih.png')
##plt.show()

import base64

with open("comparacion-contagios-gays-vih.png", "rb") as image_file:
    encoded_string2 = base64.b64encode(image_file.read())

print(encoded_string2)

plt.figure(figsize=(16, 10))
plt.title('Casos por sexo heterosexual', size=30)
df1=pd.DataFrame({'año': '2006','cantidad de infectados': aids_country_heterosexual_2006.sum()}, index=[0])
df2=pd.DataFrame({'año': '2007','cantidad de infectados': aids_country_heterosexual_2007.sum()}, index=[0])
df3=pd.DataFrame({'año': '2008','cantidad de infectados': aids_country_heterosexual_2008.sum()}, index=[0])
df4=pd.DataFrame({'año': '2009','cantidad de infectados': aids_country_heterosexual_2009.sum()}, index=[0])
df5=pd.DataFrame({'año': '2010','cantidad de infectados': aids_country_heterosexual_2010.sum()}, index=[0])
df6=pd.DataFrame({'año': '2011','cantidad de infectados': aids_country_heterosexual_2011.sum()}, index=[0])
df7=pd.DataFrame({'año': '2012','cantidad de infectados': aids_country_heterosexual_2012.sum()}, index=[0])
df8=pd.DataFrame({'año': '2013','cantidad de infectados': aids_country_heterosexual_2013.sum()}, index=[0])
df9=pd.DataFrame({'año': '2014','cantidad de infectados': aids_country_heterosexual_2014.sum()}, index=[0])
df10=pd.DataFrame({'año': '2015','cantidad de infectados': aids_country_heterosexual_2015.sum()}, index=[0])
#df1['hue']=1
#df2['hue']=2
res=pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10])
sns.barplot(x='año',y='cantidad de infectados',data=res)
plt.savefig('comparacion-contagios-hetero-vih.png')
#plt.show()


with open("comparacion-contagios-hetero-vih.png", "rb") as image_file:
    encoded_string3 = base64.b64encode(image_file.read())

print(encoded_string3)

aids_drug_injection_2006 = aids_drug_injection['2006'].dropna()
aids_drug_injection_2006
aids_drug_injection_2007 = aids_drug_injection['2007'].dropna()
aids_drug_injection_2007
aids_drug_injection_2008 = aids_drug_injection['2008'].dropna()
aids_drug_injection_2008
aids_drug_injection_2009 = aids_drug_injection['2009'].dropna()
aids_drug_injection_2009
aids_drug_injection_2010 = aids_drug_injection['2010'].dropna()
aids_drug_injection_2010
aids_drug_injection_2011 = aids_drug_injection['2011'].dropna()
aids_drug_injection_2011
aids_drug_injection_2012 = aids_drug_injection['2012'].dropna()
aids_drug_injection_2012
aids_drug_injection_2013 = aids_drug_injection['2013'].dropna()
aids_drug_injection_2013
aids_drug_injection_2014 = aids_drug_injection['2014'].dropna()
aids_drug_injection_2014
aids_drug_injection_2015 = aids_drug_injection['2015'].dropna()
aids_drug_injection_2015

aids_mother_to_child_2006 = aids_mother_to_child['2006'].dropna()
aids_mother_to_child_2006
aids_mother_to_child_2007 = aids_mother_to_child['2007'].dropna()
aids_mother_to_child_2007
aids_mother_to_child_2008 = aids_mother_to_child['2008'].dropna()
aids_mother_to_child_2008
aids_mother_to_child_2009 = aids_mother_to_child['2009'].dropna()
aids_mother_to_child_2009
aids_mother_to_child_2010 = aids_mother_to_child['2010'].dropna()
aids_mother_to_child_2010
aids_mother_to_child_2011 = aids_mother_to_child['2011'].dropna()
aids_mother_to_child_2011
aids_mother_to_child_2012 = aids_mother_to_child['2012'].dropna()
aids_mother_to_child_2012
aids_mother_to_child_2013 = aids_mother_to_child['2013'].dropna()
aids_mother_to_child_2013
aids_mother_to_child_2014 = aids_mother_to_child['2014'].dropna()
aids_mother_to_child_2014
aids_mother_to_child_2015 = aids_mother_to_child['2015'].dropna()
aids_mother_to_child_2015

plt.figure(figsize=(16, 10))
plt.title('Contagio por paternidad', size=30)
df1=pd.DataFrame({'año': '2006','cantidad de infectados': aids_mother_to_child_2006.sum()}, index=[0])
df2=pd.DataFrame({'año': '2007','cantidad de infectados': aids_mother_to_child_2007.sum()}, index=[0])
df3=pd.DataFrame({'año': '2008','cantidad de infectados': aids_mother_to_child_2008.sum()}, index=[0])
df4=pd.DataFrame({'año': '2009','cantidad de infectados': aids_mother_to_child_2009.sum()}, index=[0])
df5=pd.DataFrame({'año': '2010','cantidad de infectados': aids_mother_to_child_2010.sum()}, index=[0])
df6=pd.DataFrame({'año': '2011','cantidad de infectados': aids_mother_to_child_2011.sum()}, index=[0])
df7=pd.DataFrame({'año': '2012','cantidad de infectados': aids_mother_to_child_2012.sum()}, index=[0])
df8=pd.DataFrame({'año': '2013','cantidad de infectados': aids_mother_to_child_2013.sum()}, index=[0])
df9=pd.DataFrame({'año': '2014','cantidad de infectados': aids_mother_to_child_2014.sum()}, index=[0])
df10=pd.DataFrame({'año': '2015','cantidad de infectados': aids_mother_to_child_2015.sum()}, index=[0])
#df1['hue']=1
#df2['hue']=2
res=pd.concat([df1,df2, df3, df4, df5, df6, df7, df8, df9, df10])
sns.barplot(x='año',y='cantidad de infectados',data=res)
plt.savefig('comparacion-contagios-paternidad-vih.png')
#plt.show()


with open("comparacion-contagios-paternidad-vih.png", "rb") as image_file:
    encoded_string4 = base64.b64encode(image_file.read())

print(encoded_string4)

aids_country_age= pd.read_csv('https://raw.githubusercontent.com/LEPPEDIAZ/Pandemias_Mundiales/master/Data/vih/new_HIV_diagnoses_by_sex_age_year_of_diagnosis_2006%20to%202015.csv')
aids_country_age

labels = 'Mujeres', 'Hombres', 'Indefinido'
aids_country_age_get = aids_country_age['Cumulative Female'].dropna()
aids_country_age_sum = aids_country_age_get.sum()
aids_country_age2_get = aids_country_age['Cumulative Male'].dropna()
aids_country_age2_sum = aids_country_age2_get.sum()
aids_country_age3_get = aids_country_age['Cumulative Unknown'].dropna()
aids_country_age3_sum = aids_country_age3_get.sum()
sizes = [aids_country_age_sum, aids_country_age2_sum, aids_country_age3_sum ]
explode = (0, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.savefig('comparacion-sexo-vih.png')
#plt.show()

import base64

with open("comparacion-sexo-vih.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read())

print(encoded_string)