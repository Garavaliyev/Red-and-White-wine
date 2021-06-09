
# Case study 1 from Udacity
Our data set is about red and white wine, we have diferrent values such that:acidity,citric_acid ,residual_sugar,chlorides,sulfur-dioxide,density,pH,sulphates,alcohol, quality.
We made different operation in Jupyter Notebook: analyzing data, cleaning data and making visualizations. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
import seaborn as sns
# At first, we need to read the csv file in Jupyter, we have two separate data which are the data of red and white wine.
# for differentiating of data we named df_red and df_white, and we used the sep to add a separator between strings. 
df_red=pd.read_csv('winequality-red.csv', sep=';')
df_white=pd.read_csv('winequality-white.csv',sep=';')
# We utilized head() function to check first five index 
df_red.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur-dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_white.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>




```python
# to get brief information about the data we used info function, 
# here we can see the number of samples, rows, the size of data and the datatypes etc.
df_red.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
    fixed_acidity           1599 non-null float64
    volatile_acidity        1599 non-null float64
    citric_acid             1599 non-null float64
    residual_sugar          1599 non-null float64
    chlorides               1599 non-null float64
    free_sulfur_dioxide     1599 non-null float64
    total_sulfur-dioxide    1599 non-null float64
    density                 1599 non-null float64
    pH                      1599 non-null float64
    sulphates               1599 non-null float64
    alcohol                 1599 non-null float64
    quality                 1599 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB



```python
df_white.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4898 entries, 0 to 4897
    Data columns (total 12 columns):
    fixed_acidity           4898 non-null float64
    volatile_acidity        4898 non-null float64
    citric_acid             4898 non-null float64
    residual_sugar          4898 non-null float64
    chlorides               4898 non-null float64
    free_sulfur_dioxide     4898 non-null float64
    total_sulfur_dioxide    4898 non-null float64
    density                 4898 non-null float64
    pH                      4898 non-null float64
    sulphates               4898 non-null float64
    alcohol                 4898 non-null float64
    quality                 4898 non-null int64
    dtypes: float64(11), int64(1)
    memory usage: 459.3 KB



```python
# it was asked the number of duplicated values from the data of white wine, we used sum(df_white.duplicated()) function to find it.
# if it needs to delete the duplicated values, we will use drop() function. 
sum(df_white.duplicated())
```




    937




```python
# After all next step is that how many unique value in each column there are, 
# we found data below with using nunique() function.
df_red.nunique()
```




    fixed_acidity            96
    volatile_acidity        143
    citric_acid              80
    residual_sugar           91
    chlorides               153
    free_sulfur_dioxide      60
    total_sulfur-dioxide    144
    density                 436
    pH                       89
    sulphates                96
    alcohol                  65
    quality                   6
    dtype: int64




```python
df_white.nunique()
```




    fixed_acidity            68
    volatile_acidity        125
    citric_acid              87
    residual_sugar          310
    chlorides               160
    free_sulfur_dioxide     132
    total_sulfur_dioxide    251
    density                 890
    pH                      103
    sulphates                79
    alcohol                 103
    quality                   7
    dtype: int64




```python
# Next question is that what are averages of each columns in the data of red wine.
# We used mean() function to get this result.
df_red.mean()
```




    fixed_acidity            8.319637
    volatile_acidity         0.527821
    citric_acid              0.270976
    residual_sugar           2.538806
    chlorides                0.087467
    free_sulfur_dioxide     15.874922
    total_sulfur-dioxide    46.467792
    density                  0.996747
    pH                       3.311113
    sulphates                0.658149
    alcohol                 10.422983
    quality                  5.636023
    dtype: float64




```python
# we need to use rename function to turn into the same name,
# because when we append the data then we will see the incorrect values.
df_red.rename(columns={'total_sulfur-dioxide':'total_sulfur_dioxide'}, inplace=True)
```


```python
# create color array for red and white  dataframe
color_red = np.repeat('red', df_red.shape[0])

color_white = np.repeat('white', df_white.shape[0])
```


```python
# we want to add column to the table, which called color in both data.
df_red['color'] = color_red
df_red.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_white['color'] = color_white
df_white.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.0</td>
      <td>0.27</td>
      <td>0.36</td>
      <td>20.7</td>
      <td>0.045</td>
      <td>45.0</td>
      <td>170.0</td>
      <td>1.0010</td>
      <td>3.00</td>
      <td>0.45</td>
      <td>8.8</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6.3</td>
      <td>0.30</td>
      <td>0.34</td>
      <td>1.6</td>
      <td>0.049</td>
      <td>14.0</td>
      <td>132.0</td>
      <td>0.9940</td>
      <td>3.30</td>
      <td>0.49</td>
      <td>9.5</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>0.28</td>
      <td>0.40</td>
      <td>6.9</td>
      <td>0.050</td>
      <td>30.0</td>
      <td>97.0</td>
      <td>0.9951</td>
      <td>3.26</td>
      <td>0.44</td>
      <td>10.1</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
      <td>white</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.32</td>
      <td>8.5</td>
      <td>0.058</td>
      <td>47.0</td>
      <td>186.0</td>
      <td>0.9956</td>
      <td>3.19</td>
      <td>0.40</td>
      <td>9.9</td>
      <td>6</td>
      <td>white</td>
    </tr>
  </tbody>
</table>
</div>




```python
# after all we append dataframes
wine_df = df_red.append(df_white)

# we check the dataframe to control our data
wine_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6497 entries, 0 to 4897
    Data columns (total 13 columns):
    fixed_acidity           6497 non-null float64
    volatile_acidity        6497 non-null float64
    citric_acid             6497 non-null float64
    residual_sugar          6497 non-null float64
    chlorides               6497 non-null float64
    free_sulfur_dioxide     6497 non-null float64
    total_sulfur_dioxide    6497 non-null float64
    density                 6497 non-null float64
    pH                      6497 non-null float64
    sulphates               6497 non-null float64
    alcohol                 6497 non-null float64
    quality                 6497 non-null int64
    color                   6497 non-null object
    dtypes: float64(11), int64(1), object(1)
    memory usage: 710.6+ KB



```python
wine_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
    </tr>
  </tbody>
</table>
</div>



# After all, we make visualization to understand data, it helps us to make predictions easily


```python
## we used hist() function to show different features of data
wine_df.fixed_acidity.hist();
```


![png](output_16_0.png)



```python
wine_df.total_sulfur_dioxide.hist();
```


![png](output_17_0.png)



```python
wine_df.pH.hist();
```


![png](output_18_0.png)



```python
wine_df.alcohol.hist()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7fddef8daf28>




![png](output_19_1.png)



```python
# after that, our next mission, to create scatter plot o compare different features with quality.
wine_df.plot(x="volatile_acidity", y="quality", kind="scatter");
```


![png](output_20_0.png)



```python
wine_df.plot(x="residual_sugar", y="quality", kind="scatter");
```


![png](output_21_0.png)



```python
wine_df.plot(x="pH", y="quality", kind="scatter");
```


![png](output_22_0.png)



```python
wine_df.plot(x="alcohol", y="quality", kind="scatter");
```


![png](output_23_0.png)


#  At last we make drawing conclusion


```python
# Find the mean quality of each wine type (red and white) with groupby
wine_df.groupby('color').mean().quality
```




    color
    red      5.636023
    white    5.877909
    Name: quality, dtype: float64




```python
# View the min, 25%, 50%, 75%, max pH values with Pandas describe
wine_df.describe().pH
```




    count    6497.000000
    mean        3.218501
    std         0.160787
    min         2.720000
    25%         3.110000
    50%         3.210000
    75%         3.320000
    max         4.010000
    Name: pH, dtype: float64




```python
# Bin edges that will be used to "cut" the data into groups
bin_edges = [2, 3, 3.5, 4, 5] # Fill in this list with five values you just found
# Labels for the four acidity level groups
# Name each acidity level category
bin_names = ['high', 'mod_high', 'medium', 'low']
# Creates acidity_levels column
wine_df['acidity_levels'] = pd.cut(wine_df['pH'], bin_edges, labels=bin_names)

# Checks for successful creation of this column
wine_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fixed_acidity</th>
      <th>volatile_acidity</th>
      <th>citric_acid</th>
      <th>residual_sugar</th>
      <th>chlorides</th>
      <th>free_sulfur_dioxide</th>
      <th>total_sulfur_dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
      <th>color</th>
      <th>acidity_levels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
      <td>medium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
      <td>mod_high</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
      <td>red</td>
      <td>mod_high</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
      <td>red</td>
      <td>mod_high</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
      <td>red</td>
      <td>medium</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Find the mean quality of each acidity level with groupby
wine_df.groupby('acidity_levels').mean().quality
```




    acidity_levels
    high        5.836996
    mod_high    5.820631
    medium      5.742671
    low         6.000000
    Name: quality, dtype: float64




```python
# get the median amount of alcohol content

wine_df.median().alcohol
```




    10.300000000000001




```python
# select samples with alcohol content less than the median
low_alcohol = wine_df.query('alcohol < 10.3')

# select samples with alcohol content greater than or equal to the median
high_alcohol =  wine_df.query('alcohol >= 10.3')

# ensure these queries included each sample exactly once
num_samples = wine_df.shape[0]
num_samples == low_alcohol['quality'].count() + high_alcohol['quality'].count() # should be True
```




    True




```python
# get mean quality rating for the low alcohol and high alcohol groups
low_alcohol.quality.mean(), high_alcohol.quality.mean()
```




    (5.475920679886686, 6.1460843373493974)




```python
# get the median amount of residual sugar
wine_df.residual_sugar.median()
```




    3.0




```python
# select samples with residual sugar less than the median
low_sugar = wine_df.query('residual_sugar < 3.0')

# select samples with residual sugar greater than or equal to the median
high_sugar = wine_df.query('residual_sugar >= 3.0')

# ensure these queries included each sample exactly once
num_samples == low_sugar['quality'].count() + high_sugar['quality'].count() # should be True
```




    True




```python

# get mean quality rating for the low sugar and high sugar groups
low_sugar.quality.mean(), high_sugar.quality.mean()
```




    (5.8088007437248219, 5.8278287461773699)




```python
# Use query to select each group and get its mean quality
median = wine_df['alcohol'].median()
low = wine_df.query('alcohol < {}'.format(median))
high = wine_df.query('alcohol >= {}'.format(median))

mean_quality_low = low['quality'].mean()
mean_quality_high = high['quality'].mean()
# Create a bar chart with proper labels
locations = [1, 2]
heights = [mean_quality_low, mean_quality_high]
labels = ['Low', 'High']
plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Alcohol Content')
plt.xlabel('Alcohol Content')
plt.ylabel('Average Quality Rating');

```


![png](output_35_0.png)



```python
# Use groupby to get the mean quality for each acidity level
acdf=wine_df.groupby('acidity_levels').mean().quality
acdf
```




    acidity_levels
    high        5.836996
    mod_high    5.820631
    medium      5.742671
    low         6.000000
    Name: quality, dtype: float64




```python
locations = [4, 1, 2, 3] 
heights = acdf
labels = ['Low', 'Medium', 'Moderately High', 'High']

plt.bar(locations, heights, tick_label=labels)
plt.title('Average Quality Ratings by Acidity Level'),
plt.xlabel('Acidity Level')
plt.ylabel('Average Quality Rating');
```


![png](output_37_0.png)


# Create arrays for red bar heights white bar heights


```python
import seaborn as sns
sns.set_style('darkgrid')

```


```python
# get counts for each rating and color
color_counts = wine_df.groupby(['color', 'quality']).count()['pH']
color_counts
```




    color  quality
    red    3            10
           4            53
           5           681
           6           638
           7           199
           8            18
    white  3            20
           4           163
           5          1457
           6          2198
           7           880
           8           175
           9             5
    Name: pH, dtype: int64




```python
# get total counts for each color
color_totals = wine_df.groupby('color').count()['pH']
color_totals
```




    color
    red      1599
    white    4898
    Name: pH, dtype: int64




```python
# get proportions by dividing red rating counts by total # of red samples
red_proportions = color_counts['red'] / color_totals['red']
red_proportions
```




    quality
    3    0.006254
    4    0.033146
    5    0.425891
    6    0.398999
    7    0.124453
    8    0.011257
    Name: pH, dtype: float64




```python
# get proportions by dividing white rating counts by total # of white samples
white_proportions = color_counts['white'] / color_totals['white']
white_proportions
```




    quality
    3    0.004083
    4    0.033279
    5    0.297468
    6    0.448755
    7    0.179665
    8    0.035729
    9    0.001021
    Name: pH, dtype: float64




```python
#Plot proportions on a bar chart
#Set the x coordinate location for each rating group and and width of each bar.
ind = np.arange(len(red_proportions))  # the x locations for the groups
width = 0.35       # the width of the bars
```


```python
# plot bars
red_bars = plt.bar(ind, red_proportions, width, color='r', alpha=.7, label='Red Wine')
white_bars = plt.bar(ind+ width, white_proportions, width, color='w', alpha=.7, label='White Wine')

# title and labels
plt.ylabel('Proportion')
plt.xlabel('Quality')
plt.title('Proportion by Wine Color and Quality')
locations = ind + width / 2  # xtick locations
labels = ['3', '4', '5', '6', '7', '8', '9']  # xtick labels
plt.xticks(locations, labels)

# legend
plt.legend()
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-73-b5179befdc5a> in <module>()
          1 # plot bars
          2 red_bars = plt.bar(ind, red_proportions, width, color='r', alpha=.7, label='Red Wine')
    ----> 3 white_bars = plt.bar(ind, width, white_proportions, width, color='w', alpha=.7, label='White Wine')
          4 
          5 # title and labels


    /opt/conda/lib/python3.6/site-packages/matplotlib/pyplot.py in bar(*args, **kwargs)
       2625                       mplDeprecation)
       2626     try:
    -> 2627         ret = ax.bar(*args, **kwargs)
       2628     finally:
       2629         ax._hold = washold


    /opt/conda/lib/python3.6/site-packages/matplotlib/__init__.py in inner(ax, *args, **kwargs)
       1708                     warnings.warn(msg % (label_namer, func.__name__),
       1709                                   RuntimeWarning, stacklevel=2)
    -> 1710             return func(ax, *args, **kwargs)
       1711         pre_doc = inner.__doc__
       1712         if pre_doc is None:


    /opt/conda/lib/python3.6/site-packages/matplotlib/axes/_axes.py in bar(self, *args, **kwargs)
       2079         x, height, width, y, linewidth = np.broadcast_arrays(
       2080             # Make args iterable too.
    -> 2081             np.atleast_1d(x), height, width, y, linewidth)
       2082 
       2083         if orientation == 'vertical':


    /opt/conda/lib/python3.6/site-packages/numpy/lib/stride_tricks.py in broadcast_arrays(*args, **kwargs)
        248     args = [np.array(_m, copy=False, subok=subok) for _m in args]
        249 
    --> 250     shape = _broadcast_shape(*args)
        251 
        252     if all(array.shape == shape for array in args):


    /opt/conda/lib/python3.6/site-packages/numpy/lib/stride_tricks.py in _broadcast_shape(*args)
        183     # use the old-iterator because np.nditer does not handle size 0 arrays
        184     # consistently
    --> 185     b = np.broadcast(*args[:32])
        186     # unfortunately, it cannot handle 32 or more arguments directly
        187     for pos in range(32, len(args), 31):


    ValueError: shape mismatch: objects cannot be broadcast to a single shape



![png](output_45_1.png)



```python
#Oh, that didn't work because we're missing a red wine value for a the 9 rating. Even though this number is a 0, 
#we need it for our plot. Run the last two cells after running the cell below.
red_proportions['9'] = 0
red_proportions
```




    quality
    3    0.006254
    4    0.033146
    5    0.425891
    6    0.398999
    7    0.124453
    8    0.011257
    9    0.000000
    Name: pH, dtype: float64




```python

```
