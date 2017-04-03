# Jupyter notebooks

_Ref: Master python data analysis_

- init
```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
from pandas import Series, DataFrame
import numpy.random as rnd
import scipy.stats as st
```

- [movie rating](movieRating/movie_rating.ipynb)
    + pd.read_csv('dataset/ratings.dat.txt', sep='::', index_col=False, names=cols, encoding='utf8')
    + df.value_counts()
    + df.sort_index()
    + df.plot(kind='bar')
    + df.map(lambda x: (dramaIds==x).any())
- [gss](gss/gss.ipynb)
    + pd.read_stata('GSS2014merged_R6.dta', convert_categoricals=False)
    + gssData.set_index('id')
    + gssData.drop('id', 1, inplace=True)
    + gssData.to_csv('GSS2014merged.csv')
    + gssData['age'].hist(bins=25)
    + plt.locator_params(nbins=5)
    + inc_age = gssData[['realrinc', 'age']]
    + inc_age.head()
    + gssData[['realrinc', 'age']].dropna()
    + inc_age[inc_age['realrinc'] > 1e5].count()
    + age.plot(kind='kde', lw=2)
    + stats.probplot(age, dist='norm', plot=plt)
    + inc.plot(kind='box')
- [distributions basics](model_tutorial/model.ipynb)
    + plot_cdf
    + np.fromfile('s057.txt', sep='\n', dtype=np.float64)
    + st.norm.fit(wingLens)
    + st.norm(loc=mean, scale=std)
    + np.linspace
    + rvNorm.cdf(59), vNorm.ppf(0.25)
    + rvNorm.stats(moments='mvks')
    + st.weibull_min(beta, scale=eta)
    + st.binom(20, 0.5), rvBinom.pmf(12)
    + st.multivariate_normal(mean=[0, 0]).rvs(300)
    + df.plot(kind='scatter', x='Z1', y='Z2')
- [linear regression](LinearRegression/LinearRegression.ipynb)
    + rates['country'].isin([coords['SHORT_NAME'][i]])
    + coords.loc[i, ['LAT', 'LONG']].values.astype('float')
    + rates.loc[ind, ['lat', 'lng']] = list(val)
    + rates.loc[rates['lat'].isin(['']), ['lat']] = np.nan
    + plt.fill_between
    + np.arange(23.5, rates['dfe'].max()+1, 10, dtype='float')
    + rates.groupby(np.digitize(rates['dfe'], bins))
    + plt.errorbar
    + sel = ~rates['dfe'].isnull() * rates['dfe'] > 23.5
    + LinearRegression().fit(x, y)
    + plt.plot(xx, model.predict(xx), '--', lw=3)
