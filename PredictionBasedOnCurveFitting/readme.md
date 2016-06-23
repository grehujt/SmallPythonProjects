# Making Prediction Based On Curve Fitting

_NOTE: this example is taken from **Building Machine Learning System in Python**._

### Key points
- Basic feelings on machine learning system.
    1. data cleaning
    2. feature engineering
    3. data modeling
    4. models evaluation
- Basic usages of curve fitting functions in scipy.
- Basic usages of data visualization of matplotlib.

### Scenario
- We are running a website, as time goes by, an increasing number of users are attracted by our website.
- We are going to allocate resources (more servers) to make our clients happy.
- But we do not want to waste money on allocating too many resources.
- We have a [tsv](./web_traffic.tsv) (tab-seperated values) file, which contains the web stats for last month. And each line contains the hour consecutively and number of page views in that hour.
- We want to know in advance when our current limit will be hit (10k requests per hour).

### Step-by-step solution