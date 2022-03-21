# COMP2200/COMP6200 Practicals S2-2021

## Week 1 (28/07/21)

This was our introductory practical to Python.

The topics covered were:
* String manipulation
* Loops
* Functions
* Data and plotting


## Week 2 (04/08/21)

The topics covered were:
* Markdown
* Lists and dictionaries
* pandas dataframes

I also did Further Practice on dataframes using Chapter 2.6 as a reference.


## Week 3 (11/08/21)

We analysed the [auto mpg dataset](https://www.kaggle.com/uciml/autompg-dataset), comparing the mpg to weight of all-cylinder cars vs 8-cylinder cars.

We also analysed the [adult census income dataset](https://www.kaggle.com/uciml/adult-census-income), analysing how age varied between high and low income earners by gender (male and female). I also explored education vs income, and race vs income by gender.

The topics covered were:
* pandas dataframes
* matplotlib


## Week 4 (18/08/21)

We analysed the Pokemon dataset using `seaborn` following this [tutorial](https://elitedatascience.com/python-seaborn-tutorial). These were the data visualisation functions we looked at:
* sns.lmplot
* sns.violinplot
* sns.boxplot
* sns.stripplot
* sns.swarmplot ("A swarm plot is very similar to a strip plot, yet the locations of points are adjusted automatically to avoid overlap even if the jitter value is not applied", [O'Reilly](https://www.oreilly.com/library/view/matplotlib-2x-by/9781788295260/f71c5b21-e7ea-447f-b898-0e38ba786ad7.xhtml))
* sns.heatmap
* sns.displot
* sns.countplot
* sns.catplot (previously sns.factorplot)


## Week 5 (25/08/21)

We analysed the [Iris](https://archive.ics.uci.edu/ml/datasets/iris) and [auto mpg dataset](https://www.kaggle.com/uciml/autompg-dataset) datasets by applying linear regression. To do this we used these packages from the `sklearn` module:
* sklearn.model_selection
* sklearn.linear_model
* sklearn.metrics

I also installed the [RISE](https://rise.readthedocs.io/en/stable/) Jupyter notebook extension which instantly turns a notebook into a live reveal.js-based presentation.


## Week 6 (01/09/21)

We analysed the [Wisconsin breast cancer dataset](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data) and [sea ice dataset](https://github.com/MQCOMP2200-S2-2021/practical-week-1-LanceWhitehorn/blob/main/files/SeaIce.txt) by applying logistic regression and linear regression respectively. We imported the data, removed outliers, normalised the data, fit the data to a regression model, and look at the metrics: MSE and R2 score for the linear regression, and sensitivity, specificity, and accuracy score for the logistic regression.


## Week 7 (08/09/21)
We worked on [Portfolio 2](https://github.com/MQCOMP2200-S2-2021/data-science-portfolio-LanceWhitehorn/blob/main/Portfolio%202.ipynb).


## Week 8 (29/09/21)
We re-visited the Pokemon dataset and applied clustering techniques including:
* k-means clustering
* Hierarchical clustering

We also explored the [book summaries dataset](http://www.cs.cmu.edu/~dbamman/booksummaries.html) and applied text analysis. We first used `TfidfVectorizer` from the `sklearn` library to compute the numerical features then applied hierarchical clustering.



## Week 9 (06/10/21)
We worked on [Portfolio 3](https://github.com/MQCOMP2200-S2-2021/data-science-portfolio-LanceWhitehorn/blob/main/Portfolio%203.ipynb).


## Week 10 (13/10/21)
We practiced KNN Classifier and Naive Bayes Classifier on the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris). We also looked at methods to select the optimal K either manually or using `GridSearchCV` from `sklearn.model_selection`.

Additionally, we used a `MultinomialNB` classification model on the weather dataset. This is suitable for classification with discrete features. To let the model handle to categorical data, we often need to transform the categorical values to numberic ones, through encoding.


## Week 11 (20/10/21)
We analysed the [avila dataset](https://archive.ics.uci.edu/ml/datasets/Avila) by applying neural networks. We varied a number of parameters including:
* The number of hidden layers
* The solver for the learning process ['lbfgs', 'sgd', 'adam']
* The activation functions ['identity', 'logistic', 'tanh', 'relu']
* The alpha value
* The number of iterations


## Week 12 (27/10/21)
We analysed the [breast cancer dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer) using:
* DecisionTreeClassifier
* DecisionTreeRegressor

We also analysed the [Wisconsin breast cancer dataset](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) using both a Decision Tree Classifier and a Logistic Regression, and compared their accuracies.





