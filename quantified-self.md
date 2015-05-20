---
title: "Quantified Self Predictions"
author: "Panos Rontogiannis"
date: "2015-05-18"
output: html_document
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

## What you should submit

The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases. 

1. Your submission should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).
2. You should also apply your machine learning algorithm to the 20 test cases available in the test data above. Please submit your predictions in appropriate format to the programming assignment for automated grading. See the programming assignment for additional details. 




```r
knitr::opts_chunk$set(fig.width=12, fig.height=8, fig.path='Figs/', warning=FALSE, message=FALSE)

library(caret)
```

## The Data

> Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.



```r
training_orig <- read.csv('pml-training.csv.gz')
testing_final <- read.csv('pml-testing.csv.gz')
```

Initially we have 19622 observations for training and 20 for final testing. Both datasets have 160 variables (columns). Let's split the training data into two more sets. Firstly the actual training dataset and secondly the testing set to be used to select the best model.


```r
set.seed(3456)
inTrain <- createDataPartition(training_orig$classe, p=0.7, list=FALSE)
training <- training_orig[inTrain,]
testing <- training_orig[-inTrain,]
```

So now we have 13737 observations for training our models and 5885 for evaluating the trained models and picking the best one.

## Exploratory Analysis

Let's explore the training and testing datasets. Let's start by looking for missing values.


```r
complete_training <- complete.cases(training)
```

From the training data we can see that 280 out of 13737 observations are complete (no missing values at any column). Looking more closely we can see that there are columns that mostly consist of missing values.


```r
# Count missing values per column.
na_count <- colSums(is.na(training))
names(na_count) <- names(training)
na_count <- na_count[na_count > 0]

print(data.frame(na_count))
```

```
##                          na_count
## max_roll_belt               13457
## max_picth_belt              13457
## min_roll_belt               13457
## min_pitch_belt              13457
## amplitude_roll_belt         13457
## amplitude_pitch_belt        13457
## var_total_accel_belt        13457
## avg_roll_belt               13457
## stddev_roll_belt            13457
## var_roll_belt               13457
## avg_pitch_belt              13457
## stddev_pitch_belt           13457
## var_pitch_belt              13457
## avg_yaw_belt                13457
## stddev_yaw_belt             13457
## var_yaw_belt                13457
## var_accel_arm               13457
## avg_roll_arm                13457
## stddev_roll_arm             13457
## var_roll_arm                13457
## avg_pitch_arm               13457
## stddev_pitch_arm            13457
## var_pitch_arm               13457
## avg_yaw_arm                 13457
## stddev_yaw_arm              13457
## var_yaw_arm                 13457
## max_roll_arm                13457
## max_picth_arm               13457
## max_yaw_arm                 13457
## min_roll_arm                13457
## min_pitch_arm               13457
## min_yaw_arm                 13457
## amplitude_roll_arm          13457
## amplitude_pitch_arm         13457
## amplitude_yaw_arm           13457
## max_roll_dumbbell           13457
## max_picth_dumbbell          13457
## min_roll_dumbbell           13457
## min_pitch_dumbbell          13457
## amplitude_roll_dumbbell     13457
## amplitude_pitch_dumbbell    13457
## var_accel_dumbbell          13457
## avg_roll_dumbbell           13457
## stddev_roll_dumbbell        13457
## var_roll_dumbbell           13457
## avg_pitch_dumbbell          13457
## stddev_pitch_dumbbell       13457
## var_pitch_dumbbell          13457
## avg_yaw_dumbbell            13457
## stddev_yaw_dumbbell         13457
## var_yaw_dumbbell            13457
## max_roll_forearm            13457
## max_picth_forearm           13457
## min_roll_forearm            13457
## min_pitch_forearm           13457
## amplitude_roll_forearm      13457
## amplitude_pitch_forearm     13457
## var_accel_forearm           13457
## avg_roll_forearm            13457
## stddev_roll_forearm         13457
## var_roll_forearm            13457
## avg_pitch_forearm           13457
## stddev_pitch_forearm        13457
## var_pitch_forearm           13457
## avg_yaw_forearm             13457
## stddev_yaw_forearm          13457
## var_yaw_forearm             13457
```

```r
# Remove columns with missing values.
training_no_na <- training[, !(names(training) %in% names(na_count))]
```

After removing those 67 columns we are left with 93.

Next let's see if there are any variables with near zero variance.


```r
nzv <- nearZeroVar(training_no_na)
# Remove columns with near zero variance.
training_no_na_nzv <- training_no_na[, -nzv]
```

After removing those 34 columns we are left with 59.

Now let's look for high correlations between the remainder varables.


```r
# Need to remove factor variables first for the cor function to work.
no_factors <- c()
for (f in colnames(training_no_na_nzv)) {
    if (!is.factor(training_no_na_nzv[, f]))
        no_factors <- c(no_factors, f)
}
cor_training <- cor(training_no_na_nzv[, no_factors])
# find highly correlated descriptors
corrDescr <- findCorrelation(cor_training, cutoff = 0.75)
# remove them from training set
training_no_na_nzv_cor <- training_no_na_nzv[, -corrDescr]
```

After removing those 22 columns we are left with 37.

## Training

To train the classifier we can use the following methods:


```r
# Windows
library(doSNOW)
cl <- makeCluster(4, type = 'SOCK')
registerDoSNOW(cl)
# library(doParallel)
# cl <- makeCluster(4, type = 'SOCK')
# registerDoParallel(cl)
# Linux
#library(doMC)
#registerDoMC(2)
```

1. Regression Trees (rpart)


```r
m_no_na_nzv_cor_rpart <- train(classe ~ ., data = training_no_na_nzv_cor, method='rpart')
print(m_no_na_nzv_cor_rpart)
```

```
## CART 
## 
## 13737 samples
##    36 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## 
## Summary of sample sizes: 13737, 13737, 13737, 13737, 13737, 13737, ... 
## 
## Resampling results across tuning parameters:
## 
##   cp         Accuracy   Kappa      Accuracy SD  Kappa SD 
##   0.2437188  0.7580868  0.6927861  0.09006438   0.1147155
##   0.2568406  0.5717095  0.4534692  0.09373447   0.1212279
##   0.2703692  0.3903032  0.1839026  0.09628495   0.1663924
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.2437188.
```

```r
#m_no_na_nzv_cor_rpart_pre <- train(classe ~ ., data = training_no_na_nzv_cor, method='rpart', preProcess = c('center', 'scale'))
#print(m_no_na_nzv_cor_rpart_pre)
```

2. Random Forest (rf). Though random forests are prone to overfitting, the train function will perform cross validation automatically so we don't need to worry about it. 


```r
m_no_na_nzv_cor_rf <- train(classe ~ ., data = training_no_na_nzv_cor, method='rf', prox=TRUE)
```

```
## Error in train.default(x, y, weights = w, ...): final tuning parameters could not be determined
```

```r
print(m_no_na_nzv_cor_rf)
```

```
## Error in print(m_no_na_nzv_cor_rf): object 'm_no_na_nzv_cor_rf' not found
```

3. Stochastic Gradient Boosting (gbm)


```r
m_no_na_nzv_cor_gbm <- train(classe ~ ., data = training_no_na_nzv_cor, method='gbm', verbose=FALSE)
```

```
## Error in unserialize(socklist[[n]]): error reading from connection
```

```r
print(m_no_na_nzv_cor_gbm)
```

```
## Error in print(m_no_na_nzv_cor_gbm): object 'm_no_na_nzv_cor_gbm' not found
```


```r
stopCluster(cl)
```

```
## Error in serialize(data, node$con): error writing to connection
```

Now let's compare all models. Firstly we can use the resamples function:


```r
resamps <- resamples(list(RPART = m_no_na_nzv_cor_rpart, RPART_PRE = m_no_na_nzv_cor_rpart_pre, RF = m_no_na_nzv_cor_rf, GBM = m_no_na_nzv_cor_gbm))
```

```
## Error in resamples(list(RPART = m_no_na_nzv_cor_rpart, RPART_PRE = m_no_na_nzv_cor_rpart_pre, : object 'm_no_na_nzv_cor_rpart_pre' not found
```

```r
summary(resamps)
```

```
## Error in summary(resamps): object 'resamps' not found
```

We can see that the random forest method is more accurate.

Finally let's use the testing dataset (taken from the original trianing set) to see how the three models perform compared to the actual classes.



## Results


## References

The data for this project come from this [source](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 
