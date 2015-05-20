# Quantified-Self Predictions
Panos Rontogiannis  
`r Sys.Date()`  

## Introduction

> Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

The data for this project comes from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. The participants were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

The goal of the project is to predict the manner in which the participants did the exercise. This is the **classe** variable in the training set. We will try to use some of the remaining variables to predict *classe* using the `caret` R package. 



## The Data

The data is loaded from two different zipped files 'pml-training.csv.gz' and 'pml-testing.csv.gz' containing the training and testing observations respectively.


```r
training_orig <- read.csv('pml-training.csv.gz')
testing_final <- read.csv('pml-testing.csv.gz')
```

Initially we have 19622 observations for training and 20 for final testing by the project evaluators. Note that this dataset does not contain the `classe` variable. For this reason we will only use this dataset at the end to create the files for the project submission. 

The training dataset has 160 variables (columns) including the response variable (`classe`). Let's split it into two more sets. Firstly the actual training dataset and secondly the testing set to be used to select the best model. This way we will be able to measure the out of sample error.


```r
set.seed(3456)
inTrain <- createDataPartition(training_orig$classe, p=0.7, list=FALSE)
training <- training_orig[inTrain,]
testing <- training_orig[-inTrain,]
```

So now we have 13737 observations for training our models and 5885 for evaluating the trained models and picking the best one.

## Exploratory Analysis

Let's explore the training dataset. First let's have a look at the response variable. It is a factor variable with 5 levels (A, B, C, D, E). As can be seen from the frequency table of the types of variable, most of the are numerical:

```r
table(sapply(colnames(training), FUN = function(n) class(training[, n])))
```

```
## 
##  factor integer numeric 
##      37      35      88
```

Let's continue by looking for missing values.


```r
complete_training <- complete.cases(training)
```

From the training data we can see that 280 out of 13737 observations are complete (no missing values at any column). Looking more closely we can see that there are columns that mostly consist of missing values.


```r
# Count missing values per column.
missing_values <- data.frame(col_name = names(training), na_count = colSums(is.na(training)), row.names = NULL)
missing_values <- subset(missing_values, na_count > 0)

qplot(missing_values$na_count, xlab = '# of NA', ylab = '# of variables', main = 'Exploring missing values', binwidth = 1)
```

![](Figs/explor_missing_values_pt2-1.png) 

```r
# Remove columns with missing values.
training_no_na <- training[, !(names(training) %in% missing_values$col_name)]
```

After removing those 67 columns we are left with 93.

Next let's see if there are any variables with near zero variance.


```r
nzv <- nearZeroVar(training_no_na)
# Remove columns with near zero variance.
training_no_na_nzv <- training_no_na[, -nzv]
```

After removing those 34 columns we are left with 59.

Now let's look for high correlations between the remainder variables.


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

After removing those 22 columns we are left with 37. These are: *accel_arm_y, accel_forearm_x, accel_forearm_y, accel_forearm_z, classe, gyros_arm_x, gyros_belt_x, gyros_dumbbell_x, gyros_forearm_y, magnet_arm_x, magnet_arm_y, magnet_arm_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, magnet_dumbbell_y, magnet_dumbbell_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z, num_window, pitch_arm, pitch_dumbbell, pitch_forearm, raw_timestamp_part_1, raw_timestamp_part_2, roll_arm, roll_belt, roll_dumbbell, roll_forearm, total_accel_belt, total_accel_dumbbell, user_name, X, yaw_arm, yaw_belt, yaw_forearm*.

We can see that there are two time-stamp variables as well as a factor variable called *user_name*. Let's remove them since they are irrelevant.


```r
training_no_na_nzv_cor <- training_no_na_nzv_cor[, !(names(training_no_na_nzv_cor) %in% c('raw_timestamp_part_1', 'raw_timestamp_part_2', 'user_name'))]
```

The final dataset before we start training has 34 columns (including the response variables). These are *accel_arm_y, accel_forearm_x, accel_forearm_y, accel_forearm_z, classe, gyros_arm_x, gyros_belt_x, gyros_dumbbell_x, gyros_forearm_y, magnet_arm_x, magnet_arm_y, magnet_arm_z, magnet_belt_x, magnet_belt_y, magnet_belt_z, magnet_dumbbell_y, magnet_dumbbell_z, magnet_forearm_x, magnet_forearm_y, magnet_forearm_z, num_window, pitch_arm, pitch_dumbbell, pitch_forearm, roll_arm, roll_belt, roll_dumbbell, roll_forearm, total_accel_belt, total_accel_dumbbell, X, yaw_arm, yaw_belt, yaw_forearm*.

After all this pre-processing, the frequency table of variable types has become:

```r
table(sapply(colnames(training_no_na_nzv_cor), FUN = function(n) class(training_no_na_nzv_cor[, n])))
```

```
## 
##  factor integer numeric 
##       1      16      17
```

## Training



Because `classe` is a *factor* variable we will need to train a classifier and since it has more than two levels, we cannot use *Generalized Linear Models* (glm). Instead we have to use more advanced classifiers. More specifically *regression trees (CART)*, *random forest* and *stochastic gradient boosting*. All classifiers will be trained using *5-fold Cross Validation* to reduce model overfitting and to estimate the out of sample error during training. We expect this estimate to be close to the final out of sample error that we will calculate later on using the test dataset. Our metric will be *accuracy*.


```r
trControl <- trainControl(method="cv", number=5)
```

Let's train the classifiers:

### 1. Regression Trees (rpart)


```r
m_no_na_nzv_cor_rpart <- train(classe ~ ., data = training_no_na_nzv_cor, method='rpart', trControl = trControl)
print(m_no_na_nzv_cor_rpart)
```

```
## CART 
## 
## 13737 samples
##    33 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10991, 10989, 10989, 10990, 10989 
## 
## Resampling results across tuning parameters:
## 
##   cp         Accuracy   Kappa      Accuracy SD  Kappa SD 
##   0.2437188  0.7660960  0.7028174  0.09536328   0.1218492
##   0.2568406  0.5512688  0.4263408  0.10052663   0.1303105
##   0.2703692  0.3616623  0.1324611  0.10601323   0.1813798
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was cp = 0.2437188.
```

### 2. Random Forest (rf). 


```r
rfParam <- expand.grid(mtry = 2)
m_no_na_nzv_cor_rf <- train(classe ~ ., data = training_no_na_nzv_cor, method='parRF', prox=TRUE, tuneGrid = rfParam, trControl = trControl)
print(m_no_na_nzv_cor_rf)
```

```
## Parallel Random Forest 
## 
## 13737 samples
##    33 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10990, 10991, 10991, 10988, 10988 
## 
## Resampling results
## 
##   Accuracy   Kappa      Accuracy SD   Kappa SD    
##   0.9994543  0.9993099  0.0007716698  0.0009759545
## 
## Tuning parameter 'mtry' was held constant at a value of 2
## 
```

### 3. Stochastic Gradient Boosting (gbm)


```r
m_no_na_nzv_cor_gbm <- train(classe ~ ., data = training_no_na_nzv_cor, method='gbm', verbose=FALSE, trControl = trControl)
print(m_no_na_nzv_cor_gbm)
```

```
## Stochastic Gradient Boosting 
## 
## 13737 samples
##    33 predictors
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (5 fold) 
## 
## Summary of sample sizes: 10990, 10990, 10989, 10991, 10988 
## 
## Resampling results across tuning parameters:
## 
##   interaction.depth  n.trees  Accuracy   Kappa      Accuracy SD 
##   1                   50      0.9998545  0.9998159  0.0001992805
##   1                  100      0.9998545  0.9998159  0.0001992805
##   1                  150      0.9998545  0.9998159  0.0001992805
##   2                   50      0.9998545  0.9998159  0.0001992805
##   2                  100      0.9998545  0.9998159  0.0001992805
##   2                  150      0.9997817  0.9997239  0.0003253794
##   3                   50      0.9998545  0.9998159  0.0001992805
##   3                  100      0.9999272  0.9999079  0.0001627415
##   3                  150      0.9999272  0.9999079  0.0001627415
##   Kappa SD    
##   0.0002520541
##   0.0002520541
##   0.0002520541
##   0.0002520541
##   0.0002520541
##   0.0004115541
##   0.0002520541
##   0.0002058324
##   0.0002058324
## 
## Tuning parameter 'shrinkage' was held constant at a value of 0.1
## 
## Tuning parameter 'n.minobsinnode' was held constant at a value of 10
## Accuracy was used to select the optimal model using  the largest value.
## The final values used for the model were n.trees = 100,
##  interaction.depth = 3, shrinkage = 0.1 and n.minobsinnode = 10.
```

Now let's compare all models using the resamples function:


```r
resamps <- resamples(list(RPART = m_no_na_nzv_cor_rpart, RF = m_no_na_nzv_cor_rf, GBM = m_no_na_nzv_cor_gbm))
summary(resamps)
```

```
## 
## Call:
## summary.resamples(object = resamps)
## 
## Models: RPART, RF, GBM 
## Number of resamples: 5 
## 
## Accuracy 
##         Min. 1st Qu. Median   Mean 3rd Qu.   Max. NA's
## RPART 0.6616  0.6617 0.8355 0.7661  0.8358 0.8359    0
## RF    0.9989  0.9992 0.9995 0.9995  0.9997 1.0000    3
## GBM   0.9996  1.0000 1.0000 0.9999  1.0000 1.0000    0
## 
## Kappa 
##         Min. 1st Qu. Median   Mean 3rd Qu.  Max. NA's
## RPART 0.5693  0.5694 0.7915 0.7028  0.7919 0.792    0
## RF    0.9986  0.9990 0.9993 0.9993  0.9997 1.000    3
## GBM   0.9995  1.0000 1.0000 0.9999  1.0000 1.000    0
```

We can see that the random forest and boosting methods are significantly more accurate than CART.

## Results

Let's use the testing dataset (taken from the original training set) to see how the three models perform compared to the actual values of the response variable *classe*:

### 1. Regression Trees (rpart)

```r
pred_rpart <- predict(m_no_na_nzv_cor_rpart, testing)
cm_rpart <- confusionMatrix(pred_rpart, testing$classe)
cm_rpart
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1139    0    0    0
##          C    0    0    0    0    0
##          D    0    0    0    0    0
##          E    0    0 1026  964 1082
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6617          
##                  95% CI : (0.6494, 0.6738)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5694          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   1.0000   0.0000   0.0000   1.0000
## Specificity            1.0000   0.9998   1.0000   1.0000   0.5857
## Pos Pred Value         1.0000   0.9991      NaN      NaN   0.3522
## Neg Pred Value         0.9998   1.0000   0.8257   0.8362   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1935   0.0000   0.0000   0.1839
## Detection Prevalence   0.2843   0.1937   0.0000   0.0000   0.5220
## Balanced Accuracy      0.9997   0.9999   0.5000   0.5000   0.7928
```

We can see that the out of sample error measured by accuracy is 0.6616822. During training, this was estimated to be 0.766096. From the table it is obvious that classes 'C' and 'D' are all mistakenly predicted to be class 'E'.

### 2. Random Forest (rf)

```r
pred_rf <- predict(m_no_na_nzv_cor_rf, testing)
cm_rf <- confusionMatrix(pred_rf, testing$classe)
cm_rf
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    0    0    0
##          C    0    0 1026    1    0
##          D    0    0    0  963    1
##          E    0    0    0    0 1081
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9997     
##                  95% CI : (0.9988, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9996     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   0.9990   0.9991
## Specificity            1.0000   1.0000   0.9998   0.9998   1.0000
## Pos Pred Value         1.0000   1.0000   0.9990   0.9990   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   0.9998   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2845   0.1935   0.1743   0.1636   0.1837
## Detection Prevalence   0.2845   0.1935   0.1745   0.1638   0.1837
## Balanced Accuracy      1.0000   1.0000   0.9999   0.9994   0.9995
```

In this case accuracy is 0.9996602. During training, this was estimated to be 0.9994543. From the table we see that almost all observations were correctly predicted. The only erros are in predictions for classes 'C' and 'C'. Overall this classifier performed really well.

### 3. Stochastic Gradient Boosting (gbm)

```r
pred_gbm <- predict(m_no_na_nzv_cor_gbm, testing)
cm_gbm <- confusionMatrix(pred_gbm, testing$classe)
cm_gbm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1673    0    0    0    0
##          B    1 1139    0    0    0
##          C    0    0 1026    0    0
##          D    0    0    0  964    0
##          E    0    0    0    0 1082
## 
## Overall Statistics
##                                      
##                Accuracy : 0.9998     
##                  95% CI : (0.9991, 1)
##     No Information Rate : 0.2845     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 0.9998     
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9994   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   0.9998   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   0.9991   1.0000   1.0000   1.0000
## Neg Pred Value         0.9998   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2843   0.1935   0.1743   0.1638   0.1839
## Detection Prevalence   0.2843   0.1937   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9997   0.9999   1.0000   1.0000   1.0000
```

The accuracy of this model against the testing data is 0.9998301. During training, this was estimated to be 0.9998545. From the table we see that almost all observations were correctly predicted. The only missclassification was for class 'B'. Simillarly with the *random forest* classifier, boosting trained a classifier that performed exceptionally.

For the submission part of the project I chose to use the *random forest* classifier due to the slightly-slightly higher accuracy. Also for some reason both the *CART* and *boosting* classifiers classified all 20 observations as classe 'A'. The *random forest* classifier produced the correct results.

If it wasn't for this, we would have tried to combine (stack) these two classifiers to see if we can create an even more accurate one.

## References

The data for this project come from this [source](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013. 

## Appendix

To train in parallel the `doMC` package was used with 2 workers. The total time (in seconds) needed to train each method is:  
1. Regression Trees: 11.537  
2. Random Forest: 446.042  
3. Stochastic Gradient Boosting: 203.938  

Session Info:  

```
## R version 3.1.2 (2014-10-31)
## Platform: x86_64-pc-linux-gnu (64-bit)
## 
## locale:
##  [1] LC_CTYPE=en_US.UTF-8       LC_NUMERIC=C              
##  [3] LC_TIME=en_US.UTF-8        LC_COLLATE=en_US.UTF-8    
##  [5] LC_MONETARY=en_US.UTF-8    LC_MESSAGES=en_US.UTF-8   
##  [7] LC_PAPER=en_US.UTF-8       LC_NAME=C                 
##  [9] LC_ADDRESS=C               LC_TELEPHONE=C            
## [11] LC_MEASUREMENT=en_US.UTF-8 LC_IDENTIFICATION=C       
## 
## attached base packages:
## [1] splines   parallel  stats     graphics  grDevices utils     datasets 
## [8] methods   base     
## 
## other attached packages:
##  [1] plyr_1.8.2          gbm_2.1.1           survival_2.37-7    
##  [4] randomForest_4.6-10 rpart_4.1-9         doMC_1.3.3         
##  [7] iterators_1.0.7     foreach_1.4.2       caret_6.0-47       
## [10] ggplot2_1.0.1       lattice_0.20-29    
## 
## loaded via a namespace (and not attached):
##  [1] BradleyTerry2_1.0-6 brglm_0.5-9         car_2.0-25         
##  [4] class_7.3-12        codetools_0.2-10    colorspace_1.2-6   
##  [7] digest_0.6.8        e1071_1.6-4         evaluate_0.7       
## [10] formatR_1.2         grid_3.1.2          gtable_0.1.2       
## [13] gtools_3.4.2        htmltools_0.2.6     knitr_1.10         
## [16] labeling_0.3        lme4_1.1-7          MASS_7.3-37        
## [19] Matrix_1.1-5        mgcv_1.8-4          minqa_1.2.4        
## [22] munsell_0.4.2       nlme_3.1-119        nloptr_1.0.4       
## [25] nnet_7.3-9          pbkrtest_0.4-2      proto_0.3-10       
## [28] quantreg_5.11       Rcpp_0.11.5         reshape2_1.4.1     
## [31] rmarkdown_0.3.10    scales_0.2.4        SparseM_1.6        
## [34] stringr_0.6.2       tools_3.1.2         yaml_2.1.13
```

For the project submission, the following code was used to create the separate files for each of the 20 test observations:

```r
pml_write_files = function(x){
    n = length(x)
    for(i in 1:n){
        filename = paste0("problem_id_",i,".txt")
        write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
    }
}

final_pred <- predict(m_no_na_nzv_cor_rf, testing_final)
# pml_write_files(final_pred)
```
