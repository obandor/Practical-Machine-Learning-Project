###########################################################
## Assumption: Data already downloaded to current directory
## data files: pml-training.csv and pml-testing.csv
###########################################################

###########################################################
## load packages
###########################################################
library("caret")
library("randomForest")
library("doParallel")

###########################################################
## load training and test data
###########################################################
training <- read.csv("pml-training.csv", na.strings= c("NA",""," ","#DIV/0!"))
test <- read.csv("pml-testing.csv", na.strings= c("NA",""," ","#DIV/0!"))
dim(training)
dim(test)

###########################################################
## Pre Processing training data
###########################################################
## Removing NAs and #DIV/0!
training_no_na<-training[,colSums(is.na(training))==0]
dim(training_no_na)

## Remove the first 7 columns since they are identifiers.
idColToRemove <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
training_no_na_idcol_rem <- training_no_na[, -which(names(training_no_na) %in% idColToRemove)]
dim(training_no_na_idcol_rem)

## split training in train_nc, and test_nc 
set.seed(12345)
train_nc_part <- createDataPartition(y = training_no_na_idcol_rem$classe, p = 0.6, list = FALSE)
train_nc <- training_no_na_idcol_rem[train_nc_part, ]
test_nc <- training_no_na_idcol_rem[-train_nc_part, ]

###########################################################
## Modeling
###########################################################
## Random Forest Model
## start parallel processing
cluster <- makeCluster(detectCores() - 1)
registerDoParallel(cluster)
rfModelTrain <- randomForest(classe ~ ., data = train_nc, importance = TRUE, ntrees = 10)
## turn off parallel processing
stopCluster(cluster)
rfModelTrain
registerDoSEQ()

###########################################################
## Cross Validation
###########################################################
## test the model against train_nc
train_nc_test <- predict(rfModelTrain, train_nc)
confusionMatrix(train_nc$classe, train_nc_test) # Accuracy : 1

## Cross Validation : test the model against the test_nc 
crossVal <- predict(rfModelTrain, test_nc)
confusionMatrix(test_nc$classe, crossVal) # Accuracy : 0.9939%


###########################################################
## Pre Processing test data
###########################################################
## Removing NAs and #DIV/0!
test_no_na<-test[,colSums(is.na(test))==0]
dim(test_no_na)

## Remove the following columns since they are identifiers.
idColToRemove <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "new_window", "num_window")
test_no_na_idcol_rem <- test_no_na[, -which(names(test_no_na) %in% idColToRemove)]
dim(test_no_na_idcol_rem)

###########################################################
## Predictions
###########################################################

prediction <- predict(rfModelTrain, test_no_na_idcol_rem)
prediction 
