# HIP ARTHROSCOPY PREDICTIVE MODELS
# GBM, NB, NN, XGBOOST, ELASTIC NET, RANDOM FOREST
# CLARE GRAZAL, M.S.
# FEBRUARY 2020; FINAL EDIT OCTOBER 2020

##################
# PREPROCESSING
##################
# Load necessary packages
set.seed(1234)
library(gbm)
library(missForest)
library(readxl)
library(dplyr)
library(caret)
library(Boruta)
library(ROSE)
library(class)
library(neuralnet)
library(xgboost)
library(Matrix)
library(glmnet)
library(naniar)

#Load data pulled from MDR
data<-read.csv("hipdatasheet_HRvar.csv")

#Remove variables:
  #-ID variables
  #-CPT/DX variables categorizing Hip Arthroscopy 
  #-Fiscal year which is irrelevant in future
data <- select(data, -c(1:11, 18))

#Change data types
convertFactor <- c(2:9, 15:80)
convertNum <- c(1, 10:14)
data[,convertFactor] <- data.frame(apply(data[convertFactor], 2, as.factor))
data[,convertNum] <- data.frame(apply(data[convertNum], 2, as.numeric))

#Remove variables with 0 variation
data <- select(data, -c(26,66))

#Remove colinear variable (patient age or patient age group)
data <- select(data, -c(2))

#Change outcome variable and dosing variables to binary due to inconsistencies in ME dose ranges
data$longpost_dailydose[data$longpost_dailydose == 0.0] <- "0"
data$longpost_dailydose[data$longpost_dailydose > 0.0] <- "1"
data$longpost_dailydose <- as.factor(data$longpost_dailydose)

data$presurgical_dailydose[data$presurgical_dailydose == 0.0] <- "0"
data$presurgical_dailydose[data$presurgical_dailydose > 0.0] <- "1"
data$presurgical_dailydose <- as.factor(data$presurgical_dailydose)

data$perisurgical_dailydose[data$perisurgical_dailydose == 0.0] <- "0"
data$perisurgical_dailydose[data$perisurgical_dailydose > 0.0] <- "1"
data$perisurgical_dailydose <- as.factor(data$perisurgical_dailydose)

data$postsurgical_dailydose[data$postsurgical_dailydose == 0.0] <- "0"
data$postsurgical_dailydose[data$postsurgical_dailydose > 0.0] <- "1"
data$postsurgical_dailydose <- as.factor(data$postsurgical_dailydose)

#################################################
#Split into balanced Training and Testing groups
#################################################
#Randomize data
random_index <- sample(1:nrow(data), nrow(data))
random_train <- data[random_index,]

set.seed(345)
trainIndex <- createDataPartition(data$longpost_dailydose, p = .8, 
                                  list = FALSE, 
                                  times = 1)
Train <- data[ trainIndex,]
Test  <- data[-trainIndex,]

#Check that outcome proportions are equal
prop.table(table(Train$longpost_dailydose))
prop.table(table(Test$longpost_dailydose))

############################
#Imputation with missForest
############################
#Standardize blank data
Train %>% replace_with_na_all(condition = ~.x %in% common_na_strings)
Test %>% replace_with_na_all(condition = ~.x %in% common_na_strings)

#Run missForest
Train <- as.data.frame(Train)
Train.imp <- missForest(Train, maxiter = 10, ntree = 100, variablewise = FALSE,
                        decreasing = FALSE, verbose = TRUE,mtry = floor(sqrt(ncol(data))), 
                        replace = TRUE,classwt = NULL, cutoff = NULL, strata = NULL,
                        sampsize = NULL, nodesize = NULL, maxnodes = NULL,xtrue = NA, 
                        parallelize = c('no'))
Train <- Train.imp$ximp

#run imputation algorithm on Test data
Test <- as.data.frame(Test)
Test.imp <- missForest(Test, maxiter = 10, ntree = 100, variablewise = FALSE, 
                       decreasing = FALSE, verbose = TRUE,mtry = floor(sqrt(ncol(data))), 
                       replace = TRUE,classwt = NULL, cutoff = NULL, strata = NULL,
                       sampsize = NULL, nodesize = NULL, maxnodes = NULL,xtrue = NA, 
                       parallelize = c('no'))
Test <- Test.imp$ximp

#Create variables "Psychological Co-morbidity" and "Physical Co-morbidity", categorizing using HR variables and data dictionary
Train$Set <- "Train"
Test$Set <- "Test"
data <- rbind(Train,Test)
physical <- data[,c(15:44, 62:77)]
psychological <- data[,45:61]

physical$PhysicalComorbidity <- ifelse(physical$HR1 == "1" | physical$HR2 == "1" | physical$HR3 =="1" | physical$HR4 =="1" | 
                                         physical$HR5 =="1" | physical$HR6 =="1" | physical$HR7 =="1" | physical$HR8 =="1" | 
                                         physical$HR9 == "1" | physical$HR10 =="1" | physical$HR12 =="1" | 
                                         physical$HR13=="1"  | physical$HR14 =="1" | physical$HR15=="1" | physical$HR16=="1" | 
                                         physical$HR17=="1" | physical$HR18=="1" | physical$HR19=="1" | physical$HR20=="1" | 
                                         physical$HR21=="1" | physical$HR22=="1" | physical$HR23=="1" | physical$HR24=="1" | 
                                         physical$HR25=="1" | physical$HR26=="1" | physical$HR27=="1" | physical$HR28=="1" | 
                                         physical$HR29=="1" | physical$HR30=="1" | physical$HR31=="1" |
                                         physical$HR50=="1" | physical$HR52=="1" | physical$HR53=="1" | physical$HR54=="1" | 
                                         physical$HR55=="1" | physical$HR56=="1" | physical$HR57=="1" | physical$HR58=="1" | 
                                         physical$HR59=="1" | physical$HR60=="1" | physical$HR61=="1" | physical$HR62=="1" | 
                                         physical$HR63=="1" | physical$HR64=="1" | physical$HR65=="1", "Yes", "No")
physical$PhysicalComorbidity <- as.factor(physical$PhysicalComorbidity)

psychological$PsychologicalComorbidity <- ifelse(psychological$HR32 == "1" | psychological$HR33 == "1" | psychological$HR34 =="1" | 
                                                   psychological$HR35 =="1" | 
                                                   psychological$HR36 =="1" | psychological$HR37 =="1" | psychological$HR38 =="1" | 
                                                   psychological$HR39 =="1" | psychological$HR40 =="1" | psychological$HR41 == "1" | 
                                                   psychological$HR42 =="1" | psychological$HR43 == "1" | psychological$HR44 =="1" | 
                                                   psychological$HR45 =="1" | psychological$HR46 =="1" | psychological$HR47 =="1" | 
                                                   psychological$HR48 =="1", "Yes", "No")
psychological$PsychologicalComorbidity <- as.factor(psychological$PsychologicalComorbidity)

#Remove all HR variables and add in co-morbidity variables
data <- select(data, -c(15:77))
data$PhysicalComorbidity <- physical$PhysicalComorbidity
data$PsychologicalComorbidity <- psychological$PsychologicalComorbidity

#Put data back into original test/train groups
Train <- subset(data, Set == "Train")
Test <- subset(data, Set == "Test")
Train$Set <- NULL
Test$Set <- NULL

################################
#FEATURE SELECTION USING BORUTA
################################
set.seed(2345)
boruta.train <- Boruta(longpost_dailydose ~., data=Train, doTrace=2)
print(boruta.train)

plot(boruta.train, xlab = "", xaxt = "n")
lz <- lapply(1:ncol(boruta.train$ImpHistory),function(i)
boruta.train$ImpHistory[is.finite(boruta.train$ImpHistory[,i]),i])

names(lz) <- colnames(boruta.train$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
at = 1:ncol(boruta.train$ImpHistory), cex.axis = 0.7)

final.boruta <- TentativeRoughFix(boruta.train)
print(final.boruta)

getSelectedAttributes(final.boruta, withTentative = F)
boruta.df <- attStats(final.boruta)
class(boruta.df)
print(boruta.df)

selected <- getSelectedAttributes(final.boruta, withTentative = F)

#Subset data to selected variables
Train <- Train %>% select(c(longpost_dailydose, selected))
Test <- Test %>% select(c(longpost_dailydose, selected))

rownames(Train) <- c(1:nrow(Train))
rownames(Test) <- c(1:nrow(Test))

#Save final preprocessed dataset in case stop and pick up later
write.csv(Train, "TrainSet.csv")
write.csv(Test, "TestSet.csv")

#################################
#Begin with machine Learning
#################################
Train <- read.csv("TrainSet_Com.csv")
Test <- read.csv("TestSet_Com.csv")
Train$X <- NULL
Test$X <- NULL

#Change datatypes
convertFactor <- c(1, 3, 4, 6)
convertNum <- c(2, 5)
Train[,convertFactor] <- data.frame(apply(Train[convertFactor], 2, as.factor))
Train[,convertNum] <- data.frame(apply(Train[convertNum], 2, as.numeric))
Test[,convertFactor] <- data.frame(apply(Test[convertFactor], 2, as.factor))
Test[,convertNum] <- data.frame(apply(Test[convertNum], 2, as.numeric))

#Normalize numeric data
normalize <- function(x) {return ((x - min(x)) / (max(x) - min(x))) }
Train_normal_age <- as.data.frame(lapply(Train[2], normalize))
Test_normal_age <- as.data.frame(lapply(Test[2], normalize))
Train_normal_numDiag <- as.data.frame(lapply(Train[5], normalize))
Test_normal_numDiag <- as.data.frame(lapply(Test[5], normalize))

#Replace in original df
Train$PATAGE <- Train_normal_age$PATAGE
Train$num_diagnoses <- Train_normal_numDiag$num_diagnoses
Test$PATAGE <- Test_normal_age$PATAGE
Test$num_diagnoses <- Test_normal_numDiag$num_diagnoses

########################
#1. NAIVE BAYES MODEL
########################
Train_NB <- Train
Test_NB <- Test

y <- Train_NB$longpost_dailydose
x <- Train_NB
x$longpost_dailydose <- NULL

set.seed(400)
trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
grid <- data.frame(fL=c(0.0,0.25,0.5,0.75,1), usekernel=TRUE, adjust=c(0,0.25,0.5,0.75,1))
NBfit <- train(x=x, y=y,
             method     = "nb",
             trControl  = trControl,
             metric     = "Accuracy",
             tuneLength=10, tuneGrid=grid,
             importance=TRUE, data=Train_NB)

#See variable list
vars <- varImp(NBfit, scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#################################
# NEURAL NET
#################################
Train_NN <- Train
Test_NN <- Test

trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
NNmodel<- train(longpost_dailydose~PATAGE+presurgical_dailydose+postsurgical_dailydose+num_diagnoses+PsychologicalComorbidity, data=Train_NN, method="nnet", trControl = trControl, importance= TRUE, tuneLength=10)

#See variable list
vars <- varImp(NNmodel,scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#See tools for NN: number of hidden layers, etc.
library(NeuralNetTools)
NNmodel$finalModel

#################################
# RANDOM FOREST MODEL
#################################
Train_RF <- Train
Test_RF <- Test

y <- Train_RF$longpost_dailydose
x <- Train_RF
x$longpost_dailydose <- NULL

trControl <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
tunegrid <- expand.grid(.mtry=(1:15))
RF_fit <- train(x=x, y=y, data=Train_RF, method="rf", trControl=trControl,metric="Accuracy", 
                importance=TRUE, tuneLength=10, tuneGrid=tunegrid)

#See variable list
vars <- varImp(RF_fit,scale=TRUE)
varsImp <- as.matrix(vars$importance)
Features <- rownames(varsImp)
varsImp <- as.data.frame(varsImp)
varsImp$Features <- Features
varsImp$X0 <- NULL
colnames(varsImp) <- c("Importance", "Features")

#################################
# XG BOOST
#################################
Train_XG <- Train
Test_XG <- Test

Train_XG$longpost_dailydose <- as.numeric(Train_XG$longpost_dailydose)
Test_XG$longpost_dailydose <- as.numeric(Test_XG$longpost_dailydose)
Train_XG$longpost_dailydose <- Train_XG$longpost_dailydose - 1
Test_XG$longpost_dailydose <- Test_XG$longpost_dailydose - 1

#Create matrix and convert to sparse matrix format; create list format
train_matrix_label <- (Train_XG$longpost_dailydose)
train_matrix_vars <- data.matrix(subset(Train_XG, select = -c(longpost_dailydose)))
test_matrix_label <- (Test_XG$longpost_dailydose)
test_matrix_vars <- data.matrix(subset(Test_XG, select = -c(longpost_dailydose)))
train_sparse_vars <- as(train_matrix_vars, "dgCMatrix")
test_sparse_vars <- as(test_matrix_vars, "dgCMatrix")
train_XG <- list("data"=train_sparse_vars, "label"=train_matrix_label)
test_XG <- list("data"=test_sparse_vars, "label"=test_matrix_label)

#Manually tune parameters
#General ranges:
#eta=(0.01-0.2)
#learning_rate =(0.001-0.1)
#max_depth=(3-10)
#min_child_weight=(1-10)
#gamma=(0-1)
#sub_sample=(0.5-1)
#colsample_bytree=(0.5-1)

#Run cv tuning, cycling through options
XG_cv <- xgb.cv(data = train_XG$data, label=train_XG$label, objective = "binary:logistic", 
ntrees=2500, nrounds = 10, nthread = 5, nfold = 10, eta = 0.05, 
learning_rate=0.001, 
max_depth=3, 
min_child_weight=1, 
gamma=.1, 
sub_sample=0.8,
colsample_bytree=.7,
metrics = list("auc") )

#Create final model using top AUC in test set
set.seed(10)
XG_model <- xgboost(data = train_XG$data, label=train_XG$label, objective = "binary:logistic", 
ntrees=2500, nrounds = 10, nthread = 5, nfold = 10, eta = 0.05, 
learning_rate=0.001, 
max_depth=3, 
min_child_weight=1, 
gamma=.1, 
sub_sample=0.8,
colsample_bytree=.7,
metrics = list("auc") )

#See variable list
importance <- xgb.importance(feature_names = colnames(train_XG), model = XG_model)
importanceRaw <- xgb.importance(feature_names = colnames(train_XG), model = XG_model, data = train_XG$data, label = train_XG$label)

#################################
# ELASTIC NET
#################################
Train_EN <- Train
Test_EN <- Test

#Change to Matrix format
Train_EN_matrix <- model.matrix(data=Train_EN, ~ longpost_dailydose+PATAGE+presurgical_dailydose+postsurgical_dailydose+num_diagnoses+PsychologicalComorbidity)
Test_EN_matrix <- model.matrix(data=Test_EN, ~longpost_dailydose+PATAGE+presurgical_dailydose+postsurgical_dailydose+num_diagnoses+PsychologicalComorbidity)

#Create model
colnames(Train_EN_matrix)
ENmodel <- cv.glmnet(x=Train_EN_matrix[,c(1,3:60)], y=Train_EN_matrix[,2], family="binomial", 
                     type.measure="auc", k=10, intercept=FALSE)

#See variable list
varImpFunct <- function(object, lambda = NULL, ...) {
	beta <- predict(object, s = lambda, type = "coef")
	if(is.list(beta)) {
	out <- do.call("cbind", lapply(beta, function(x) x[,1]))
	out <- as.data.frame(out)} 
	else out <- data.frame(Overall = beta[,1])
	out <- abs(out[rownames(out) != "(Intercept)",,drop = FALSE])
	out}
varsImp <- varImpFunct(ENmodel, lambda=ENmodel$lambda.min)
int <- as.matrix(coef(ENmodel,s="lambda.min"))
rownames(int)
varsImp$Variable <- rownames(int)
varsImp <- varsImp[order(varsImp$Overall, decreasing=TRUE),]

#################################
# GRADIENT BOOSTING MACHINE MODEL
#################################
Train_GBM <- Train
Test_GBM <- Test

#Tune GBM
hyper_grid <- expand.grid(shrinkage=c(0.001, 0.01, 0.1), interaction.depth=c(1, 3, 5, 7, 9),
n.minobsinnode=c(5, 10, 15), bag.fraction=c(0.7, 0.8, 0.9), optimal_trees=0,
min_CVLoss=0)

for (i in 1:nrow(hyper_grid)) {
	set.seed(123)
	
	gbm.tune <- gbm(formula=longpost_dailydose ~ ., distribution="bernoulli", 
	data=random_train,
	n.trees=2500, interaction.depth=hyper_grid$interaction.depth[i],
	shrinkage=hyper_grid$shrinkage[i], 
	n.minobsinnode=hyper_grid$n.minobsinnode[i], bag.fraction=hyper_grid$bag.fraction[i], 			train.fraction=.8, n.cores=NULL, verbose=TRUE, cv.folds=10)

	hyper_grid$optimal_treesCV[i] <- which.min(gbm.tune$cv.error)
	hyper_grid$min_CVLoss[i] <- min(gbm.tune$cv.error) }

hyper_grid %>%
	dplyr::arrange(min_CVLoss) %>%
	head(5)

#Create final model using best parameters from output above
hyper_grid <- expand.grid(shrinkage=c(.01), interaction.depth=c(7),
n.minobsinnode=c(5), bag.fraction=c(.8), optimal_trees=0,
min_ValidLoss=0)

for (i in 1:nrow(hyper_grid)) {
	set.seed(123)
	
	gbm.final <- gbm(formula=longpost_dailydose ~ ., distribution="bernoulli", 
	data=random_train,
	n.trees=2500, interaction.depth=hyper_grid$interaction.depth[i],
	shrinkage=hyper_grid$shrinkage[i], 
	n.minobsinnode=hyper_grid$n.minobsinnode[i], bag.fraction=hyper_grid$bag.fraction[i], 			train.fraction=.8, n.cores=NULL, verbose=TRUE, cv.folds=10) }

#See variable list
varsImp <- summary(gbm.final, n.trees=best.iter_cv)

#########################
# CREATE PREDICTIONS
#########################
CaretPredNB <- predict(NBfit, newdata = Test_NB, type = "prob")
Pred_NB <- CaretPredNB[,2]
CaretPredNN <- predict(NNmodel, newdata = Test_NN, type="prob")
Pred_NN <- CaretPredNN[,2]
CaretPredRF <- predict(RF_fit, newdata = Test_RF, type="prob")
Pred_RF <- CaretPredRF[,2]
Pred_XGBoost <- predict(XG_model, test_XG$data)
Pred_EN <- predict(ENmodel, type="response", newx=Test_EN_matrix[,c(1,3:60)], s="lambda.min")
Pred_GBM <- predict(gbm.tune, newdata = Test_GBM, n.trees = best.iter_cv, type="response")

#See ROC
roc.curve(Test_NB$longpost_dailydose, Pred_NB)
roc.curve(Test_NN$longpost_dailydose, Pred_NN)
roc.curve(Test_RF$longpost_dailydose, Pred_RF)
roc.curve(Test_XG$longpost_dailydose, Pred_XGBoost)
roc.curve(Test_EN$longpost_dailydose, Pred_EN)
gbm.roc.area(Test_GBM$longpost_dailydose, Pred_GBM)

#Save predictions
Test_NB$Preds <- Pred_NB
Test_NN$Preds <- Pred_NN
Test_RF$Preds <- Pred_RF
Test_XG$Preds <- Pred_XGBoost
Test_EN$Preds <- Pred_EN
Test_GBM$Preds <- Pred_GBM

