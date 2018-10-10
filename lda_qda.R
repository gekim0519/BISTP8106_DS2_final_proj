###########
#
# LDA, QDA
#
############

library(pROC)
library(MASS)

data = data[,-33] # delete column of na's

# Create 10 Folds

library(caret)
cvSplits <- createFolds(data$diagnosis, 
                        k = 10, 
                        returnTrain = T)
str(cvSplits)

K <- 10
lda.test.err <- rep(NA, K)
qda.test.err <- rep(NA, K)

for(k in 1:K)
{
  trainInd <- cvSplits[[k]]
  
  set.seed(2018)
  
  # LDA 
  lda.model = lda(diagnosis~., 
                  data = data, 
                  subset = trainInd)
  
  # QDA
  qda.model = qda(diagnosis~., 
                  data = data, 
                  subset = trainInd)
  
  # Predict using LDA
  lda.fit = predict(lda.model,
                    newdata = data[-trainInd,],
                    type = "class")
  
  # Predict using QDA
  qda.fit = predict(qda.model,
                    newdata = data[-trainInd,],
                    type = "class")
  
  # Misclassification rate: LDA
  lda.test.err[k] = mean(lda.fit$class != data$diagnosis[-trainInd])
  
  # Misclassification rate: QDA
  qda.test.err[k] = mean(qda.fit$class != data$diagnosis[-trainInd])
  
}

# 10-fold mean misclassification rate
lda.test.err = mean(ridge.test.err)
qda.test.err = mean(lasso.test.err) 

# LDA has lower mean misclassification rate of 3 percent, compared to QDA's 4.2
