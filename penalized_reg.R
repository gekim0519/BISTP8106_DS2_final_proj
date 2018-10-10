#############################################
#
# Penalized Logistic Regression: Ridge, Lasso
#
#############################################

data <- bc.data

library(magrittr)

# Create 10 Folds

library(caret)
cvSplits <- createFolds(data$diagnosis, 
                        k = 10, 
                        returnTrain = T)
str(cvSplits)


library(glmnet)

x <- data[,c(-1,-2,-33)] %>% data.matrix()# input matrix
y <- data$diagnosis # binary response (M:malignant, B:benign)

K <- 10
ridge.test.err <- rep(NA, K)
lasso.test.err <- rep(NA, K)

for(k in 1:K)
{
  trainInd <- cvSplits[[k]]
  
  set.seed(2018)
  
  # Find optimal lambda for ridge
  cv.fit.ridge <- cv.glmnet(x[trainInd,], y[trainInd], 
                            family = "binomial",
                            alpha = 0, 
                            type.measure = "class") # use misclassification error
  
  opt.lamb.ridge <- cv.fit.ridge$lambda.min
  
  # Find optimal lambda for lasso
  cv.fit.lasso <- cv.glmnet(x[trainInd,], y[trainInd], 
                            family = "binomial",
                            alpha = 1, 
                            type.measure = "class") 
  
  opt.lamb.lasso <- cv.fit.lasso$lambda.min
  
  # Fit ridge logistic reg. model (alpha=0)
  ridge.mod <- glmnet(x[trainInd,], y[trainInd], 
                     family = "binomial",
                     alpha = 0, 
                     lambda = opt.lamb.ridge)
  
  # Fit lasso logistic reg. model (alpha=1)
  lasso.mod <- glmnet(x[trainInd,], y[trainInd], 
                      family = "binomial",
                      alpha = 1, 
                      lambda = opt.lamb.ridge)
  
  # Predict using ridge.mod
  ridge.fit <- predict(ridge.mod, 
                       newx = x[-trainInd,],
                       type = "class") # predict diagnosis
  
  # Predict using lasso.mod
  lasso.fit <- predict(lasso.mod, 
                       newx = x[-trainInd,],
                       type = "class")
  
  # Misclassification test error: ridge
  ridge.test.err[k] = mean(ridge.fit != y[-trainInd])

  # Misclassification test error: lasso
  lasso.test.err[k] = mean(lasso.fit != y[-trainInd])
}

# 10-fold mean misclassification rate
ridge.test.err = mean(ridge.test.err)
lasso.test.err = mean(lasso.test.err) 

# Ridge has lower mean misclassification rate of 3 percent, compared to lasso's 4.2 percent



