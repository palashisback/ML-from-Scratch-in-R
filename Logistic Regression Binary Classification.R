library(caret)
library(MASS)

# Load Iris dataset
df <- iris
df$Species <- ifelse(df$Species == 'setosa',1,0)

# Create Partitioning Vector with a 70:30 split
intrain <- createDataPartition(df$Species,p = 0.7,list = F)

# Data Partitioning 
x_train <- df[intrain,1:4]
y_train <- df[intrain,5]
x_test <- df[-intrain,1:4]
y_test <- df[-intrain,5]

logistic <- function(x_train,y_train,x_test,n_iterations = 4000,learning_rate = 0.1,gradient_descent = T){
  limit = 1/sqrt(ncol(x_train))
  param = matrix(runif(min=-1*limit,max=limit,n=ncol(x_train)),ncol = 4)
  x_train <- scale(x_train)
  
  for(i in 1:n_iterations){
    y_pred = rowSums(1/(1+exp(-1*(x_train %*% t(param)))))
    if(gradient_descent){
      param = param - (learning_rate * (-1*((y_train - y_pred) %*% x_train)))
    }
    
    else{
      diag_gradient <- (1/(1+exp(-1*(x_train %*% param)))) * (1- (1/(1+exp(-1*(x_train %*% param)))))
      diag_gradient <- diag(diag_gradient[,1])
      param = ginv((t(x_train) %*% diag_gradient) %*% x_train) %*% t(x_train) %*% (diag_gradient %*% x_train %*% param + y_train - y_pred)
      }
  }
  
  y_pred <- round(rowSums(1/(1+exp(-1*(scale(x_test) %*% t(param))))))
  return(y_pred)
}

confusionMatrix(data = factor(logistic(x_train,y_train,x_test = x_train)),reference = factor(y_train))
