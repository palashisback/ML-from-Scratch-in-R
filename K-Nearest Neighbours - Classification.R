library(caret)

# Load Iris dataset
df <- iris

# Create Partitioning Vector with a 70:30 split
intrain <- createDataPartition(df$Species,p = 0.7,list = F)

# Data Partitioning 
x_train <- df[intrain,1:4]
y_train <- as.character(df[intrain,5])
x_test <- df[-intrain,1:4]


# This function takes a vector and returns the Mode 
vote <- function(x){
  ux <- unique(x)
  return(ux[which.max(tabulate(match(x, ux)))])
}

# This function computes K-nearest Neighbours and calls on "vote" function to generate predictions

k_nearest <- function(x_train,y_train,x_test,k=5){
  y_pred <- rep(0,nrow(x_test)) # Empty vector for predictions
  
  # Loop through x_test
  for(i in 1:nrow(x_test)){
    test_sample <- unlist(x_test[i,]) # Isolate the sample to be predicted
    distances <- sqrt(rowSums((test_sample - x_train)^2)) # Calculate Euclidean distance to each observation in x_train
    
    # Sort distances, take indices of first k observations 
    # Extract those indices from y_train 
    k_nearest_neighbours <- y_train[sort(distances,index.return = T)$ix[1:k]] 
    
    # Call vote to calculate Mode and assign prediction
    y_pred[i] <- vote(k_nearest_neighbours)
  }
  
  return(y_pred)
}

# Example

k_nearest(x_train,y_train,x_test, k =5)
