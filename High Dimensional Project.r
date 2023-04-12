################################ Main Code ##########################################


########## First: To obtain the best regularization parameter (C) using cross validation (5 folds) #############

rm(list = ls(all.names = TRUE))
library(caret)
library(e1071)

data1=read.csv("dataset/parkinsons.csv")
sum(is.na(data1))
data1$status[data1$status == 0] <- -1
data2=data1[,-c(1,18)]
data3=data.frame(data2,data1[,18])
colnames(data3)[23]  <- "status"
data=data3

# Split data into training and testing sets
set.seed(123)
train_idx <- sample(1:nrow(data), 0.8 * nrow(data))
train_data <- data[train_idx, ]
test_data <- data[-train_idx, ]

# Define cost values to try
cost_values <- seq(0.01, 10, length=100)

# Perform cross-validation to find best cost value
accuracy =c()
for (i in 1:length(cost_values)) {
  cost <- cost_values[i]
  folds <- cut(seq(1, nrow(train_data)), breaks = 5, labels = FALSE)
  acc=c()
  for (j in 1:5) {
    # Train model on 4 folds
      train_idx <- which(folds != j)
      train <- train_data[train_idx, ]
      # Train SVM model
      model <- svm(as.factor(status) ~ ., data = train, kernel = "radial", cost = cost)
    # Test model on remaining fold
    test_idx <- which(folds == j)
    test <- train_data[test_idx, ]
    # Make predictions
    pred <- predict(model, newdata = test[,-23])
    # Calculate accuracy
    acc[j] = sum(pred == test$status)
  }
  accuracy[i] = mean(acc)
}
cost_values[which.max(accuracy)]

######################### Second: SVM with linear and non-linear kernel ########################


rm(list = ls(all.names = TRUE))
library(caret)
library(e1071)
library(ggplot2)

data1=read.csv("dataset/parkinsons.csv")
sum(is.na(data1))
data1$status[data1$status == 0] <- -1
data2=data1[,-c(1,18)]
data3=data.frame(data2,data1[,18])
colnames(data3)[23]  <- "status"
data=data3
dim(data)
head(data)
#data$status <- factor(data$status, levels = c(-1, 1))
names(data)
str(data)

set.seed(123)
trainIndex <- createDataPartition(data$status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
dim(trainData)
dim(testData)
trainData[,-23]=scale(trainData[,-23])
testData[-23]=scale(testData[,-23])

X=trainData[,-23]
X=as.matrix(X)
y=trainData[,23]
y
#y=as.numeric(y)
#y=as.vector(y)
m=ncol(X)
n=nrow(X)

# Set hyperparameters
C <- 3.642727  #3.642727 (radial case), 4.147273 (linear case)
gamma <- 0.1   #0.1 (radial case), 0.1 (linear case)
n_iter <- 100


# Define kernel functions
linear_kernel <- function(x1, x2) {
  return(sum(x1 * x2))
}

rbf_kernel <- function(x1, x2, gamma) {
  return(exp(-gamma*sum(x1-x2)^2))
}

# Initialize Lagrange multipliers
alpha <- rep(0, n)

# Initialize biases
b <- 0

# Calculate Gram matrix
K <- matrix(0, n, n)
for (i in 1:n) {
  for (j in 1:n) {
    K[i,j] <- rbf_kernel(X[i,], X[j,], gamma)
  }
}
eta=0.1
# Perform stochastic gradient ascent
for (iter in 1:n_iter) {
  idx <- sample(n)
  for (i in 1:n) {
    j <- idx[i]
    margin <- y[j] * (sum(alpha * y * K[,j]) + b)
    if (margin < 1) {
      dL_dalpha_j <- C * y[j] * K[,j] - y[j] * sum(alpha * y * K[,j])
      #alpha[j] <- alpha[j] + alpha * dL_dalpha_j
      alpha[j] <- pmax(0, pmin(C, alpha[j] + eta * (1 - y[j] * margin) * y[j] * K[, j]))
      b <- b + alpha[j] * y[j]
      #b <- b + eta * y[j] * (1 - y[j] * margin)
    }
  }
}

# Calculate weights
w <- t(X) %*% (alpha * y)
w
dim(w)
#intercept
intercept=b
intercept

#Or
#intercept<- b <- mean(y - K %*% (alpha * y))
#intercept

# Calculate number of support vectors
n_sv <- which(alpha > 1e-6)
length(n_sv)
cat("Number of support vectors:", n_sv, "\n")

# Calculate value of objective function
obj_value <- sum(alpha) - 0.5 * t(alpha * y) %*% K %*% (alpha * y) - C * sum(ifelse(alpha > 0, alpha, 0))
cat("Value of objective function:", obj_value, "\n")


# Predict on test data
x_test <- testData[,-23]
x_test=as.matrix(x_test)
y_test <- testData[,23]

# Calculate kernel values between training and test data
K_test <- matrix(0, nrow=nrow(x_test), ncol=n)
for (i in 1:nrow(x_test)) {
  for (j in 1:n) {
    K_test[i,j] <- rbf_kernel(x_test[i,], X[j,], gamma)
  }
}

# Make predictions on test data
y_pred <- sign(K_test %*% (alpha * y) + b)

# Calculate accuracy
accuracy <- mean(y_pred == y_test)
accuracy

# Find support vectors
sv_idx <- alpha > 1e-5
sv_alpha <- alpha[sv_idx]
sv_X <- X[sv_idx, ]
sv_y <- y[sv_idx]

################ Third: To draw the relevant graph just for two variables ###################################################################################

rm(list = ls(all.names = TRUE))
library(caret)
library(ggplot2)

data1=read.csv("dataset/parkinsons.csv")
sum(is.na(data1))
data1$status[data1$status == 0] <- -1
data2=data1[,-c(1,18)]
data3=data.frame(data2,data1[,18])
colnames(data3)[23]  <- "status"
data=data3
dim(data)
head(data)
names(data)

set.seed(123)
trainIndex <- createDataPartition(data$status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
dim(trainData)
dim(testData)
trainData[,-23]=scale(trainData[,-23])
testData[-23]=scale(testData[,-23])

X=trainData[,c(3,4)]
X=as.matrix(X)
y=trainData[,23]
m=ncol(X)
n=nrow(X)

# Set hyperparameters
C <- 3.642727  #3.642727 (radial case), 4.147273 (linear case)
gamma <- 0.1   #0.1 (radial case), 0.1 (linear case)
n_iter <- 100

# Define kernel functions
linear_kernel <- function(x1, x2) {
  return(sum(x1 * x2))
}

rbf_kernel <- function(x1, x2, gamma) {
  return(exp(-gamma*sum(x1-x2)^2))
}

# Initialize Lagrange multipliers
alpha <- rep(0, n)

# Initialize biases
b <- 0

# Calculate Gram matrix
K <- matrix(0, n, n)
for (i in 1:n) {
  for (j in 1:n) {
    K[i,j] <- rbf_kernel(X[i,], X[j,], gamma)
  }
}
eta=0.1
# Perform stochastic gradient ascent
for (iter in 1:n_iter) {
  idx <- sample(n)
  for (i in 1:n) {
    j <- idx[i]
    margin <- y[j] * (sum(alpha * y * K[,j]) + b)
    if (margin < 1) {
      dL_dalpha_j <- C * y[j] * K[,j] - y[j] * sum(alpha * y * K[,j])
      #alpha[j] <- alpha[j] + alpha * dL_dalpha_j
      alpha[j] <- pmax(0, pmin(C, alpha[j] + eta * (1 - y[j] * margin) * y[j] * K[, j]))
      b <- b + alpha[j] * y[j]
      #b <- b + eta * y[j] * (1 - y[j] * margin)
    }
  }
}

# Calculate weights
w <- t(X) %*% (alpha * y)
w
dim(w)
#intercept
intercept=b
intercept

# Calculate number of support vectors
n_sv <- which(alpha > 1e-6)
length(n_sv)
cat("Number of support vectors:", n_sv, "\n")

# Calculate value of objective function
obj_value <- sum(alpha) - 0.5 * t(alpha * y) %*% K %*% (alpha * y) - C * sum(ifelse(alpha > 0, alpha, 0))
cat("Value of objective function:", obj_value, "\n")


# Predict on test data
x_test <- testData[,c(3,4)]
x_test=as.matrix(x_test)
y_test <- testData[,23]

# Calculate kernel values between training and test data
K_test <- matrix(0, nrow=nrow(x_test), ncol=n)
for (i in 1:nrow(x_test)) {
  for (j in 1:n) {
    K_test[i,j] <- rbf_kernel(x_test[i,], X[j,], gamma)
  }
}

# Make predictions on test data
y_pred <- sign(K_test %*% (alpha * y) + b)

# Calculate accuracy
accuracy <- mean(y_pred == y_test)
accuracy

# To show conducting linear nonseparable SVM with RBF kernel with just two variables
set = trainData[,c(3,4,23)]
X1 = seq(min(set[, 1]), max(set[, 1]), by = 0.06)
X2 = seq(min(X[, 2]), max(X[, 2]), by = 0.06)
length(X1)
length(X2)
grid_set = expand.grid(X1, X2)
#grid_set=as.matrix(grid_set)
dim(grid_set)
head(grid_set)
names(trainData)
colnames(grid_set) = c('MDVP.Fo.Hz.', 'MDVP.Fhi.Hz.')

# Calculate kernel values between grid points and training data
K_grid <- matrix(0, nrow = nrow(grid_set), ncol = n)
for (i in 1:nrow(grid_set)) {
  for (j in 1:n) {
    K_grid[i, j] <- rbf_kernel(grid_set[i, ], X[j, ], gamma)
  }
}

# Make predictions on grid points
y_pred_grid <- sign(K_grid %*% (alpha * y) + b)
length(y_pred_grid)
plot(set[,-3],
     main = 'Kernel SVM (Training set)',
     xlab = 'MDVP.Fo.Hz.', ylab = 'MDVP.Fhi.Hz.',
     xlim = range(X1), ylim = range(X2))
contour(X1, X2, matrix(y_pred_grid, length(X1), length(X2)), levels=c(-1, 1), labels="", lwd=2, add=TRUE)
points(grid_set, pch = '.', col = ifelse(y_pred_grid == 1, 'springgreen3', 'tomato'))
points(set, pch = 21, bg = ifelse(set[,3] == 1, 'green4', 'red3'))

b

# Plot contour
contour(X1, X2, matrix(y_pred_grid, nrow=length(X1)), levels = c(-1, 0, 1), labels = c("y = -1", "y = 0", "y = 1"))
points(X, col = ifelse(y == -1, "blue", "red"))

# Calculate separating hyperplane
w <- t(X) %*% (alpha * y)
b <- mean(y - K %*% (alpha * y))

# Plot separating hyperplane
abline(a = -b/w[2], b = -w[1]/w[2], col = "green", lwd = 2)

##################################### Forth: To visualize the results of SVM by the first two components #############################################################

rm(list = ls(all.names = TRUE))
library(caret)
library(ggplot2)
library(dplyr)
library(ggfortify)

# Load and preprocess data
data1=read.csv("dataset/parkinsons.csv")
sum(is.na(data1))
data1$status[data1$status == 0] <- -1
data2=data1[,-c(1,18)]
data3=data.frame(data2,data1[,18])
colnames(data3)[23]  <- "status"
data=data3
dim(data)
head(data)

set.seed(123)
trainIndex <- createDataPartition(data$status, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]
dim(trainData)
dim(testData)
trainData[,-23]=scale(trainData[,-23])
testData[-23]=scale(testData[,-23])

X=trainData[,-23]
X=as.matrix(X)
y=trainData[,23]
y
#y=as.numeric(y)
#y=as.vector(y)
m=ncol(X)
n=nrow(X)

# Set hyperparameters
C <- 4.147273
gamma <- 0.1
n_iter <- 100

# Define kernel functions
linear_kernel <- function(x1, x2) {
  return(sum(x1 * x2))
}

rbf_kernel <- function(x1, x2, gamma) {
  return(exp(-gamma*sum(x1-x2)^2))
}

# Initialize Lagrange multipliers
alpha <- rep(0, n)

# Initialize biases
b <- 0

# Calculate Gram matrix
K <- matrix(0, n, n)
for (i in 1:n) {
  for (j in 1:n) {
    K[i,j] <- linear_kernel(X[i,], X[j,])
  }
}
eta=0.1
# Perform stochastic gradient ascent
for (iter in 1:n_iter) {
  idx <- sample(n)
  for (i in 1:n) {
    j <- idx[i]
    margin <- y[j] * (sum(alpha * y * K[,j]) + b)
    if (margin < 1) {
      dL_dalpha_j <- C * y[j] * K[,j] - y[j] * sum(alpha * y * K[,j])
      #alpha[j] <- alpha[j] + alpha * dL_dalpha_j
      alpha[j] <- pmax(0, pmin(C, alpha[j] + eta * (1 - y[j] * margin) * y[j] * K[, j]))
      b <- b + alpha[j] * y[j]
      #b <- b + eta * y[j] * (1 - y[j] * margin)
    }
  }
}

# Calculate weights
w <- t(X) %*% (alpha * y)
w
dim(w)
#intercept
intercept=b
intercept

# Calculate number of support vectors
n_sv <- which(alpha > 1e-6)
length(n_sv)
cat("Number of support vectors:", n_sv, "\n")

# Calculate value of objective function
obj_value <- sum(alpha) - 0.5 * t(alpha * y) %*% K %*% (alpha * y) - C * sum(ifelse(alpha > 0, alpha, 0))
cat("Value of objective function:", obj_value, "\n")


# Predict on test data
x_test <- testData[,-23]
x_test=as.matrix(x_test)
y_test <- testData[,23]

# Calculate kernel values between training and test data
K_test <- matrix(0, nrow=nrow(x_test), ncol=n)
for (i in 1:nrow(x_test)) {
  for (j in 1:n) {
    K_test[i,j] <- linear_kernel(x_test[i,], X[j,])
  }
}

# Make predictions on test data
y_pred <- sign(K_test %*% (alpha * y) + b)
y_pred

# Calculate accuracy
accuracy <- mean(y_pred == y_test)
accuracy

# Perform PCA on preprocessed data
pca <- prcomp(X, center = TRUE, scale. = TRUE)
dim(pca$x)
pca_df <- as.data.frame(pca$x[,1:2])

pca_df <- pca_df[1:length(y_pred),]
pca_df$status <- factor(y_test, levels = c(-1, 1))
# Combine PCA and SVM results
pca_df$y_pred <- factor(y_pred, levels = c(-1, 1))

# Plot results
ggplot(pca_df, aes(x = PC1, y = PC2, color = y_pred, shape = status)) + 
  geom_point(size = 3) +
  scale_color_manual(values = c("#FF0000", "#0000FF")) +
  scale_shape_manual(values = c(16, 17)) +
  theme_bw()


####################################################################################################################################



















