# Load package ----
library(caret)
library(tidyverse) #data manipulation and visualization
library(dplyr)
library(ggplot2)
library(class) #basic KNN
library(randomForest) #Random Forest implementation
library(stats) # Logistic regression is included in base R through the glm() function.
library(e1071) #SVM



# Load data ----

# Loading iris dataset
iris.data <- iris

# Viewing iris dataset structure and attributes
str(iris.data)

# Create a scatter plot
scatter_iris <- ggplot(iris.data, aes(x = Petal.Width, y = Petal.Length, color = Species)) +
  geom_point(size = 5, alpha = 0.6) +
  theme_classic() +
  theme(legend.position = c(0.8, 0.3))

# Save the plot as a file
ggsave(filename = "Images/scatter_iris.png", plot = scatter_iris, 
       width = 8, height = 6, dpi = 300)

# Create a boxplot for Sepal.Length by Species
boxplot_iris <- ggplot(iris.data, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Boxplot of Sepal Length by Species",
       x = "Species",
       y = "Sepal Length") +
  theme_classic()

# Save the plot as a file
ggsave(filename = "Images/boxplot_sepal_length.png", plot = boxplot_iris, 
       width = 8, height = 6, dpi = 300)



# Logistic Regression ----
# Create a binary classification problem
iris.data$is_versicolor <- ifelse(iris$Species == "versicolor", 1, 0)

# Split data into training and testing sets
set.seed(123)
train_index <- sample(1:nrow(iris.data), size = 0.7 * nrow(iris.data))
train_data <- iris.data[train_index, ]
test_data <- iris.data[-train_index, ]

# Fit logistic regression model
logistic_model <- glm(is_versicolor ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width, 
                      data = train_data, 
                      family = binomial)

# Summary of the model
summary(logistic_model)

# Predict probabilities for the test set
predicted_probs <- predict(logistic_model, newdata = test_data, type = "response")

# Convert probabilities to binary predictions
predicted_classes <- ifelse(predicted_probs > 0.5, 1, 0)

# Evaluate the model
CM_LR <- confusionMatrix(as.factor(predicted_classes), as.factor(test_data$is_versicolor))
CM_LR$table

accuracy <- mean(predicted_classes == test_data$is_versicolor)
print(paste("Accuracy:", round(accuracy * 100, 2), "%"))



# KNN ----

# Train the KNN model
k_value <- 3  # Number of neighbors
predicted_classes_knn <- knn(train = train_data[, -c(5, 6)],  # Exclude Species and is_versicolor columns
                             test = test_data[, -c(5, 6)], 
                             cl = train_data$is_versicolor, 
                             k = k_value)

# View the predicted classes
predicted_classes_knn

# Evaluate the KNN model performance
CM_KNN <- confusionMatrix(as.factor(predicted_classes_knn), as.factor(test_data$is_versicolor))
CM_KNN$table

# Calculate accuracy
accuracy_knn <- mean(predicted_classes_knn == test_data$is_versicolor)
print(paste("Accuracy:", round(accuracy_knn * 100, 2), "%"))



# Decision Tree ----




# SVM ----











