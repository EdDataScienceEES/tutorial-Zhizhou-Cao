# Load package ----
library(caret)
library(tidyverse) #data manipulation and visualization
library(dplyr)
library(ggplot2)
library(class) #basic KNN
library(randomForest) #Random Forest implementation
library(rpart)       # For decision tree
library(rpart.plot)  # For visualizing the tree
library(stats) # Logistic regression is included in base R through the glm() function.
library(e1071) #SVM
library(knitr)


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
kable(CM_LR$table, format = "html")
accuracy_LR <- mean(predicted_classes == test_data$is_versicolor)
print(paste("Accuracy:", round(accuracy_LR * 100, 2), "%"))



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
kable(CM_KNN$table, format = "html")

# Calculate accuracy
accuracy_knn <- mean(predicted_classes_knn == test_data$is_versicolor)
print(paste("Accuracy:", round(accuracy_knn * 100, 2), "%"))


# Looping for k = 1 to 10

# Create a vector to store accuracy values for each k
k_values <- 1:15  # Range of k to evaluate
accuracies <- numeric(length(k_values))  # Initialize vector for accuracies

# Loop through each k value
for (k in k_values) {
  # Apply KNN
  predicted_classes_knn <- knn(
    train = train_data[, -c(5, 6)],  # Exclude Species and is_versicolor columns
    test = test_data[, -c(5, 6)], 
    cl = train_data$is_versicolor, 
    k = k
  )
  
  # Calculate accuracy
  confusion_mat <- table(Prediction = predicted_classes_knn, Reference = test_data$is_versicolor)
  accuracy <- sum(diag(confusion_mat)) / sum(confusion_mat)  # Correct predictions / Total predictions
  accuracies[k] <- accuracy
}

# Create a data frame for visualization
accuracy_data <- data.frame(k = k_values, Accuracy = accuracies)

# Plot the accuracies
KNN_plot <- ggplot(accuracy_data, aes(x = k, y = Accuracy)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point(color = "black", size = 2, alpha = 0.5) +
  labs(
    title = "Accuracy of KNN for Different k Values",
    x = "k (Number of Neighbors)",
    y = "Accuracy"
  ) +
  ylim(0.95, 1.02)+
  theme_bw()
KNN_plot
ggsave(filename = "Images/KNN_plot.png", plot = KNN_plot)



# Decision Tree ----

# Step 1: Fit the Decision Tree Model
decision_tree <- rpart(
  is_versicolor ~ .,  # Formula: Predicting is_versicolor based on all other variables
  data = train_data,  # Training data
  method = "class"    # Classification tree
)

# Step 2: Visualize the Decision Tree
png("Images/decision_tree_plot.png", width = 800, height = 600)  # Set the file name and dimensions
rpart.plot(decision_tree, main = "Decision Tree for Classifying Versicolor")  # Plot the tree
dev.off()  # Close the graphics device

# Step 3: Make Predictions on the Test Data
predicted_classes_dt <- predict(decision_tree, test_data, type = "class")

# Step 4: Evaluate the Model
CM_dt <- confusionMatrix(as.factor(predicted_classes_dt), as.factor(test_data$is_versicolor))

# Print the Confusion Matrix
print(CM_dt$table)
kable(CM_dt$table, format = "html")
# Extract Accuracy
accuracy_dt <-as.numeric(CM_dt$overall["Accuracy"])
print(paste("Decision Tree Accuracy:", round(accuracy_dt * 100, 2), "%"))

# Random Forest
random_forest <- randomForest(is_versicolor ~ ., data = train_data)
predicted_classes_rf <- predict(random_forest, test_data)


# SVM ----

# Ensure that the outcome variable is a factor (classification)
train_data$is_versicolor <- factor(train_data$is_versicolor)
test_data$is_versicolor <- factor(test_data$is_versicolor)

# Train an SVM model using a radial basis function kernel (default kernel)
svm_model <- svm(is_versicolor ~ ., data = train_data, kernel = "radial", scale = TRUE)

# Predict the classes using the trained SVM model
predicted_classes_svm <- predict(svm_model, test_data)

# Confusion Matrix and Accuracy for SVM
CM_SVM <- confusionMatrix(as.factor(predicted_classes_svm), as.factor(test_data$is_versicolor))
CM_SVM$table
kable(CM_SVM$table, format = "html")

accuracy_svm <- as.numeric(CM_SVM$overall["Accuracy"])

# Print Accuracy
print(paste("SVM Accuracy:", round(accuracy_svm * 100, 2), "%"))

# Visualization
# Reduce data to two features
# Sepal
Sepal_train_data <- train_data[, c(1, 2, 6)]
# Train new model
Sepal_svm_model <- svm(is_versicolor ~ ., data = Sepal_train_data, kernel = "radial", scale = TRUE)

png("Images/svm_Sepal_plot.png", width = 800, height = 600)  # Set the file name and dimensions
plot(Sepal_svm_model, Sepal_train_data)
dev.off()  # Close the graphics device

# Petal
Petal_train_data <- train_data[, c(3,4, 6)]
# Train new model
Petal_svm_model <- svm(is_versicolor ~ ., data = Petal_train_data, kernel = "radial", scale = TRUE)

png("Images/svm_Petal_plot.png", width = 800, height = 600)  # Set the file name and dimensions
plot(Petal_svm_model, Petal_train_data)
dev.off()  # Close the graphics device


# Comparison-----

# Create a data frame of accuracies
accuracy_df <- data.frame(
  Method = c("Logistic Regression", "kNN", "Decision Tree", "SVM"),
  Accuracy = c(accuracy_LR, accuracy_knn, accuracy_dt, accuracy_svm)
)

# Ensure the Method column has the specified order
accuracy_df$Method <- factor(accuracy_df$Method, levels = c("Logistic Regression", "kNN", "Decision Tree", "SVM"))

# Create a bar plot
(comparison_plot <- ggplot(accuracy_df, aes(x = Method, y = Accuracy, fill = Method)) +
  geom_bar(stat = "identity", color = "black") +
  labs(
    title = "Accuracy Comparison of Different Methods",
    x = "Method",
    y = "Accuracy"
  ) +
  ylim(0, 1.055) +
  theme_minimal(base_size = 15) +
  theme(legend.position = "none") + # Hide legend as it's redundant
  geom_text(aes(label = round(Accuracy, 3)), vjust = -0.5, size = 5))

ggsave(filename = "Images/comparison_plot.png", plot = comparison_plot)






