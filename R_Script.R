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


# KNN ----

# Decision Tree ----




# SVM ----











