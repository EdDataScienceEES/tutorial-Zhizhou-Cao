<center><img src="/Images/background.jpeg" alt="Img"/></center>

::: {style="text-align: center;"}
```         
<img src="/Images/background.jpeg" alt="Img"/>
```
:::

### Tutorial Aims

#### <a href="#section1"> 1. Introduction to Machine Learning</a>

#### <a href="#section2"> 2. Supervised Learning Algorithms</a>

##### <a href="#section2-1"> 2.1 Logistic Regression</a>

##### <a href="#section2-2"> 2.2 K-Nearest Neighbors (KNN)</a>

##### <a href="#section2-3"> 2.3 Decision Trees/Random Forests</a>

##### <a href="#section2-4"> 2.4 Support Vector Machines (SVM)</a>

#### <a href="#section3"> 3. Comparison and Summary</a>


<a name="section1"></a>

## 1. What is Machine Learning?

<a href="https://en.wikipedia.org/wiki/Machine_learning" target="_blank">Machine learning (ML)</a> is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Today,this technologies have become some of the biggest players in the world of artificial intelligence and computer science.\
In other words, just like repeatly showing items to a children to help them recognise, Machine learning makes computers more intelligent without explicitly teaching them how to behave.

There are **four types** of Machine Learning algorithms,

| Types | Description |
|------------------------------------|------------------------------------|
| <a href="https://en.wikipedia.org/wiki/Machine_learning" target="_blank"> Supervised Learning</a> | Supervised learning involves training a model on labeled data, where the desired output is known. The model learns to map inputs to outputs based on the provided examples. |
| <a href="https://en.wikipedia.org/wiki/Unsupervised_learning" target="_blank">Unsupervised Learning</a> | Unsupervised learning works with unlabeled data and aims to find hidden patterns or intrinsic structures in the input data. |
| <a href="https://en.wikipedia.org/wiki/Reinforcement_learning" target="_blank">Reinforcement Learning</a> | Reinforcement learning involves training agents to make a sequence of decisions by rewarding them for good actions and penalizing them for bad ones. |
| <a href="https://en.wikipedia.org/wiki/Ensemble_learning" target="_blank">Ensemble Learning</a> | Ensemble learning combines multiple models to improve performance by leveraging the strengths of each model. |



### Have you ever wondered how to recognise different species of iris flowers?

<center><img src="/Images/iris.png" alt="Img"/></center>

While botanists use physical characteristics like petal and sepal measurements, in this tutorial, we will focus on teaching a computer to do the same! By leveraging the powerful R programming language, we’ll guide you through the process of using data analysis and classification techniques to identify iris species. Whether you’re a curious beginner or an experienced data enthusiast, this tutorial will equip you with the tools to build your own flower-recognition model.

The `Iris` dataset, a cornerstone in the field of data science and machine learning, serves as a classic example for exploring data analysis and classification techniques. Collected by the botanist *Edgar Anderson*, this dataset provides measurements of sepal and petal dimensions for three iris species: Iris setosa, Iris versicolor, and Iris virginica. In this tutorial, we will harness the power of R to classify iris flowers based on their unique features. Whether you're a beginner looking to enhance your R skills or a data enthusiast eager to delve into supervised learning, this guide will walk you through each step of the process, blending theory with practical implementation.**Again, in this tutorial, we will only cover four algorithms in supervised learning.**

##### Load packages

``` r
# Instal package if it's not done yet
install.packages('package name')

# Loading required packages for this tutorial
library(caret)
library(tidyverse) #data manipulation and visualization
library(dplyr)
library(ggplot2)
library(class) #basic KNN
library(randomForest) #Random Forest implementation
library(stats) # Logistic regression is included in base R through the glm() function.
library(e1071) #SVM

```

##### Load dataset

``` r
# Loading iris dataset
iris.data <- iris

# Viewing iris dataset structure and attributes
str(iris.data)
```
There are 150 observations in total.
Before we look into the algorithms, first take a brief look through the data set.

```r
# Create a scatter plot
scatter_iris <- ggplot(iris.data, aes(x = Petal.Width, y = Petal.Length, color = Species)) +
  geom_point(size = 5, alpha = 0.6) +
  theme_classic() +
  theme(legend.position = c(0.8, 0.3))
scatter_iris  
  
# Create a boxplot for Sepal.Length by Species
boxplot_iris <- ggplot(iris.data, aes(x = Species, y = Sepal.Length, fill = Species)) +
  geom_boxplot() +
  labs(title = "Boxplot of Sepal Length by Species",
       x = "Species",
       y = "Sepal Length") +
  theme_classic()
boxplot_iris 
```
Ensure to call the plot name again so it can be displayed on the Plots panel. 

<center><img src="/Images/scatter_iris.png" alt="Img"/></center>

<center><img src="/Images/boxplot_sepal_length.png" alt="Img"/></center>

From the above two plots, we can see the same species are tend to cluster together.  Now that we know that there is a clear difference in structural traits between species.

<a name="section2"></a>

## 2. Supervised Learning


<a name="section2-1"></a>

## 2.1 K NEAREST NEIGHBOURS

2.12.1

<a name="section2-2"></a>

## 2.2 K-Nearest Neighbors (KNN)

2.22.2

<a name="section2-3"></a>

## 2.3 Decision Trees/Random Forests

2.32.3

<a name="section2-4"></a>

## 2.4 Support Vector Machines (SVM)

2.42.4

<a name="section3"></a>

## 3. The third section

More text, code and images.

This is the end of the tutorial. Summarise what the student has learned, possibly even with a list of learning outcomes. In this tutorial we learned:

##### - how to generate fake bivariate data

##### - how to create a scatterplot in ggplot2

##### - some of the different plot methods in ggplot2

We can also provide some useful links, include a contact form and a way to send feedback.

For more on `ggplot2`, read the official <a href="https://www.rstudio.com/wp-content/uploads/2015/03/ggplot2-cheatsheet.pdf" target="_blank">ggplot2 cheatsheet</a>.

Everything below this is footer material - text and links that appears at the end of all of your tutorials.

<hr>

<hr>

#### Check out our <a href="https://ourcodingclub.github.io/links/" target="_blank">Useful links</a> page where you can find loads of guides and cheatsheets.

#### If you have any questions about completing this tutorial, please contact us on [ourcodingclub\@gmail.com](mailto:ourcodingclub@gmail.com){.email}

#### <a href="INSERT_SURVEY_LINK" target="_blank">We would love to hear your feedback on the tutorial, whether you did it in the classroom or online!</a>

<ul class="social-icons">

<li>

<h3><a href="https://twitter.com/our_codingclub" target="_blank"> Follow our coding adventures on Twitter! <i class="fa fa-twitter"></i></a></h3>

</li>

</ul>

###   Subscribe to our mailing list:

::: container
```         
<div class="block">
    <!-- subscribe form start -->
    <div class="form-group">
        <form action="https://getsimpleform.com/messages?form_api_token=de1ba2f2f947822946fb6e835437ec78" method="post">
        <div class="form-group">
            <input type='text' class="form-control" name='Email' placeholder="Email" required/>
        </div>
        <div>
                        <button class="btn btn-default" type='submit'>Subscribe</button>
                    </div>
                </form>
    </div>
</div>
```
:::
