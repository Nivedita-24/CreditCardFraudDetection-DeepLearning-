## PROJECT SUMMARY AND TASK DETAIL
#----------------------------------------------------------------------------------------------
## Goal : Identify fraudulent credit card transactions.
## Challenges:
# The data is massive with 284,807 observations and 31 variables
# The data is imbalanced, the fraudulent records only make up to 0.172% of all transactions
#----------------------------------------------------------------------------------------------
# R Packages
library(neuralnet)
library(caret)
library(dplyr)

# Reading the file
CreditCard <-read.csv("creditcard.csv")
dim(CreditCard)
str(CreditCard)
head(CreditCard)

# Summarising the dataset
summary(CreditCard)

## Data Prep
## Partitioning the dataset

index <- createDataPartition(y=CreditCard$Class, p= 0.7, list =F)
CreditCard.training <- CreditCard[index,]
CreditCard.test <- CreditCard[-index,]

## Neural Nets Using BP(Back propagation)

#Model Training
# Normalize all independent variables using the scale function.
CreditCard.training.norm <- CreditCard.training %>%mutate_at(c(1:30), funs(c(scale(.))))

# using hidden level, as 5 neurons with 3 layers (arbitrarily)
nn_model<- neuralnet(Class ~. , data = CreditCard.training.norm, hidden =c(5,3),linear.output = F)

# To show the plot, we are setting the rep argument to best to show the iteration with the smallest error.
plot(nn_model, rep ="best")

# Model Testing
CreditCard.test.norm <- CreditCard.test %>%mutate_at(c(1:30), funs(c(scale(.))))
predicted.nn.values <- neuralnet::compute(nn_model, CreditCard.test.norm)

## Lets see how the result look like
head(predicted.nn.values$net.result)

## predictions for fraud
predictions <-ifelse(predicted.nn.values$net.result[,2]>0.5,1,0)
head(predictions)

## Creating table to see the model performance
table(predictions, CreditCard.test.norm$Class)

## Accurately predicted frauds in 73% of the cases.