library(dplyr)
library(tibble)
library(glmnet)
library(caret)
library(randomForest)

df <- read.csv("~/Heart_disease_prediction_project/bankloan.csv")
tibble::view(df)

# turning the categorical values into REAL categorical values
df_cat <- df |> 
  dplyr::mutate(Personal.Loan = as.factor(Personal.Loan),
                Securities.Account = as.factor(Securities.Account),
                CD.Account = as.factor(CD.Account),
                Online = as.factor(Online),
                CreditCard = as.factor(CreditCard)) |> 
  tibble::view()


# SEPARATING DATA
set.seed(123)
num_rows <- nrow(df_cat)
print(num_rows)

train_indexes <- sample(1:num_rows, 0.8 * num_rows, replace = FALSE)

train_set <- df_cat[train_indexes, ]
test_set <- df_cat[-train_indexes, ]

x_train <- model.matrix(~., data = train_set[, !colnames(train_set) %in% c("Personal.Loan")])
y_train <- as.numeric(as.character(train_set$Personal.Loan))

x_test <- model.matrix(~., data = test_set[, !colnames(test_set) %in% c("Personal.Loan")])
y_test <- as.numeric(as.character(test_set$Personal.Loan))


# LOGISTIC REGRESION

logistic_model <- glm(Personal.Loan ~ ., data = train_set, family = "binomial")
logistic_predictions <- predict(logistic_model, newdata = test_set, type = "response")
predicted_classes <- ifelse(logistic_predictions > 0.5, 1, 0)
accuracy_logistic <- mean(predicted_classes == test_set$Personal.Loan)
print(paste("Logistic Regression Accuracy:", accuracy_logistic))


# APPLYING LASSO REGRESSION
lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
plot(lasso_model)

prediction_lasso <- predict(lasso_model, newx = x_test, s = "lambda.min", type = "response")
predicted_classes_lasso <- as.numeric(prediction_lasso > 0.5)

accuracy_lasso <- mean(predicted_classes_lasso == y_test)

print(paste("LASSO regression accuracy on test dataset:", round(accuracy_lasso, 4)))

conf_matrix_lasso <- confusionMatrix(factor(predicted_classes_lasso), factor(y_test))
print("Confusion Matrix (LASSO):")
print(conf_matrix_lasso)


# APPLYING RIDGE REGRESSION
ridge_model <- cv.glmnet(x_train, y_train, alpha=0)
plot(ridge_model)

prediction <- predict(ridge_model, newx=  x_test, s = "lambda.min", type = "response")
predicted_classes <- as.numeric(prediction > 0.5)

accuracy <- mean(predicted_classes == y_test)

print(paste("Ridge regresion accuracy on test dataset:", round(accuracy, 4)))

conf_matrix_ridge <- confusionMatrix(factor(predicted_classes), factor(y_test))
print("Confusion Matrix (Ridge):")
print(conf_matrix_ridge)


# APPLYING ELASTIC NET
elastic_net_model <- cv.glmnet(x_train, y_train, alpha = 0.5)
plot(elastic_net_model)

prediction_en <- predict(elastic_net_model, newx = x_test, s = "lambda.min", type = "response")
predicted_classes_en <- as.numeric(prediction_en > 0.5)

accuracy_en <- mean(predicted_classes_en == y_test)
print(paste("Elastic Net accuracy on test data:", round(accuracy_en, 4)))

# confusion matrix
conf_matrix_en <- confusionMatrix(factor(predicted_classes_en), factor(y_test))
print("Confusion Matrix:")
print(conf_matrix_en)


# RANDOM FOREST
train_set$Personal.Loan <- as.factor(train_set$Personal.Loan)
test_set$Personal.Loan <- as.factor(test_set$Personal.Loan)
x_train <- model.matrix(~., data = train_set[, !colnames(train_set) %in% c("Personal.Loan")])
y_train <- train_set$Personal.Loan

x_test <- model.matrix(~., data = test_set[, !colnames(test_set) %in% c("Personal.Loan")])
y_test <- test_set$Personal.Loan

rf_model <- randomForest(
  x = x_train,
  y = y_train,
  ntree = 100,
  mtry = sqrt(ncol(x_train)),
  importance = TRUE
)

prediction_rf <- predict(rf_model, newdata = x_test)
results <- data.frame(Real = y_test, Predicted = prediction_rf)
accuracy_rf <- sum(results$Real == results$Predicted) / length(results$Real)
print(paste("Random Forest model accuracy:", accuracy_rf))




