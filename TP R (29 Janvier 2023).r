url <- "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv"
df <- read.csv(url)

head(df)

library(ggplot2)
library(dplyr)

# Visualisation de la distribution des prix
ggplot(df, aes(x = price)) +
  geom_histogram(fill = "blue", bins = 30) +
  labs(x = "Price", y = "Count", title = "Distribution of House Prices")

# Visualisation de la relation entre le nombre de chambres et le prix
ggplot(df, aes(x = bedrooms, y = price)) +
  geom_point(color = "blue") +
  labs(x = "Bedrooms", y = "Price", title = "Number of Bedrooms vs Price")

# Calcul de la corrélation entre les variables numériques
cor_matrix <- cor(df[, c("price", "bedrooms", "bathrooms", "sqft_living")])
print(cor_matrix)

# Module 3: Modèle de régression linéaire
# Importation des bibliothèques
library(caret)
library(ggplot2)

# Sélection des variables d'intérêt
df_selected <- df[, c("bedrooms", "bathrooms", "sqft_living", "floors", "waterfront", "view", "grade", "yr_built", "zipcode")]

# Suppression des valeurs manquantes
df_selected <- na.omit(df_selected)

# Création de l'ensemble d'apprentissage et de test
set.seed(123)
train_index <- createDataPartition(df_selected$price, p = 0.7, list = FALSE)
train_data <- df_selected[train_index, ]
test_data <- df_selected[-train_index, ]

# Construction du modèle de régression linéaire
linear_model <- lm(price ~ ., data = train_data)

# Affichage des coefficients du modèle
summary(linear_model)

# Prédiction sur l'ensemble de test
test_predictions <- predict(linear_model, newdata = test_data)

# Calcul de l'erreur quadratique moyenne (RMSE)
rmse <- sqrt(mean((test_predictions - test_data$price)^2))
print(rmse)

# Visualisation des prédictions par rapport aux valeurs réelles
ggplot(data = test_data, aes(x = price, y = test_predictions)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual Price", y = "Predicted Price", title = "Actual vs Predicted House Prices")

# Module 4: Amélioration du modèle de régression linéaire
# Importation des bibliothèques
library(caret)

# Création de la transformation de données
preprocess <- preProcess(train_data[, -1], method = c("center", "scale", "knnImpute"))

# Application de la transformation aux ensembles d'apprentissage et de test
train_data_processed <- predict(preprocess, newdata = train_data[, -1])
test_data_processed <- predict(preprocess, newdata = test_data[, -1])

# Construction du modèle de régression linéaire amélioré
linear_model_improved <- lm(price ~ ., data = train_data_processed)

# Affichage des coefficients du modèle amélioré
summary(linear_model_improved)

# Prédiction sur l'ensemble de test
test_predictions_improved <- predict(linear_model_improved, newdata = test_data_processed)

# Calcul de l'erreur quadratique moyenne (RMSE) du modèle amélioré
rmse_improved <- sqrt(mean((test_predictions_improved - test_data$price)^2))
print(rmse_improved)

# Visualisation des prédictions du modèle amélioré par rapport aux valeurs réelles
ggplot(data = test_data, aes(x = price, y = test_predictions_improved)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual Price", y = "Predicted Price", title = "Actual vs Predicted House Prices (Improved Model)")

# Module 5: Régression polynomiale
Voici la suite du code R équivalent pour le contenu de Peer-graded Assignment_ House Sales in King County.ipynb :

```R
# Importation des bibliothèques
library(caret)

# Création de la transformation de données avec une régression polynomiale d'ordre 2
preprocess_poly <- preProcess(train_data[, -1], method = c("center", "scale", "knnImpute", "poly"), degree = 2)

# Application de la transformation aux ensembles d'apprentissage et de test
train_data_processed_poly <- predict(preprocess_poly, newdata = train_data[, -1])
test_data_processed_poly <- predict(preprocess_poly, newdata = test_data[, -1])

# Construction du modèle de régression linéaire avec régression polynomiale d'ordre 2
linear_model_poly <- lm(price ~ ., data = train_data_processed_poly)

# Affichage des coefficients du modèle avec régression polynomiale d'ordre 2
summary(linear_model_poly)

# Prédiction sur l'ensemble de test avec régression polynomiale d'ordre 2
test_predictions_poly <- predict(linear_model_poly, newdata = test_data_processed_poly)

# Calcul de l'erreur quadratique moyenne (RMSE) avec régression polynomiale d'ordre 2
rmse_poly <- sqrt(mean((test_predictions_poly - test_data$price)^2))
print(rmse_poly)

# Visualisation des prédictions du modèle avec régression polynomiale d'ordre 2 par rapport aux valeurs réelles
ggplot(data = test_data, aes(x = price, y = test_predictions_poly)) +
  geom_point(color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  labs(x = "Actual Price", y = "Predicted Price", title = "Actual vs Predicted House Prices (Polynomial Model)")

# Module 6: Résumé et conclusions
summary_df <- data.frame(Model = c("Linear Regression", "Improved Linear Regression", "Polynomial Regression"),
                         RMSE = c(rmse, rmse_improved, rmse_poly))
print(summary_df)