library(tidyverse)
library(randomForest)
library(caret)
library(reshape2)

FILE_PATH <- 'data/Houses.csv'
REPORT_DATA <- list()
TARGET_COLUMNS <- c('price', 'sq', 'rooms', 'city')


load_and_clean_data_r <- function() {
  cat("--- R: 1. Data Loading and Feature Engineering ---\n")

  tryCatch({
    data <- read_csv(FILE_PATH, locale = locale(encoding = "WINDOWS-1250"),
                     show_col_types = FALSE)

    cat(paste0(" Successfully loaded ", nrow(data), " rows.\n"))

    data_clean <- data %>%
      select(all_of(TARGET_COLUMNS)) %>%
      drop_na(price, sq, rooms)

    # Enhanced Outlier Filtering
    data_final <- data_clean %>%
      filter(
        sq > 10,
        sq < 400,            # Max area 400 sq m
        rooms >= 1,
        price > 1000,
        price < 5000000      # Max price 5,000,000 PLN
      ) %>%
      mutate(
        rooms = as.integer(rooms),
        city = as.factor(city),
        price_log = log(price) # Feature Engineering: Log-transform
      )

    REPORT_DATA[['Initial Rows']] <<- nrow(data)
    REPORT_DATA[['Final Rows']] <<- nrow(data_final)

    cat(paste0("Final cleaned dataset size: ", nrow(data_final), " rows.\n"))
    return(data)

  }, error = function(e) {
    cat(paste0("❌ An error occurred during loading/cleaning: ", e$message, "\n"))
    return(NULL)
  })
}

visualize_data_r <- function(data) {
  cat("\n--- R: 2. Enhanced Data Visualization ---\n")

  plot1 <- ggplot(data, aes(x = sq, y = price_log)) +
    geom_point(alpha = 0.6, size = 1) +
    geom_smooth(method = "lm", color = "red", se = FALSE) +
    labs(title = 'Relationship Between Log(Price) and Area', x = 'Area [sq m]', y = 'Log(Price)') +
    theme_minimal()
  print(plot1)

  plot2 <- ggplot(data, aes(x = city, y = price, fill = city)) +
    geom_violin(trim = TRUE) +
    geom_boxplot(width = 0.1, fill = "white") +
    labs(title = 'Price Distribution Across Cities (Violin Plot)', x = 'City', y = 'Price [PLN]') +
    theme_minimal() +
    theme(legend.position = "none")
  print(plot2)
}


train_and_analyze_model_r <- function(data) {
  cat("\n--- R: 3. Final Model Training (Random Forest) ---\n")

  set.seed(42)
  index_train <- createDataPartition(data$price_log, p = 0.8, list = FALSE)
  train_data <- data[index_train, ]
  test_data <- data[-index_train, ]

  model_rf <- randomForest(
    price_log ~ sq + rooms + city,
    data = train_data,
    ntree = 100,
    importance = TRUE,
    do.trace = 20 #
  )

  y_pred_log <- predict(model_rf, newdata = test_data)
  y_pred <- exp(y_pred_log)
  y_test_original <- exp(test_data$price_log)

  rmse <- sqrt(mean((y_test_original - y_pred)^2))
  r2 <- cor(y_test_original, y_pred)^2

  REPORT_DATA[['RMSE']] <<- rmse
  REPORT_DATA[['R2']] <<- r2

  cat(sprintf("Model Random Forest: R2=%.4f, RMSE=%.2f PLN\n", r2, rmse))

  importance_data <- importance(model_rf)
  importance_df <- data.frame(Feature = rownames(importance_data), Importance = importance_data[, "%IncMSE"])

  city_importance <- importance_df %>% filter(grepl('city', Feature)) %>% pull(Importance) %>% sum()

  final_importances <- importance_df %>%
    filter(!grepl('city', Feature)) %>%
    bind_rows(data.frame(Feature = "city", Importance = city_importance)) %>%
    arrange(desc(Importance))

  REPORT_DATA[['Feature Importance']] <<- final_importances

  residuals <- y_test_original - y_pred

  plot3 <- ggplot(data.frame(Predicted = y_pred, Residuals = residuals),
                  aes(x = Predicted, y = Residuals)) +
    geom_point(alpha = 0.5) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(title = 'Residual Plot (Random Forest)', x = 'Predicted Price [PLN]', y = 'Residuals (Actual - Predicted)') +
    theme_minimal()
  print(plot3)

  return(model_rf)
}

display_combined_report_r <- function() {
  cat("\n--- R: 4. Combined Visual Report ---\n")

  metrics_data <- data.frame(
    Metric = c("R-squared (R²)", "RMSE"),
    RandomForest = c(
      sprintf("%.4f", REPORT_DATA$R2),
      sprintf("%.0fK PLN", REPORT_DATA$RMSE / 1000)
    )
  )
  importance_df <- REPORT_DATA[['Feature Importance']]

  p_importance <- ggplot(importance_df, aes(x = reorder(Feature, Importance), y = Importance, fill = Feature)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(title = "Feature Importance (What drives the price?)", x = "", y = "Importance Score (% IncMSE)") +
    theme_minimal() +
    theme(legend.position = "none")

  print(p_importance)

  cat("\nModel Performance Metrics:\n")
  print(metrics_data)
}

if (interactive()) {
  data_final <- load_and_clean_data_r()

  if (!is.null(data_final) && nrow(data_final) > 0) {

    visualize_data_r(data_final)

    best_model <- train_and_analyze_model_r(data_final)

    display_combined_report_r()


    if (!is.null(best_model)) {
      cat("\n--- FINAL PREDICTION ---\n")


      new_data_input <- data.frame(sq = 65, rooms = 3, city = factor("Kraków"))


      predicted_price_log <- predict(best_model, newdata = new_data_input)
      predicted_price <- exp(predicted_price_log)


      results <- data.frame(
        Feature = c('Area [sq m]', 'Rooms', 'City', 'Predicted Price (PLN)'),
        Value = c(
          new_data_input$sq,
          new_data_input$rooms,
          as.character(new_data_input$city),
          sprintf("%.0f", predicted_price)
        )
      )

      cat("Input Data and Prediction Result:\n")
      print(results)
      cat("========================================\n")
    }
  }
} else {
  data_final <- load_and_clean_data_r()
  if (!is.null(data_final) && nrow(data_final) > 0) {
    train_and_analyze_model_r(data_final)
  }
}