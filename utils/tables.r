compute_moments_feature <- function(feature_name, feature_data, decimals = 2) {
  summary_feature <- data.frame(
    variable = feature_name,
    mean = round(mean(feature_data), decimals),
    std_dev = round(sd(feature_data), decimals),
    skewness = round(skewness(feature_data), decimals),
    kurtosis = round(kurtosis(feature_data), decimals)
  )
  return(summary_feature)
}

summary_moments <- function(df) {
  summary <- c()
  for (feature in colnames(df)) {
    summary <- bind_rows(summary, compute_moments_feature(feature, df[[feature]]))
  }
  return(summary)
}

model_comparison_table <- function(model, name) {
  model_summary <- glance(model)
  
  return(data.frame(
    Model = name,
    Number_of_variables = length(tidy(model)$term) - 1,
    AIC = round(model_summary$AIC, 2),
    #BIC = round(model_summary$BIC, 2), 
    #Residuals_deviance = round(model_summary$deviance, 2),
    Adjusted_R2 = round(model_summary$adj.r.squared, 2)
  ))
}