one_hot_encoding = function(df, columns){
  # create a copy of the original data.frame for not modifying the original
  df = cbind(df)
  # convert the columns to vector in case it is a string
  columns = c(columns)
  # for each variable perform the One hot encoding
  for (column in columns){
    unique_values = sort(unique(df[column])[,column])
    non_reference_values  = unique_values[c(-1)] # the first element is going 
                                                 # to be the reference by default
    for (value in non_reference_values){
      # the new dummy column name
      new_col_name = paste0(column, '.', str_replace_all(tolower(value), " ", "_"))
      # create new dummy column for each value of the non_reference_values
      df[new_col_name] <- with(df, ifelse(df[,column] == value, 1, 0))
    }
    # delete the one hot encoded column
    df[column] = NULL

  }
  return(df)
}

get_model_formula <- function(model) {
  explanatory_variables <- tidy(model)$term[!tidy(model)$term == "(Intercept)"]
  paste("life.expectancy ~", paste(explanatory_variables, collapse = "+"))
}



