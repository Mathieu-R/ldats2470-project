histplot <- function(df, feature) {
  ggplot(df, aes({{feature}})) +
    geom_histogram(color = "#FFFFFF", fill = "#3F6D9B") +
    labs(x = "", y = "Count") +
    ggtitle(rlang::englue("{{feature}}")) +
    theme_classic()
}

bplot <- function(df, feature) {
  ggplot(df, aes(x = 1, y = {{feature}})) +
    geom_boxplot() +
    stat_summary(fun = "mean", geom = "point", shape = 23, size = 3, fill = "white") +
    scale_x_continuous(breaks = NULL) +
    theme(axis.title.x = element_blank()) +
    theme_minimal()
}

scplot_residuals_variable <- function(model, variable) {
  ggplot(augment(model), aes({{variable}}, y = .resid)) +
    geom_point() +
    labs(x = rlang::englue("{{variable}}"), y = "residuals") +
    theme_minimal()
}

scplot_residuals_fitted <- function(model, variable) {
  ggplot(augment(model), aes({{variable}}, y = .fitted)) +
    geom_point() +
    labs(x = rlang::englue("{{variable}}"), y = "fitted") +
    theme_minimal()
}