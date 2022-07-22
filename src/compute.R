# DESCRIPTION ──────────────────────────────────────────────────────────────── #

# Functions and variable definitions related to project structure and directory
# management.

# DEPENDENCIES ─────────────────────────────────────────────────────────────── #

box::use(magrittr[`%>%`])

# FUNCTIONS ────────────────────────────────────────────────────────────────── #

#' Returns a tibble with conditional effects of spatial predictors.
#'
#' @param model An object of class brmsfit.
#' @param varnames A character vector naming effects.
#' @param resolution Number of points to use.
#' @param ndraws Number of draws from posterior.
#' @return A tibble containing estimates.
spat_condeffs <- function(model,
                          varnames = "x:y",
                          resolution = 200,
                          ndraws = 500) {
    gp_plot <- brms::conditional_effects(
        x = model,
        effects = varnames,
        ndraws = ndraws,
        resolution = resolution,
        surface = TRUE,
        robust = TRUE,
        points = TRUE,
        plot = FALSE
    )
    vars <- strsplit(varnames, split = ":")[[1]]
    spatdata <- plot(gp_plot, plot = FALSE, stype = "raster")
    estimates <- ggplot2::ggplot_build(spatdata[[1]])$plot$data %>%
        .[, c(vars[1], vars[2], "estimate__", "se__")] %>%
        tibble::as_tibble() %>%
        dplyr::rename(estimate = estimate__, se = se__)
    return(estimates)
}


log10_floor <- function(x) {
    10^(floor(log10(x)))
}

dec_places <- function(x) {
    if ((x %% 1) != 0) {
        nchar(strsplit(sub(
            "0+$", "",
            as.character(x)
        ), ".", fixed = TRUE)[[1]][[2]])
    } else {
        return(0)
    }
}
