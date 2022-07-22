# DESCRIPTION ──────────────────────────────────────────────────────────────── #

# Functions and variable definitions related to project structure and directory
# management.

# DEPENDENCIES ─────────────────────────────────────────────────────────────── #

box::use(magrittr[`%>%`])
box::use(. / compute[dec_places, log10_floor])

# FUNCTIONS ────────────────────────────────────────────────────────────────── #

#' Expands a spatial dataframe along one axis by duplicating edge values
#'
#' @param df A tibble with columns ('x', 'y', 'estimate', 'se')
#' @param axis Integer, one of (0, 1): Axis to expand.
#' @param by An integer: length to expand, e.g. in metres.
#' @return A tibble
expand_1d <- function(df,
                      axis = 0,
                      position = "first",
                      by = 100) {
    uniquex <- unique(if (axis == 1) df$x else df$y)
    if (position == "first") {
        v <- min(if (axis == 1) df$x else df$y)
        xseq <- seq(v, v - by, (uniquex[1] - uniquex[2]))
    } else if (position == "last") {
        v <- max(if (axis == 1) df$x else df$y)
        xseq <- seq(v, v + by, -1 * (uniquex[1] - uniquex[2]))
    } else {
        stop("Position must be 'first' or 'last'")
    }
    ex <- df[if (axis == 1) df$x == v else df$y == v, ]
    if (axis == 1) {
        tobind <- tibble::tibble(
            x = rep(xseq, each = length(ex$y)),
            y = rep(ex$y, times = length(xseq))
        )
    } else {
        tobind <- tibble::tibble(
            y = rep(xseq, each = length(ex$y)),
            x = rep(ex$x, times = length(xseq))
        )
    }
    tobind <- tobind %>% dplyr::mutate(
        estimate = rep(ex$estimate, times = length(xseq)),
        se = rep(ex$se, times = length(xseq))
    )
    return(tobind)
}


#' Expands the size of a spatial dataframe
#'
#' @param df A tibble with columns ('x', 'y', 'estimate', 'se')
#' @param by An integer: length to expand, e.g. in metres.
#' @return A tibble
expand_rast <- function(df, by = 100) {
    # Expand borders by repeating values:
    its <- c("first" = 1, "first" = 0, "last" = 1, "last" = 0)
    margins <- list()
    for (i in seq(length((its)))) {
        margins[[i]] <- expand_1d(df,
            axis = its[i],
            position = names(its[i]), by = by
        )
    }
    margins[[5]] <- df
    dfbind <- do.call(rbind, margins)
    return(dfbind)
}

#' Calculate plot ticks
#'
#' @param plot_type One of ('estimate', 'se')
#' @param df A tibble with columns ('x', 'y', 'estimate', 'se')
#' @param nticks Number of ticks for legend
#' @return A tibble
plot_ticks <- function(plot_type, df, nticks) {
    x_max <- max(df[[plot_type]])
    x_min <- min(df[[plot_type]])
    d <- dec_places(log10_floor((x_max - x_min) / nticks))
    colticks <- round(seq(
        f = x_min, t = x_max,
        by = round((x_max - x_min) / nticks, digits = d)
    ), digits = d)
    return(colticks)
}

#' Set plot theme
#'
#' @return a ggplot2 theme object
settheme <- function(text.size = 8, text.colour = "#262626",
                     back.fill = "#f0f0f0") {
    return(
        ggplot2::theme(
            text = ggplot2::element_text(size = text.size),
            panel.grid.major = ggplot2::element_blank(),
            panel.grid.minor = ggplot2::element_blank(),
            panel.border = ggplot2::element_rect(
                colour = text.colour,
                fill = NA, size = 1
            ),
            panel.background =
                ggplot2::element_rect(fill = back.fill, color = NA),
            axis.ticks.y = ggplot2::element_blank(),
            axis.ticks.x = ggplot2::element_blank(),
            axis.text.x = ggplot2::element_text(colour = text.colour),
            axis.text.y = ggplot2::element_text(colour = text.colour),
            plot.title = ggplot2::element_text(
                size = text.size + 6, face = "bold",
                colour = text.colour
            ),
            plot.title.position = "panel",
            plot.subtitle = ggtext::element_markdown(
                size = text.size, colour = text.colour
            ),
            plot.caption = ggplot2::element_text(size = 8, hjust = 0),
            plot.caption.position = "plot",
            plot.background =
                ggplot2::element_rect(fill = back.fill, color = NA),
            legend.key.width = ggplot2::unit(0.5, "cm"),
            legend.position = c(.9, .75),
            legend.background =
                ggplot2::element_rect(fill = "transparent", color = NA),
            legend.box.background =
                ggplot2::element_rect(fill = "transparent", color = NA),
            legend.text =
                ggplot2::element_text(color = text.colour, size = text.size),
            legend.title =
                ggplot2::element_text(color = text.colour, size = text.size - 1)
        )
    )
}
