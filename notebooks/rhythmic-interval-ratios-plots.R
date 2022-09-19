
box::use(src / dirs[dirs])
box::use(compute = src / compute)
box::use(plt = src / plot)
box::use(readr[read_csv])
box::use(ggplot2[...])
box::use(ggtext[...])
box::use(ggtern[...])
box::use(magrittr[`%>%`])
box::use(matrixStats[colMedians])
box::use(MASS[fractions])

# DATA INGEST ──────────────────────────────────────────────────────────────── #

data <- read_csv(file.path(
    dirs$data, "derived", "interval_ratios.csv"
))



# Histogram of rhythmic ratios

breaks <- seq(0.25, 2, by = 0.25)
labs <- as.character(fractions(seq(0.25, 2, by = 0.25))) %>%
    gsub("/", ":", .) %>%
    gsub("^1$", "1:1", .) %>% 
gsub("^2$", "2:1", .)
# labs = c("1:4", "1:2", "3:4", "1:2", "5:4", "3:2")
copper = "#bd7129"
nighshadz <- "#63313c"
background = "#262626"
text_colour <- "#f2f2f2"
textsize <- 30
xtick_colour <- rep("#757575", length(breaks))
xtick_colour[which(breaks==1)] <- text_colour

data %>%
    ggplot(aes(x = (ioi_1 / (ioi_1 + ioi_2)*2))) +
    geom_vline(xintercept = 1, linetype = 2, lwd = 2, colour = "grey") +
    geom_histogram(
        aes(y = ..density..),
        binwidth = 0.05,
        fill = nighshadz,
        colour = NA,
        alpha = .85
    ) +
    geom_density(
        lwd = 2,
        linetype = 1,
        colour = text_colour,
        n = 1024,
        bw = 0.05,
        fill = NA,
        alpha = .6
    ) +
    scale_x_continuous(
        breaks = breaks,
        labels = labs,
        limits = c(breaks[1], breaks[length(breaks)])
    ) +
    scale_y_continuous(expand = expansion(
        mult = c(0, 0.2),
        add = c(0, 0)
    )) +
    labs(
        x = "Rhythmic interval ratio",
        y = "Density"
    ) +
    theme(
        legend.text = element_text(size = textsize),
        legend.title = element_text(size = textsize),
        panel.background = element_rect(fill = background),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none",
        plot.title.position = "panel",
        panel.border = element_rect(colour = text_colour, fill = NA, size = 3),
        text = element_text(colour = text_colour, size = textsize),
        plot.title = element_text(
            size = textsize + 10, face = "bold", colour = text_colour,
            hjust = 0.5, vjust = -3
        ),
        plot.subtitle = element_text(
            size = textsize + 3, colour = text_colour, face = "italic",
            hjust = 0.5, vjust = -4
        ),
        plot.background = element_rect(fill = background, color = NA ),
        plot.margin = unit(c(.8, .8, .8, .8), "cm"),
        axis.ticks.x = element_blank(),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        axis.text.x = element_text(
            colour = xtick_colour, size = textsize,
            margin = ggplot2::margin(t = 20, r = 0, b = 0, l = 0)
        ),
        axis.title.x.bottom = ggtext::element_markdown(
            margin = ggplot2::margin(t = 20, r = 0, b = 0, l = 0)
        ),
        axis.title.y.left = ggtext::element_markdown(
            margin = ggplot2::margin(t = 0, r = 20, b = 0, l = 0)
        )
    ) -> rhythm_2d

ggsave(
    rhythm_2d,
    path = dirs$figs,
    filename = "rhythmic_2d_intervals.png",
    width = 22,
    height = 22,
    units = "cm",
    dpi = 350,
    limitsize = FALSE,
    bg = "transparent"
)


# TERNARY PLOTS ────────────────────────────────────────────────────────────── #

# Actual interval times
colnames <- c("name", "ioi_1", "ioi_2", "ioi_3")
ratios <- data.frame(
    c(111, 212, 121),
    c(1, 2, 1),
    c(1, 1, 2),
    c(1, 2, 1)
)
colnames(ratios) <- colnames

# ratios <- expand.grid(X = 1:5, Y = 1:5, Z = 1:5) %>%
#     dplyr::mutate(
#         name = paste0(X, Y, Z),
#         ioi_1 = X,
#         ioi_2 = Y,
#         ioi_3 = Z
#     )

# df.mu <- as.data.frame(t(colMedians(as.matrix(data[colnames]))))
# colnames(df.mu) <- c(colnames)

breaks <- seq(0, 1, by = 0.25)
labs <- as.character(fractions(seq(0, 1, by = 0.25)))
background = "#262626"
text_colour <- "#f2f2f2"
textsize <- 18

data %>%
    ggtern(aes(x = ioi_1, y = ioi_3, z = ioi_2)) +
    stat_density_tern(
        geom = "polygon",
        aes(
            fill = ..level..
        ),
        binwidth = 3
    ) +
    # scale_T_continuous(limits = c(.1, .8), breaks = breaks, labels = breaks) +
    # scale_L_continuous(limits = c(.1, .8), breaks = breaks, labels = breaks) +
    # scale_R_continuous(limits = c(.1, .8), breaks = breaks, labels = breaks) +
    colorspace::scale_fill_continuous_sequential(
        palette = "Lajolla",
        rev = T,
        n_interp = 100,
        name = "Estimate\n",
        guide = guide_colorbar(
            ticks = FALSE,
            nbin = 1000
        )
    ) +
    geom_point(size = 0.1, color = "#ffffff", alpha = 0.05) +
    geom_point(
        data = ratios,
        shape = 3, size = 6, color = "#ffffff", stroke = 1, alpha = 1
    ) +
    geom_text(
        data = ratios, label = ratios$name, hjust = .5,
        vjust = -1.6, color = "#ffffff", size = textsize - 13
    ) +
    labs(
        title = "Ternary rhythm space",
        subtitle = "Kernel Density Estimate of interval ratios",
        x = "",
        xarrow = "1st IOI",
        y = "",
        yarrow = "3rd IOI",
        z = "",
        zarrow = "2nd IOI"
    ) +
    theme_hidegrid() +
    theme_arrowlarge() +
    theme_hidelabels() +
    theme_arrowsmall() +
    # theme_clockwise() +
    theme(
        legend.text = element_text(size = textsize),
        legend.title = element_text(size = textsize),
        panel.background = element_rect(fill = "#1d0b14"),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        legend.position = "none",
        plot.title.position = "panel",
        panel.border = element_rect(colour = text_colour, fill = NA, size = 1),
        text = element_text(size = textsize),
        plot.title = element_text(
            size = textsize + 10, face = "bold", colour = text_colour,
            hjust = 0.5, vjust = -3
        ),
        plot.subtitle = element_text(
            size = textsize + 3, colour = text_colour, face = "italic",
            hjust = 0.5, vjust = -4
        ),
        tern.axis.arrow = element_line(size = 2, color = text_colour),
        tern.axis.arrow.text = ggtext::element_markdown(
            size = textsize - 2, vjust = -1.5, colour = text_colour
        ),
        tern.axis.arrow.text.R = ggtext::element_markdown(
            size = textsize - 2, vjust = 2, colour = text_colour
        ),
        plot.background = element_rect(fill = background, color = NA),
        plot.margin = unit(c(.8, .8, .8, .8), "cm")
    ) -> rhythmplot

# rhythmplot

ggsave(
    rhythmplot,
    path = dirs$figs,
    filename = "ternary_rhythm.png",
    width = 22,
    height = 22,
    units = "cm",
    dpi = 350,
    limitsize = FALSE,
    bg = "transparent"
)


data %>%
    ggplot(aes(x = ratio_2, y = ratio_1)) +
    stat_density_2d(
        geom = "polygon",
        aes(
            fill = ..level..,
            alpha = abs(..level..)
        ),
        binwidth = 0.05
    ) +
    geom_abline(intercept = 0, linetype = 1, colour = "white", size = 1, alpha = 0.5) +
    colorspace::scale_fill_continuous_sequential(
        palette = "Lajolla",
        rev = T,
        n_interp = 30,
        name = "Estimate\n",
        guide = guide_colorbar(
            ticks = FALSE,
            nbin = 1000
        )
    ) +
    geom_point(size = 0.1, color = "#ffffff", alpha = 0.4) +
    theme(panel.background = element_rect(fill = "#000000"))
# xlim(0.5, 2) +
# ylim(0.5, 2)


data %>%
    ggplot(aes(x = ratio_1, y = ratio_0)) +
    geom_point(size = 1, fill = "#000000", alpha = 0.05) +
    geom_smooth(method = "gam", se = FALSE) +
    stat_density_2d(
        geom = "polygon",
        aes(
            fill = ..level..,
            alpha = abs(..level..)
        ),
        binwidth = 0.5
    ) +
    geom_abline(intercept = 0, linetype = 1, colour = "white", size = 1, alpha = 0.5) +
    scale_fill_viridis_c(
        option = "magma",
        alpha = 1,
        name = "Estimate",
        # breaks = colticks,
        guide = guide_colorbar(
            ticks = FALSE,
            nbin = 1000
        )
    ) +
    geom_point(size = 0.1, color = "#ffffff", alpha = 0.4) +
    theme(panel.background = element_rect(fill = "#000000"))
