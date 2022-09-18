
box::use(src / dirs[dirs])
box::use(compute = src / compute)
box::use(plt = src / plot)
box::use(readr[read_csv])
box::use(ggplot2[...])
box::use(ggtern[...])
box::use(magrittr[`%>%`])
box::use(matrixStats[colMedians])
box::use(MASS[fractions])

# DATA INGEST ──────────────────────────────────────────────────────────────── #

data <- read_csv(file.path(
    dirs$data, "derived", "peak_freqs.csv"
))

# TERNARY PLOTS ────────────────────────────────────────────────────────────── #

# Actual interval times
colnames <- c("name", "freq_1", "freq_2", "freq_3")
ratios <- data.frame(
    c(111, 454, 545),
    c(1, 4, 5),
    c(1, 5, 4),
    c(1, 4, 5)
)

colnames(ratios) <- colnames
breaks <- seq(0, 1, by = 0.25)
labs <- as.character(fractions(seq(0, 1, by = 0.25)))
background = "#262626"
text_colour <- "#f2f2f2"
textsize <- 18

data %>%
    ggtern(aes(x = freq_1, y = freq_3, z = freq_2)) +
    stat_density_tern(
        geom = "polygon",
        aes(
            fill = ..level..
        ),
        binwidth = 3
    ) +
    scale_T_continuous(limits = c(.2, .6), breaks = breaks, labels = breaks) +
    scale_L_continuous(limits = c(.2, .6), breaks = breaks, labels = breaks) +
    scale_R_continuous(limits = c(.2, .6), breaks = breaks, labels = breaks) +
    scale_fill_viridis_c(
        option = "magma",
        alpha = 1,
        name = "Estimate",
        # breaks = colticks,
        guide = guide_colorbar(
            ticks = FALSE,
            nbin = 80
        )
    ) +
    geom_point(size = 0.1, color = "#ffffff", alpha = 0.05) +
    geom_point(
        data = ratios,
        shape = 3, size = 6, color = "#ffffff", stroke = 1, alpha = 1
    ) +
    geom_text(
        data = ratios, label = ratios$name, hjust = 0,
        vjust = -1, color = "#ffffff", size = textsize - 13
    ) +
    labs(
        title = "Ternary melodic space",
        subtitle = "Kernel Density Estimate of melodic ratios",
        x = "",
        xarrow = "1st frequency",
        y = "",
        yarrow = "3rd frequency",
        z = "",
        zarrow = "2nd frequency"
    ) +
    theme_hidegrid() +
    theme_arrowlarge() +
    theme_hidelabels() +
    theme_arrowsmall() +
    # theme_clockwise() +
    theme(
        legend.text = element_text(size = textsize),
        legend.title = element_text(size = textsize),
        panel.background = element_rect(fill = "#000000"),
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
        plot.margin = margin(1, 1, 1, 1, "cm")
    ) -> freqplot

freqplot

ggsave(
    freqplot,
    path = dirs$figs,
    filename = "ternary_melody.png",
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
