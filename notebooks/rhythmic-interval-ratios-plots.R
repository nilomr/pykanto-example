
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
    dirs$data, "derived", "interval_ratios.csv"
))

# TERNARY PLOTS ────────────────────────────────────────────────────────────── #

# Actual interval times
colnames <- c("name", "ioi_1", "ioi_2", "ioi_3")
ratios <- data.frame(
    c(111, 323, 232, 212, 121),
    c(1, 3, 2, 2, 1),
    c(1, 2, 3, 1, 2),
    c(1, 3, 2, 2, 1)
)
colnames(ratios) <- colnames


# df.mu <- as.data.frame(t(colMedians(as.matrix(data[colnames]))))
# colnames(df.mu) <- c(colnames)

breaks <- seq(0, 1, by = 0.25)
labs <- as.character(fractions(seq(0, 1, by = 0.25)))

data %>%
    ggtern(aes(x = ioi_1, y = ioi_3, z = ioi_2)) +
    stat_density_tern(
        geom = "polygon",
        aes(
            fill = ..level..
        ),
        binwidth = 0.1
    ) +
    theme_hidegrid() +
    theme_arrowlarge() +
    theme_clockwise() +
    scale_T_continuous(limits = c(.15, .7), breaks = breaks, labels = labs) +
    scale_L_continuous(limits = c(.15, .7), breaks = breaks, labels = labs) +
    scale_R_continuous(limits = c(.15, .7), breaks = breaks, labels = labs) +
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
    geom_point(size = 0.1, color = "#ffffff", alpha = 0.1) +
    geom_point(
        data = ratios,
        shape = 3, size = 5, color = "#ffffff", stroke = 1, alpha = 1
    ) +
    geom_text(
        data = ratios, label = ratios$name, hjust = 0,
        vjust = -1, color = "#ffffff"
    ) +
    theme(panel.background = element_rect(fill = "#000000"))

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
