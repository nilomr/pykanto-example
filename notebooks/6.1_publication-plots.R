box::use(
  ggplot2[...], dplyr, readr[read_csv], ggdist[stat_dots], here, beeswarm,
  shades, patchwork[...], svglite
)

# DATA INGEST ──────────────────────────────────────────────────────────────── #

# find repository root
DATASET_ID <- "pykanto-example"
main_df <- here::here("data", "datasets", DATASET_ID, "distances.csv") |>
  read_csv()

umap_df <- here::here("data", "datasets", DATASET_ID, "embedding.csv") |>
  read_csv()


# Plot settings

fig_dir <- here::here("reports", "figures")
main_df$colour <- ifelse(grepl("different bird", main_df$type), "different", "same")
background_colour <- "#363636"
level_order <- c("different bird", "different year", "top hits", "same year")
labels <- c(
  "Different birds", "Same bird, diff. year", "Top hit:\nSame bird, diff. year",
  "Same bird, same year"
)
label_colours <- c("#38808a", "#d1a11e", "#bb4631", "#df882b")
label_colours <- shades::brightness(label_colours, .9)

colour_dict <- list(
  "grey_text" = "#403e44"
)

font_size <- 15


# Main plot (acoustic similarity)
main_df |> ggplot(aes(
  x = dist,
  fill = factor(type, level = level_order)
)) +
  ggdist::stat_slab(
    alpha = .8,
    trim = FALSE,
    adjust = 1.2,
    colour = "#f0f0f0",
    size = .5
    # fill = rgb2[2]
  ) +
  geom_rug(
    sides = "b",
    length = unit(10, "mm"),
    linewidth = .8,
    mapping = aes(
      alpha = factor(type, level = level_order),
      color = factor(type, level = level_order)
    ),
  ) +
  scale_alpha_manual(
    values = c(.1, .1, .1, .5),
    breaks = level_order,
    # do not include legend:
    guide = "none"
  ) +
  scale_fill_manual(
    name = "Grouping",
    values = label_colours,
    breaks = level_order,
    labels = labels
  ) +
  scale_color_manual(
    guide = "none",
    values = label_colours,
    breaks = level_order,
  ) +
  geom_hline(
    yintercept = 0,
    colour = "#f0f0f0",
    size = .5
  ) +
  labs(
    title = "Comparing acoustic similarity",
    y = "Density", x = "Acoustic similarity"
  ) +
  scale_y_continuous(limits = c(-.046, 1)) +
  # add ticks and labels at 0 to 1 in 0.2 steps, 0 and 1 without decimals:
  scale_x_continuous(
    breaks = seq(0, 1, by = .2),
    labels = c("0", "0.2", "0.4", "0.6", "0.8", "1")
  ) +
  guides(fill = guide_legend(byrow = TRUE)) +
  # force square plot:
  coord_fixed(ratio = 1) +
  theme(
    # force square:
    legend.justification = c(0, 1),
    legend.position = c(.05, .95),
    legend.spacing.y = unit(.8, "lines"),
    legend.background = element_blank(),
    legend.text = element_text(color = "white", size = font_size),
    legend.title = element_text(color = "white", size = font_size, face = "bold"),
    aspect.ratio = 1,
    text = element_text(size = font_size, family = "Helvetica"),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.text.y = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 0, r = 10, b = 0, l = 0)
    ),
    panel.grid.major = element_line(color = "#4d4d4d"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(
      size = font_size * 1.3, face = "bold",
      margin = margin(t = 0, r = 0, b = 10, l = 0)
    ),
    axis.title.y = element_text(
      size = font_size * 1.2,
      margin = margin(t = 0, r = 20, b = 0, l = 0)
    ),
    axis.title.x = element_text(
      size = font_size * 1.2,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.line = element_blank(),
    panel.border = element_rect(
      colour = background_colour, fill = NA,
      linewidth = 10
    ),
    panel.background = element_rect(
      colour = background_colour,
      fill = background_colour, linewidth = 10
    )
  ) -> ac_sim_densiplot


ggsave(
  filename = file.path(fig_dir, "ac_sim_densiplot.png"),
  plot = ac_sim_densiplot,
  width = 20,
  height = 20,
  units = "cm",
  dpi = 300
)

# PLOT UMAP EMBEDDING ──────────────────────────────────────────────────────── #

umap_df |> ggplot(
  aes(
    x = x, y = y,
    shape = factor(markers),
    color = factor(labs),
    alpha = factor(markers)
  )
) +
  # marker type by group 'markers':
  geom_point(
    size = 3,
  ) +
  scale_shape_manual(
    name = "Year",
    values = c(3, 19),
    labels = c("2020", "2021")
  ) +
  scale_alpha_manual(
    values = c(.7, .5),
    guide = "none"
  ) +
  # vector with 12 colours:
  scale_colour_manual(
    # add 12 differnt colours:
    values = c(
      "#e2a71e", "#000000", "#000000", "#000000", "#000000",
      "#000000", "#c5453c", "#000000", "#000000", "#000000", "#000000", "#29bec9"
    ),
    guide = "none"
  ) +
  # scale_color_discrete(type = c()) +
  labs(
    title = "2D embedding of all songs",
    y = "UMAP 2", x = "UMAP 1"
  ) +
  scale_y_continuous(expand = expansion(mult = c(.1, .1))) +
  guides(fill = guide_legend(byrow = TRUE), shape = guide_legend(override.aes = list(color = "white"))) +
  coord_fixed(ratio = 1) +
  theme(
    legend.justification = c(0, 1),
    legend.position = c(.05, .95),
    legend.spacing.y = unit(.8, "lines"),
    legend.background = element_rect(fill = "transparent", colour = NA),
    legend.box.background = element_rect(fill = "transparent", colour = NA),
    legend.key = element_rect(fill = "transparent", colour = NA),
    legend.text = element_text(color = "white", size = font_size),
    legend.title = element_text(color = "white", size = font_size, face = "bold"),
    aspect.ratio = 1,
    text = element_text(size = font_size, family = "Helvetica"),
    axis.ticks = element_blank(),
    axis.text.x = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.text.y = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 0, r = 10, b = 0, l = 0)
    ),
    panel.grid.major = element_line(color = "#4d4d4d"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(
      size = font_size * 1.3, face = "bold",
      margin = margin(t = 0, r = 0, b = 10, l = 0)
    ),
    axis.title.y = element_text(
      size = font_size * 1.2,
      margin = margin(t = 0, r = 20, b = 0, l = 0)
    ),
    axis.title.x = element_text(
      size = font_size * 1.2,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.line = element_blank(),
    panel.border = element_rect(
      colour = background_colour, fill = NA,
      linewidth = 10
    ),
    panel.background = element_rect(
      colour = background_colour,
      fill = background_colour, linewidth = 10
    )
  ) -> ac_sim_scatterplot


ggsave(
  filename = file.path(fig_dir, "ac_sim_scatterplot.png"),
  plot = ac_sim_scatterplot,
  width = 20,
  height = 20,
  units = "cm",
  dpi = 300
)


(ac_sim_densiplot + plot_spacer() + ac_sim_scatterplot +
  plot_layout(widths = c(4, .3, 4))) -> ac_sim_both

for (f in c("png", "svg")) {
  ggsave(
    filename = file.path(fig_dir, paste0("ac_sim_both.", f)),
    plot = ac_sim_both,
    width = 40,
    height = 20,
    units = "cm",
    dpi = 300
  )
}

print("Done!")
