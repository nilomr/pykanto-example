box::use(ggplot2[...], dplyr, readr[read_csv], ggdist[stat_dots], here, beeswarm)

# DATA INGEST ──────────────────────────────────────────────────────────────── #

# find repository root
DATASET_ID <- "pykanto-example"
main_df <- here::here("data", "datasets", DATASET_ID, "distances.csv") |>
  read_csv()

umap_df <- here::here("data", "datasets", DATASET_ID, "embedding.csv") |>
  read_csv()


# Plot settings
main_df$colour <- ifelse(grepl("different bird", main_df$type), "different", "same")
background_colour <- "#eeeeee"
level_order <- c("different bird", "different year", "same year", "top hits")
labels <- c(
  "Different birds", "Same bird, diff. year",
  "Same bird, same year", "Same bird, diff. year\n(top result)"
)
label_colours <- c("#38808a", "#d1a11e", "#df882b", "#bd503d")
font_size <- 15


# Main plot (acoustic similarity)
main_df |> ggplot(aes(
  x = dist,
  fill = factor(type, level = level_order)
)) +
  geom_density(alpha = .7, colour = NA, adjust = 1.3) +
  geom_rug(
    sides = "b",
    length = unit(10, "mm"),
    size = .8,
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
  ggtitle("Comparing acoustic similarity") +
  labs(y = "Density", x = "Acoustic similarity") +
  scale_y_continuous(expand = expansion(mult = c(.1, .1))) +
  guides(fill = guide_legend(byrow = TRUE)) +
  theme(
    # add space between legend items:
    legend.spacing.y = unit(.8, "lines"),
    aspect.ratio = 1,
    text = element_text(size = font_size, family = "Helvetica"),
    axis.text.x = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.text.y = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 0, r = 10, b = 0, l = 0)
    ),
    panel.grid.major = element_blank(),
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
  )


# JITTER PLOT VERSION ──────────────────────────────────────────────────────── #

level_order <- c("different bird", "top hits", "same year", "different year")
colour_dict <- list(
  "grey_text" = "#403e44"
)

main_df |> ggplot(aes(y = dist, x = factor(type, level = level_order))) +
  geom_jitter(aes(colour = colour, alpha = type), width = .2, height = 0) +
  scale_alpha_manual(
    values = c(.3, .8, .3, .3),
    breaks = level_order,
    # do not include legend:
    guide = "none"
  ) +
  scale_colour_manual(
    name = "Which bird?",
    values = c("#c9992d", "#24808d"),
    breaks = c("different", "same")
  ) +
  scale_x_discrete(
    labels = c("different bird", "top hits\n(different year)", "same year", "different year")
  ) +
  ggdist::stat_pointinterval(
    position = "dodge",
    scale = .5,
    point_size = 5,
    point_interval = ggdist::median_qi,
    .width = c(.66, .95),
    interval_size_range = c(.9, 2.5),
    interval_colour = "black",
    point_colour = "black",
    fatten_point = 3
  ) +
  ggtitle("Comparing acoustic similarity") +
  labs(x = NULL, y = "Acoustic similarity") +
  theme(
    # remove panel background and leave outline:
    panel.background = element_blank(),
    panel.border = element_rect(fill = NA, color = "black"),
    aspect.ratio = 1,
    text = element_text(size = 15, family = "Helvetica"),
    axis.text.x = element_text(
      color = colour_dict$grey_text, size = 15,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.text.y = element_text(
      color = colour_dict$grey_text, size = 10,
      margin = margin(t = 0, r = 10, b = 0, l = 0)
    ),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.title.y = element_text(
      size = 15,
      margin = margin(t = 0, r = 20, b = 0, l = 0)
    )
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
    size = 5,
  ) +
  scale_shape_manual(
    name = "Year",
    values = c(3, 19),
    labels = c("2020", "2021")
  ) +
  scale_alpha_manual(
    values = c(.7, .3),
    guide = "none"
  ) +
  scale_color_discrete() +
  ggtitle("Comparing acoustic similarity") +
  labs(y = "Density", x = "Acoustic similarity") +
  scale_y_continuous(expand = expansion(mult = c(.1, .1))) +
  guides(fill = guide_legend(byrow = TRUE)) +
  theme(
    # add space between legend items:
    legend.spacing.y = unit(.8, "lines"),
    aspect.ratio = 1,
    text = element_text(size = font_size, family = "Helvetica"),
    axis.text.x = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 20, r = 0, b = 0, l = 0)
    ),
    axis.text.y = element_text(
      color = colour_dict$grey_text, size = font_size * .8,
      margin = margin(t = 0, r = 10, b = 0, l = 0)
    ),
    panel.grid.major = element_blank(),
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
  )
