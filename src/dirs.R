# DESCRIPTION ──────────────────────────────────────────────────────────────── #

# Functions and variable definitions related to project structure and directory
# management.

# PATH VARIABLES ───────────────────────────────────────────────────────────── #

dirs <- list()
dirs$root <- rprojroot::find_rstudio_root_file()
dirs$resources <- file.path(dirs$root, "resources")
dirs$figs <- file.path(dirs$root, "reports", "figures")
dirs$data <- file.path(dirs$root, "data")
dirs$model_fits <- file.path(dirs$data, "models", "fits")
for (dir in dirs) {
    if (!dir.exists(dir)) {
        dir.create(dir, recursive = TRUE)
    }
}

# FUNCTIONS ────────────────────────────────────────────────────────────────── #
