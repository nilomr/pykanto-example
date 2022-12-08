box::use(ggplot2, dplyr, readr[read_csv])

# Read in the data
# find repository root
rprojroot::find_rstudio_root_file()
here::here("data", "data.csv") |>
    read_csv() |>
    head()
