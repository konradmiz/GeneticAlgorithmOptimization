suppressMessages(suppressWarnings(library(readr)))
suppressMessages(suppressWarnings(library(ggplot2)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(ggrepel)))
suppressMessages(suppressWarnings(library(geosphere))) 
suppressMessages(suppressWarnings(library(patchwork)))
suppressMessages(suppressWarnings(library(glue)))

oldw <- getOption("warn") # change warning settings to not warn
options(warn = -1)
options(readr.num_columns = 0)

setwd("C:/Users/Konrad/Desktop/GeneticAlgorithm/")

credentials <- config::get(file = "C:/Users/Konrad/Documents/credentials_config.yml")

vehicle_info <- read_csv("Data/df.csv")
not_working_vehicles <- read_csv("Data/not_working.csv")
route_idx <- read_csv("Results/route_numbers.csv", col_names = FALSE)
trial_summary <- read_csv("Results/summary.csv", col_names = TRUE)

join_data_fn <- function(file_path) { # join the routes to their proper vehicle ids used by Biketown
  read_csv(paste0("Results/", file_path), col_names = FALSE) %>%
    left_join(vehicle_info, by = c("X1" = "id_no")) %>%
    left_join(not_working_vehicles, by = "id")
}


first_route <- join_data_fn("first_route.csv")
second_route <- join_data_fn("one_third_route.csv")
third_route <- join_data_fn("two_third_route.csv")
fourth_route <- join_data_fn("best_route.csv")

workshop_df <- data_frame(lon = credentials$location_info$lon, 
                          lat = credentials$location_info$lat)


plot_fn <- function(route_data, graph_title, graph_subtitle) {
  ggplot() + 
    geom_point(data = workshop_df, aes(lon, lat), shape = 1) + 
    geom_point(data = not_working_vehicles, aes(lon, lat), col = "red", size = 1) + 
    geom_point(size = 3) + 
    theme_bw() +
    geom_segment(data = route_data, aes(x = lon, xend = dplyr::lead(lon), y = lat, yend = dplyr::lead(lat)), #arrow = arrow(), 
                 color = "#566427", size = 1) + 
    theme(axis.ticks.y = element_blank()) +
    ylab("lat") +
    ggtitle(label = graph_title)
}

title_fn <- function(graph_number) {
  iteration = route_idx$X1[graph_number]
  if (iteration == 0) {
    iteration = 1
  }
  paste0("Generation: ", iteration,
         " Fitness: ", trial_summary$fitness[iteration], 
         " Distance: ", scales::comma(round(trial_summary$distance[iteration]), 2))
  
}

first_title <- title_fn(1)
second_title <- title_fn(2)
third_title <- title_fn(3)
fourth_title <- title_fn(4)

first_g <- plot_fn(first_route, graph_title = first_title)
second_g <- plot_fn(second_route, graph_title = second_title)
third_g <- plot_fn(third_route, graph_title = third_title)
fourth_g <- plot_fn(fourth_route, graph_title = fourth_title)

fourth_g + ggsave("Images/FinalRoute.png", height = 10, width = 12)

first_g + second_g + third_g + fourth_g + ggsave("Images/Routes.png", height = 10, width = 12)


modified_trial_summary <- trial_summary %>%
  mutate(time = row_number() - 1,
         distance = as.numeric(distance),
         fitness = as.numeric(fitness)) 

distinct_fitness_times <- modified_trial_summary %>%
  distinct(fitness, .keep_all = TRUE)

ggplot(modified_trial_summary, aes(time, distance)) +
  theme_bw() +
  geom_line() +
  geom_point(data = distinct_fitness_times, aes(x = time, y = distance)) + 
  geom_text_repel(data = distinct_fitness_times, aes(label = fitness, x = time, y = distance), 
                  force = 0.5, min.segment.length = 0.25) + 
  labs(x = "generation", y = "distance") + 
  scale_y_continuous(labels = scales::comma) +
  ggsave("Images/FitnessOverTime.png", height = 8, width = 12)

options(warn = oldw)
