library(readr)
library(ggplot2)
library(dplyr)
library(ggrepel)
library(geosphere)
library(patchwork)
library(glue)

setwd("C:/Users/Konrad/Desktop/GeneticAlgorithm/")

vehicle_info <- read_csv("Data/df.csv")
not_working_vehicles <- read_csv("Data/not_working.csv")

trial_summary <- read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/Results/summary.csv", col_names = TRUE)

join_data_fn <- function(file_path) {
  read_csv(paste0("Results/", file_path), col_names = FALSE) %>%
    left_join(vehicle_info, by = c("X1" = "id_no")) %>%
    left_join(not_working_vehicles, by = "id")
}

route_idx <- read_csv("Results/route_numbers.csv", col_names = FALSE)

first_route <- join_data_fn("first_route.csv")
second_route <- join_data_fn("one_third_route.csv")
third_route <- join_data_fn("two_third_route.csv")
fourth_route <- join_data_fn("best_route.csv")

workshop_df <- data_frame(lat = 45.512669, lon = -122.659312)

plot_fn <- function(route_data, graph_title, graph_subtitle) {
  ggplot() + 
    geom_point(data = workshop_df, aes(lon, lat), shape = 1) + 
    geom_point(data = not_working_vehicles, aes(lon, lat), col = "red", size = 1) + 
    #geom_text_repel(data = not_working_bikes, aes(lon, lat, label = id)) +
    geom_point(size = 3) + 
    theme_bw() +
    geom_segment(data = route_data, aes(x = lon, xend = dplyr::lead(lon), y = lat, yend = dplyr::lead(lat)), #arrow = arrow(), 
                 color = "#566427", size = 1) + 
    theme(axis.text.y = element_blank(),
          axis.ticks.y = element_blank()) +
    ylab("") +
    ggtitle(label = graph_title,
            subtitle = graph_subtitle)
    
}

subtitle_fn <- function(graph_number) {
  iteration = route_idx$X1[graph_number]
  if (iteration == 0) {
    iteration = 1
  }
  paste0("Iteration: ", iteration,
         " Fitness: ", trial_summary$fitness[iteration], 
         " Distance: ", scales::comma(round(trial_summary$distance[iteration]), 2))
  
}

first_subtitle <- subtitle_fn(1)
second_subtitle <- subtitle_fn(2)
third_subtitle <- subtitle_fn(3)
fourth_subtitle <- subtitle_fn(4)

first_g <- plot_fn(first_route, graph_title = "Best of First Generation", first_subtitle)
second_g <- plot_fn(second_route, graph_title = "Best 1/3 Through", second_subtitle)
third_g <- plot_fn(third_route, graph_title = "Best 2/3 Through", third_subtitle)
fourth_g <- plot_fn(fourth_route, graph_title = "Best Overall", fourth_subtitle)

first_g + second_g + third_g + fourth_g + ggsave("C:/Users/Konrad/Desktop/GeneticAlgorithm/Routes.png", height = 10, width = 12)

trial_summary %>%
  mutate(time = row_number()) %>%
  #rename(fitness = X3, 
  #       distance = X2) %>%
  mutate(distance = as.numeric(distance),
         fitness = as.numeric(fitness)) %>%
  ggplot(aes(time, distance)) +
  geom_line() + 
  geom_text(aes(label = fitness, x = time, y = distance)) + 
  #scale_x_log10() + 
  theme_bw() + 
  labs(x = "iteration", y = "distance") + 
  scale_y_continuous(labels = scales::comma) #+
  ggsave("C:/Users/Konrad/Desktop/GeneticAlgorithm/FitnessOverTime.png", height = 8, width = 12)
