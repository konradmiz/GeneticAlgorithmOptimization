library(readr)
library(ggplot2)
library(dplyr)
library(ggrepel)
library(geosphere)

num_bikes = 10
bikes_df = data_frame(x = 0, y = 0) %>%
  bind_rows(data_frame(x = runif(num_bikes),
           y = runif(num_bikes)))

dist_m <- melt(as.matrix(dist(bikes_df, diag = TRUE, upper = TRUE, method = "manhattan")), 
               varnames = c("from", "to")) %>%
  rename(dist = value) %>%
  mutate(from = from-1, to = to-1) %>%
  write_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/dist.csv")

bikes_df %>%
  write_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/df.csv")

bikes_df <- bikes_df %>%
  mutate(idx = 0:(nrow(.)-1))

my_route <- read_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/route.csv", col_names = FALSE) %>%
  left_join(bikes_df, by = c("X1" = "idx"))

ggplot(my_route, aes(x, y)) + 
  theme_bw() + 
  geom_segment(aes(xend = dplyr::lead(x), yend = dplyr::lead(y)),
               arrow = arrow() ) + 
  geom_point(data = bikes_df, aes(x, y), col = "red", size = 3) + 
  geom_text_repel(data = bikes_df, aes(x, y, label = idx)) + 
  geom_point(size = 3)

my_route
bike_route
