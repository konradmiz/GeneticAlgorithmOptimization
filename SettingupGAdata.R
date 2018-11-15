library(DBI)
library(reshape2)
library(sf)
library(dplyr)
library(geosphere)
library(readr)
library(docopt)

'Usage:
   settingupGAdata.R [-v <variable>]

Options:
   -v Whether to use bikes or scooters [default: bikes]

 ]' -> doc

opts <- docopt(doc)

credentials <- config::get(file = "C:/Users/Konrad/Documents/credentials_config.yml")

con <- dbConnect(drv = RPostgreSQL::PostgreSQL(),
                 user = credentials$database_info$user,
                 password = credentials$database_info$pwd,
                 host = credentials$database_info$host,
                 port = credentials$database_info$port,
                 dbname = credentials$database_info$dbname)



if (opts$v == "scooters") {

  vehicle_data <- dbGetQuery(con, "SELECT * FROM scooters WHERE time = (SELECT MAX(TIME) FROM scooters);") 
  dbDisconnect(con)
  vehicle_data <- vehicle_data %>%
    filter(battery_level <= 30) %>%
    mutate(id = as.numeric(id),
           id = 1:nrow(.))
} else {
  vehicle_data <- dbGetQuery(con, "SELECT * FROM current_bikes") %>%
    filter(repair_state != "working")
}

workshop <- c(credentials$location_info$lon, credentials$location_info$lat)

not_working_vehicles <- all_vehicles %>%
  mutate(dist = distHaversine(cbind(lon, lat), workshop)) %>%
  filter(dist >= 100) %>%
  select(id, lat, lon) %>%
  bind_rows(data_frame(id = 0, lat = credentials$location_info$lat, lon = credentials$location_info$lon)) 

not_working_vehicles %>%
  write_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/not_working.csv")

vehicle_info <- not_working_vehicles %>%
  arrange(id) %>%
  mutate(id_no = 1:nrow(.) - 1) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326) %>%
  st_transform(26910) 

vehicle_coords <- vehicle_info %>%
  st_coordinates()

vehicle_info %>%
  write_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/df.csv")
  
dist_m <- melt(as.matrix(dist(bike_coords, diag = TRUE, upper = TRUE, method = "manhattan")), 
               varnames = c("from", "to")) %>%
  rename(dist = value) %>%
  mutate(from = from - 1,
         to = to - 1) %>%
  tidyr::spread(key = "from", value = "dist") %>%
  select(-to)
rownames(dist_m) <- vehicle_info$id_no
colnames(dist_m) <- vehicle_info$id_no

write_csv(dist_m, "C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/dist.csv")

bike_dist <- sapply(2:nrow(vehicle_coords), function(x)  {
  abs(vehicle_coords[1, 1] - vehicle_coords[x, 1]) + abs(vehicle_coords[1, 2] - vehicle_coords[x, 2])}) %>%
  data_frame() %>% 
  write_csv("C:/Users/Konrad/Desktop/GeneticAlgorithm/Data/workshopdist.csv")

