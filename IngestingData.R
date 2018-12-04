suppressMessages(suppressWarnings(library(DBI)))
suppressMessages(suppressWarnings(library(reshape2)))
suppressMessages(suppressWarnings(library(sf)))
suppressMessages(suppressWarnings(library(dplyr)))
suppressMessages(suppressWarnings(library(geosphere)))
suppressMessages(suppressWarnings(library(readr)))
suppressMessages(suppressWarnings(library(docopt)))

oldw <- getOption("warn") # change warnings in code to not be displayed
options(warn = -1)

# Docopt code
'Usage:
   settingupGAdata.R [-v <variable>]

Options:
   -v Whether to use bikes or scooters [default: bikes]

 ]' -> doc

opts <- docopt(doc)

setwd("C:/Users/Konrad/Desktop/GeneticAlgorithm")

credentials <- config::get(file = "C:/Users/Konrad/Documents/credentials_config.yml")

con <- dbConnect(drv = RPostgreSQL::PostgreSQL(),
                 user = credentials$database_info$user,
                 password = credentials$database_info$pwd,
                 host = credentials$database_info$host,
                 port = credentials$database_info$port,
                 dbname = credentials$database_info$dbname)

if (opts$v == "scooters") {

  vehicle_data <- dbGetQuery(con, "SELECT * FROM scooters WHERE time = (SELECT MAX(TIME) FROM scooters);") %>%
    filter(battery_level <= 30) %>%
    mutate(id = 1:nrow(.))
  
  dbDisconnect(con)
  
} else {
  vehicle_data <- dbGetQuery(con, "SELECT * FROM bikes WHERE time = (SELECT MAX(TIME) FROM bikes);") %>%
    filter(repair_state != "working")
  
  dbDisconnect(con)
}

workshop <- c(credentials$location_info$lon, credentials$location_info$lat)

not_working_vehicles <- vehicle_data %>%
  mutate(dist = distHaversine(cbind(lon, lat), workshop)) %>%
  filter(dist >= 100) %>%
  select(id, lat, lon) %>%
  bind_rows(data_frame(id = 0, lat = credentials$location_info$lat, lon = credentials$location_info$lon)) 

not_working_vehicles %>%
  write_csv("Data/not_working.csv")

vehicle_info <- not_working_vehicles %>%
  arrange(id) %>%
  mutate(id_no = 1:nrow(.) - 1) %>%
  st_as_sf(coords = c("lon", "lat"), crs = 4326) %>% #define spatial object, with lat/lon proj
  st_transform(26910) #reproject to UTM

vehicle_coords <- vehicle_info %>%
  st_coordinates() # get the coordinates of the vehicles

vehicle_info %>%
  write_csv("Data/df.csv")
  
#create Manhattan distance matrix, cast into dataframe
dist_m <- melt(as.matrix(dist(vehicle_coords, diag = TRUE, upper = TRUE, method = "manhattan")), 
               varnames = c("from", "to")) %>%
  rename(dist = value) %>%
  mutate(from = from - 1,
         to = to - 1) %>%
  tidyr::spread(key = "from", value = "dist") %>%
  select(-to)
rownames(dist_m) <- vehicle_info$id_no
colnames(dist_m) <- vehicle_info$id_no

write_csv(dist_m, "Data/dist.csv")

#Get the distances of each bike to the warehouse
bike_dist <- sapply(2:nrow(vehicle_coords), function(x)  {
  abs(vehicle_coords[1, 1] - vehicle_coords[x, 1]) + abs(vehicle_coords[1, 2] - vehicle_coords[x, 2])}) %>%
  data_frame() %>% 
  write_csv("Data/workshopdist.csv")

options(warn = oldw)
