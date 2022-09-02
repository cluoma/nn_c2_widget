# to_json.R
#
# colin luoma
#
# turns output files from 'nn_c2.c' into a single JSON file

# libraries ----
library(tidyverse)
library(readr)
library(jsonlite)

# read data ----
biases0  = read_csv("biases0.csv", col_names = FALSE)
biases1  = read_csv("biases1.csv", col_names = FALSE)
biases2  = read_csv("biases2.csv", col_names = FALSE)
weights0 = read_csv("weights0.csv", col_names = FALSE)
weights1 = read_csv("weights1.csv", col_names = FALSE)
weights2 = read_csv("weights2.csv", col_names = FALSE)

# combine data ----
weights_as_json = toJSON(x = list(
  biases0  = biases0$X1,
  biases1  = biases1$X1,
  biases2  = biases2$X1,
  weights0 = weights0$X1,
  weights1 = weights1$X1,
  weights2 = weights2$X1
))

# save JSON file ----
write_file(weights_as_json, "weights_as_json.json")
