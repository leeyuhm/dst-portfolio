# ================ #
# Loading the data #
# ================ #

temperature=read.table("https://berkeley-earth-temperature.s3.us-west-1.amazonaws.com/Global/Land_and_Ocean_complete.txt",skip = 86,nrows = 1997)
colnames(temperature)=c("Year","Month","MA","MACI","AA","AACI","A5","A5CI","A10","A10CI","A20","A20CI")
temperature$Time=temperature$Year+(temperature$Month-1)/12

# Defining the time series
ma_ts <- ts(temperature$MA, start = c(temperature$Year[1], temperature$Month[1]),
         frequency = 12)

# ======================= #
# ARIMA model: Blocked CV #
# ======================= #

# my_ts: given time series
# n_blocks: number of blocks
# h: forecast horizon

blocked_cv <- function(my_ts, n_blocks = 5, h = 295) {
  n <- length(my_ts)
  block_size <- floor(n / n_blocks)
  
  errors <- c()  # empty vector to store errors
  
  # Looping over each block
  for (b in 1:n_blocks) {
    start <- (b - 1) * block_size + 1
    end   <- b * block_size
    block <- my_ts[start:end]
    
    # Skipping if future data is insufficient
    if ((end + h) > n) {
      next
    }
    
    # Fitting proposed ARIMA model on current training block
    fit <- Arima(block, order = c(2,1,1),
                 seasonal = list(order = c(0,0,2), period = 12))
    
    # Forecasting h steps ahead
    fc <- forecast(fit, h = h)$mean
    
    # h-step-ahead errors
    actual_h <- my_ts[end + h]
    errors <- c(errors, actual_h - fc[h])
  }
  
  # Performance metrics for chosen h
  MAE  <- mean(abs(errors))
  RMSE <- sqrt(mean(errors^2))
  SD_Uncertainty <- sd(errors)
  
  return(c(MAE = MAE, RMSE = RMSE, SD_Uncertainty = SD_Uncertainty))
}

# === Comparing varying h === #

library(doParallel)
library(forecast)
library(foreach)

# Setup
cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

# Candidate forecast horizons
hs <- c(1, 12, 60, 120, 295)

# Performing parallel blocked CV for each h, combining results by stacking them
blocked_results_arima <- foreach(h = hs, .combine = rbind,
                                 .packages = c("forecast")) %dopar% {
                                   blocked_cv(my_ts, n_blocks = 5, h = h)
                                 }

stopCluster(cl)

rownames(blocked_results_arima) <- paste0("h=", hs)
colnames(blocked_results_arima) <- c("MAE", "RMSE", "SD_Uncertainty")

blocked_results_arima

# ============================================ #
# Polynomial Regression (degree 5): Blocked CV #
# ============================================ #

poly_blocked_cv <- function(df, n_blocks, h ) {
  n <- nrow(df)
  block_size <- floor(n / n_blocks)
  
  errors <- c()
  
  for (b in 1:n_blocks) {
    start <- (b - 1) * block_size + 1
    end   <- b * block_size
    
    if ((end + h) > n) next
    
    # Training block
    train_data <- df[start:end, ]
    
    # Fitting the polynomial regression used in the workshop
    model <- lm(MA ~ poly(Time, 5), data = train_data)
    
    # Time of future point to predict
    t_future <- df$Time[end + h]
    
    # h-step prediction
    pred <- predict(model, newdata = data.frame(Time = t_future))
    
    # Calculating errors
    actual <- df$MA[end + h]
    errors <- c(errors, actual - pred)
  }
  
  # Metrics
  MAE  <- mean(abs(errors))
  RMSE <- sqrt(mean(errors^2))
  SD_Uncertainty <- sd(errors)
  
  return(c(MAE = MAE, RMSE = RMSE, SD_Uncertainty = SD_Uncertainty))
}

# === Comparing varying h === #

cores <- detectCores() - 1
cl <- makeCluster(cores)
registerDoParallel(cl)

hs <- c(1, 12, 60, 120, 295)

blocked_results_poly <- foreach(h = hs, .combine = rbind,
                                .packages = c("stats")) %dopar% {
                                  poly_blocked_cv(temperature, n_blocks = 10,
                                                  h = h)
                                }

stopCluster(cl)

rownames(blocked_results_poly) <- paste0("h=", hs)
colnames(blocked_results_poly) <- c("MAE", "RMSE", "SD_Uncertainty")

blocked_results_poly
