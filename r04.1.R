######################
# LOOCV using knnopt #
######################

library(RANN)

# Loading the data
source("https://raw.githubusercontent.com/dsbristol/dst/master/code/loadconndata.R")

testdata=conndata[,c("orig_bytes","resp_bytes",
                     "orig_ip_bytes","resp_ip_bytes")]
testdata[testdata=="-"]=0
testdata[testdata=="0"]=0
for(i in 1:4) testdata[,i]=log10(as.numeric(testdata[,i])+1)
rownames(testdata)=NULL

# Restricting the data to successful flows
nz <- which(testdata$orig_bytes > 0 & testdata$resp_bytes > 0)

x <- as.matrix(testdata[nz, "orig_bytes"])
y <- testdata[nz, "resp_bytes"]

xy <- cbind(x, y)
md <- mahalanobis(xy, colMeans(xy), cov(xy)) # Mahalanobis distance

sig_lvl <- qchisq(0.975, df = 2) # 95% confidence level ellipse
outliers <- which(md > sig_lvl) # outliers lie outside the ellipse

keep <- setdiff(1:nrow(xy), outliers) # removal of outliers

x <- as.matrix(x[keep])
y <- y[keep]

# Creating a grid for plotting
xnew <- as.matrix(seq(min(x), max(x), length.out = 200), ncol = 1)

set.seed(1) # setting seed due to random tie-breaking

pred <- predict.knnopt(x, y, xnew, binary = FALSE, k_max = 100)
k_opt <- pred$kopt
k_preds <- as.numeric(pred$predict)
k_mse <- pred$MSE

# === Bootstrap analysis of optimal k === #

library(doParallel)
library(foreach)

B <- 100 # 100 resamples
cl <- makeCluster(detectCores() - 1)
registerDoParallel(cl)

k_boot <- foreach(b = 1:B, .combine = c, .packages = "RANN") %dopar% {
  
  idx <- sample(seq_along(y), replace = TRUE)
  
  x_boot <- x[idx, , drop = FALSE]
  y_boot <- y[idx]
  
  knnopt(x_boot, y_boot, binary = FALSE, k_max = 20)$kopt
}

stopCluster(cl)

# === Plotting === #

plot(x, y,
     pch = 19, cex = 0.3,
     xlab = "orig_bytes", ylab = "resp_bytes",
     main = "kNN regression with k = 8")

lines(xnew, preds, col = "blue", lwd = 3)