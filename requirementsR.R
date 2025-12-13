pkgs= c("doParallel", "forecast", "foreach" "RANN")

for(p in pkgs){
  if (!require(p,character.only = TRUE)) install.packages(p)
}