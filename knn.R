########################################################
#                                                      #
#       Finding Best k for kNN regression by LOOCV     #
#                                                      #
########################################################

# knnopt -------------------------------------------------------------------------
#' Best k for the kNN regression
#'
#' \code{knnopt} returns an object of class knnopt with three attributes
#' 1) kopt: the optimal k chosen by LOOCV
#' 2) predict: a vector of the same length as y of the predicted y's using knn with x's
#' 3) MSE: the estimate mean squared error
#'
#' @param x: a matrix or dataframe.
#' @param y: a vector.
#' @param k_max: a positive constant less than the number of the sample size. Warning: a user
#' defined value for k_max may end up in a non-optimal result.
#'
#' @return an object of class knnopt.
#' @author
#' Azadkia, M. \email{mazadkia@stanford.edu}
#' @references \href{https://arxiv.org/abs/1909.05495}{Azadkia, M. Optimal choice of k for k-nearest neighbor regression.}
#' @examples
#' n <- 100
#' x <- rnorm(n)
#' y <- x + x^2 + rnorm(n)
#' a <- knnopt(x, y)
#' a$kopt
#' a$MSE
#' a$predict
#' # the theoretical result is for when k_max = n - 1
#' b <- knnopt(x, y, k_max = 20)
#' b$kopt
#' b$MSE
#' b$predict
knnopt <- function(x, y, binary, k_max = NULL) {

  if(!is.matrix(y)) {
    y = as.matrix(y)
  }
  if(!is.matrix(x)) {
    x = as.matrix(x)
  }
  # if(ncol(y) != 1) stop("y should be a one dimensional.")
  if(nrow(x) != length(y)) stop("number of rows of y and x should be equal.")

  n = nrow(x)

  if(n < 3) stop("number samples should be at least 3.")

  if(is.null(k_max)) {
    k_max = n - 1
  }

  if (k_max >= n) stop("k_max should be smaller than sample size.")
  if (k_max < 1 || floor(k_max) != k_max) stop("k_max should be a postive constant.")

  # the extra nearest neighbor is computed is take care of ties
  nn_X = nn2(x, query = x, k = k_max + 1)
  nn_index_X = nn_X$nn.idx
  # handling repeated data
  repeat_data = which(nn_X$nn.dists[, 2] == 0)
  for(i in repeat_data) {
    if(nn_index_X[i, 1] != i) {
      temp = nn_index_X[i, 1]
      nn_index_X[i, 1] = i
      nn_index_X[i, which(nn_index_X[i, ] == i)] = temp
    }
  }
  nn_index_X = nn_index_X[, -1]
  nn_dist = nn_X$nn.dists[, -1]

  f = rep(0, k_max)
  yhat = rep(0, n)
  min_mse = Inf
  for (k in 1:k_max) {
    # helper function
    compute_CV_MSE <- function (i) {

      if ((nn_X$nn.dists[i, k] < nn_X$nn.dists[i, k + 1])) {
        return (mean(y[nn_index_X[i, 1:k]]))
      }

      if(k == 1) {
        eq_dist = which (nn_dist[i, ] == nn_dist[i, k])
        return ( sample (y[nn_index_X[i, eq_dist]], 1, replace = FALSE))
      }
      # ties and k > 1
      less_dist = which (nn_dist[i, 1:(k - 1)] < nn_dist[i, k])
      n_less = length (less_dist)
      eq_dist = which (nn_dist[i, ] == nn_dist[i, k])
      n_more = k - n_less
      if (!binary) {
        return ( mean (c(y[nn_index_X[i, 1:(k - 1)]],
                         sample (y[nn_index_X[i, eq_dist]],
                                 n_more,
                                 replace = FALSE))))
      } else {
        return ( min(1, floor(2 * mean (c(y[nn_index_X[i, 1:(k - 1)]],
                         sample (y[nn_index_X[i, eq_dist]],
                                 n_more,
                                 replace = FALSE))))))
      }
    }

    y_hat = sapply (seq(1, n), compute_CV_MSE)
    f[k] = mean ((y - y_hat) ^ 2)
    if (f[k] < min_mse) {
      min_mse = f[k]
      yhat = y_hat
    }
  }

  kopt = min(which(f == min(f)))
  result <- list(kopt = kopt, predict = yhat, MSE = min_mse)
  class(result) = "knnopt"

  return(result)
}

# predict.knnopt -------------------------------------------------------------------------
#' kNN regression with best k
#'
#' \code{predict.knnopt} returns an object of class knnopt with three attribute \\
#' 1) kopt : the optimal k chosen by LOOCV
#' 2) predict : a vector of the same length as number of rows of xnew
#' of the predicted y's using knn with x's
#' 3) MSE : the estimate mean squared error
#'
#' @param x: a matrix or dataframe (training set).
#' @param y: a vector (training set).
#' @param xnew: a matrix or dataframe (test set)
#' @param k_max: a positive constant less than the number of the sample size. Warning: a user
#' defined value for k_max may end up in a non-optimal result.
#'
#' @return an object of class knnopt.
#' @author
#' Azadkia, M. \email{mazadkia@stanford.edu}
#' @references \href{https://arxiv.org/abs/1909.05495}{Azadkia, M. Optimal choice of k for k-nearest neighbor regression.}
#' @examples
#' n <- 100
#' x <- rnorm(n)
#' y <- x + x^2 + rnorm(n)
#' xnew <- rnorm(10)
#' a <- predict.knnopt(x, y, xnew)
#' a$kopt
#' a$MSE
#' a$predict
predict.knnopt <- function (x, y, xnew, binary, k_max = NULL) {

  if(!is.matrix(y)) {
    y = as.matrix(y)
  }
  if(!is.matrix(x)) {
    x = as.matrix(x)
  }

  if(ncol(y) != 1) stop("y should be a one dimensional.")
  if(nrow(x) != nrow(y)) stop("number of rows of y and x should be equal.")

  if (!is.null(k_max)) {
    warning("with upper bound k_max on the k, the result may not give the optimal number of neighbors.")
    result <- knnopt(x, y, binary, k_max)
  } else {
    result <- knnopt(x, y, binary)
  }
  # no new data has been provided
  if (is.null(xnew)) {
    return(result)
  }

  kopt = result$kopt
  n = nrow(x)

  if(!is.matrix(xnew)) {
    xnew = as.matrix(xnew)
  }

  # the extra nearest neighbor is computed to take care of the ties
  nn_X = nn2(x, query = xnew, k = kopt + 1)

  ties_id = which(nn_X$nn.dists[, kopt] == nn_X$nn.dists[, kopt + 1])
  if (length(ties_id) > 0) {
    nn_X_ties = nn2(x, query = matrix(xnew[ties_id, ], nrow = length(ties_id)), k = n)
    for (i in 1:length(ties_id)) {
      ids = which(nn_X_ties$nn.dists[i, ] == nn_X_ties$nn.dists[i, kopt])
      nn_X$nn.idx[ties_id[i], ids[1:((kopt - ids[1]) + 1)]] = sample(ids, kopt - ids[1] + 1, replace = FALSE)
    }
  }

  yhat = apply(matrix(y[as.vector(nn_X$nn.idx[, 1:kopt])], nrow = nrow(xnew)), 1, mean)
  if(binary) {
    yhat = sapply(floor(2 * yhat), min1 <- function(x){min(x, 1)})
  }
  result$predict <- yhat
  return(result)
}
