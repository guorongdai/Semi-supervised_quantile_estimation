# This is the functions of the functions for semi-supervised quantile estimation.

.libPaths("/home/grad/rondai/RLibrary")

suppressMessages({
  if (!require(np)) install.packages("/home/grad/rondai/ssl/np_0.60-10.tar.gz", 
                                     repos = NULL, type = .Platform$pkgType)
  
  if (!require(glmnet)) install.packages('glmnet', repos =
                                           'https://cran.revolutionanalytics.com/')
  
  if (!require(LassoSIR)) install.packages('LassoSIR', repos =
                                             'https://cran.revolutionanalytics.com/')
  
  if (!require(randomForest)) install.packages('randomForest', repos =
                                                 'https://cran.revolutionanalytics.com/')
  
  if (!require(expm)) install.packages('expm', 
                                       repos='https://cran.revolutionanalytics.com/')
  
  library(np)
  library(glmnet)
  library(LassoSIR)
  library(randomForest)
  library(expm)
  
})


#################################################################################
# supervised and semi-supervised quantile estimation with data splitting
# x1 is the covariate matrix of the labeled set.
# y1 is the response vector of the unlabeled set.
# x2 is the covariate matrix of the unlabeled set.
# tau is the quantile level.
# method = "ks" or "epi" or "rf" means imputations based on kernel smoothing, entirely  
# parametric method or random forest.
# drmethod is the method for dimension reduction.
# drmethod = "linear" means using linear regression.
# drmethod = "logistic" means using logstic regression.
# drmethod = "save" means using SAVE on I(Y < theta).
# drmethod = "sir" means using SIR.
# drmethod = "save.con" means using SAVE on Y.
# drmethod = "plinear" means using penalized linear regression.
# drmethod = "plogistic" means using penalized logistic regression.
# drmethod = "psir" means using the penalized sliced inverse regression.
# r is the number of directions in dimension reduction.
# ss is the number of slices in sir.
# dr = T means applying dimension reduction via the semi-supervised SIR.
# epimethod = "logistic", linear", "plogistic" or "plinear" means logistic, linear, 
# penalized logistic or penalized linear model for the entirely parametrically imputation.
# epimethod = "known" means using the known function logit(sum(x) + sum(x ^ 2)).
# k is the number of folds for data splitting.
# There are four outputs: 
# $ ssest is the ss estimator; 
# $ supest is the supervised estimator.
# $ sssd is the ss standard deviation
# $ supsd is the supervised standard deviation.
qss = function(x1, y1, x2, tau, method = "ks", drmethod = NULL, 
               r = NULL, ss = NULL, dr = F, epimethod = "linear", k)
{
  
  sup = quantile(y1, tau) # supervised estimator
  
  sink("aux")
  ftheta = npudens(tdat = y1, edat = sup, nmulti = 1) $ dens # density estimation
  sink(NULL)
  # ftheta = density(y1, n = 1, from = sup) $ y # density estimation
  
  yind = as.numeric(y1 < sup) 
  
  if(method == "ks") res = ksedrds(sup, ftheta, tau, x1, y1, yind, x2, 
                                   drmethod, r, ss, dr, k)
  
  if(method == "epi") res = epi(sup, ftheta, tau, x1, y1, yind, x2, epimethod, k)
  
  if(method == "rf") res = rf(sup, ftheta, tau, x1, yind, x2, k)
  
  return(res)
  
}
#################################################################################



#################################################################################
# imputation based on random forest
# sup is the initial estimator.
# ftheta is the estimator.
# tau is the quantile level.
# x is the covariate matrix of the labeled set.
# yind is the indicator I(y1 < theta_initial).
# xnew is the covariate matrix of the unlabeled set.
# k is the number of folds for data splitting
# There are four outputs: 
# $ ssest is the ss estimator; 
# $ supest is the supervised estimator.
# $ sssd is the ss standard deviation
# $ supsd is the supervised standard deviation.
rf = function(sup, ftheta, tau, x, yind, xnew, k)
{
  
  n1 = nrow(x)
  n2 = nrow(xnew)
  n = n1 + n2
  nk = floor(n1 / k)
  
  sum1 = 0 # sum of fitted values of the labeled set
  sum2 = 0 # sum of fitted values of the unlabeled set
  phihat = numeric(n1) # estimator of F(theta | g(X))
  
  for(i in 1 : (k - 1))
  {
    ind = (nk * (i - 1) + 1) : (nk * i)
    ind1 = 1 : length(ind)
    
    xx1 = x[-ind, ]
    yyind = as.factor(yind[-ind])
    xx2 = rbind(x[ind, ], xnew)
    
    rfm = randomForest(x = xx1, y = yyind)
    fit = predict(rfm, xx2, type = "prob")[, 2]
    
    phihat[ind] = fit[ind1]
    
    sum1 = sum1 + sum(fit[ind1])
    sum2 = sum2 + sum(fit[-ind1])
    
  }
  
  
  ind = (nk * (k - 1) + 1) : n1
  ind1 = 1 : length(ind)
  
  xx1 = x[-ind, ]
  yyind = as.factor(yind[-ind])
  xx2 = rbind(x[ind, ], xnew)
  
  rfm = randomForest(x = xx1, y = yyind)
  fit = predict(rfm, xx2, type = "prob")[, 2]
  
  phihat[ind] = fit[ind1]
  
  sum1 = sum1 + sum(fit[ind1])
  sum2 = sum2 + sum(fit[-ind1])
  
  
  #########################################
  
  imp = sum1 * n2 / (n1 * n) - sum2 / (n * k)
  
  ss = sup + ( imp - (mean(yind) - tau) ) / ftheta
  #########################################
  # variance of the ss estimator
  
  sigma = sqrt(n2 / n * var(yind - phihat) + n1 / n * mean((yind - tau) ^ 2))
  
  ss.sd = sigma / (ftheta * sqrt(n1)) # ss sd
  
  sup.sd = sqrt(tau - tau ^ 2) / (ftheta * sqrt(n1)) # supervised sd
  
  
  #########################################
  
  res = list("ssest" = ss, "supest" = sup, "sssd" = ss.sd, "supsd" = sup.sd)
  
  return(res)
  
}
#################################################################################



#################################################################################
# entirely parametric imputation 
# sup is the initial estimator.
# ftheta is the estimator.
# tau is the quantile level.
# x is the covariate matrix of the labeled set.
# y is the response vector of the labeled set.
# yind is the indicator I(y1 < theta_initial).
# xnew is the covariate matrix of the unlabeled set.
# epimethod = "logistic", linear", "plogistic" or "plinear" means logistic, linear, 
# penalized logistic or penalized linear model for the entirely parametrically imputation.
# epimethod = "known" means using the known function logit(mean(x) + mean(x ^ 2)).
# k is the number of folds for data splitting
# There are four outputs: 
# $ ssest is the ss estimator; 
# $ supest is the supervised estimator.
# $ sssd is the ss standard deviation
# $ supsd is the supervised standard deviation.
epi = function(sup, ftheta, tau, x, y, yind, xnew, epimethod = "linear", k)
{
  
  #########################################
  # estimate imputations with data splitting
  n1 = nrow(x)
  n2 = nrow(xnew)
  n = n1 + n2
  nk = floor(n1 / k)
  
  if (epimethod != "known")
  {
    sum1 = 0 # sum of fitted values of the labeled set
    sum2 = 0 # sum of fitted values of the unlabeled set
    phihat = numeric(n1) # estimator of F(theta | g(X))
    
    for(i in 1 : (k - 1))
    {
      ind = (nk * (i - 1) + 1) : (nk * i)
      ind1 = 1 : length(ind)
      
      xx1 = x[-ind, ]
      yy1 = y[-ind]
      yyind = yind[-ind]
      xx2 = rbind(x[ind, ], xnew)
      
      fit = epifit(xx1, yy1, yyind, xx2, epimethod)
      
      phihat[ind] = fit[ind1]
      
      sum1 = sum1 + sum(fit[ind1])
      sum2 = sum2 + sum(fit[-ind1])
      
    }
    
    
    ind = (nk * (k - 1) + 1) : n1
    ind1 = 1 : length(ind)
    
    xx1 = x[-ind, ]
    yy1 = y[-ind]
    yyind = yind[-ind]
    xx2 = rbind(x[ind, ], xnew)
    
    fit = epifit(xx1, yy1, yyind, xx2, epimethod)
    
    phihat[ind] = fit[ind1]
    
    sum1 = sum1 + sum(fit[ind1])
    sum2 = sum2 + sum(fit[-ind1])
    
    
    #########################################
    
    imp = sum1 * n2 / (n1 * n) - sum2 / (n * k)
    
  }
  
  if (epimethod == "known")
  {
    
    fit1 = hf(apply(x, 1, mean) + apply(x ^ 2, 1, mean))
    fit2 = hf(apply(xnew, 1, mean) + apply(xnew ^ 2, 1, mean))
    
    fit = c(fit1, fit2)
    
    phihat = fit1
    
    imp = sum(fit1) * n2 / (n1 * n) - sum(fit2) / n
    
  }
  
  ss = sup + ( imp - (mean(yind) - tau) ) / ftheta
  #########################################
  # variance of the ss estimator
  
  sigma = sqrt(n2 / n * var(yind - phihat) + n1 / n * mean((yind - tau) ^ 2))
  
  ss.sd = sigma / (ftheta * sqrt(n1)) # ss sd
  
  sup.sd = sqrt(tau - tau ^ 2) / (ftheta * sqrt(n1)) # supervised sd
  
  
  #########################################
  
  res = list("ssest" = ss, "supest" = sup, "sssd" = ss.sd, "supsd" = sup.sd)
  
  return(res)
  
}

#################################################################################




#################################################################################
# ss estimator using kernel smoothing with dimension reduction and data splitting
# sup is the initial estimator.
# ftheta is the estimator.
# tau is the quantile level.
# x is the covariate matrix of the labeled set.
# y is the response vector of the labeled set.
# yind is the indicator I(y1 < theta_initial).
# xnew is the covariate matrix of the unlabeled set.
# drmethod is the method for dimension reduction.
# drmethod = "linear" means using linear regression.
# drmethod = "logistic" means using logstic regression.
# drmethod = "save" means using SAVE on I(Y < theta).
# drmethod = "sir" means using SIR.
# drmethod = "save.con" means using SAVE on Y.
# drmethod = "plinear" means using penalized linear regression.
# drmethod = "plogistic" means using penalized logistic regression.
# r is the number of directions in dimension reduction.
# ss is the number of slices in sir.
# dr = T means applying dimension reduction via the semi-supervised SIR.
# k is the number of folds for data splitting
# There are four outputs: 
# $ ssest is the ss estimator; 
# $ supest is the supervised estimator.
# $ sssd is the ss standard deviation
# $ supsd is the supervised standard deviation.
ksedrds = function(sup, ftheta, tau, x, y, yind, xnew, 
                   drmethod = NULL, r = NULL, ss = NULL, dr = F, k)
{
  
  n1 = nrow(x) # size of the labeled set
  n2 = nrow(xnew) # size of the unlabeled set
  n = n1 + n2
  
  
  #########################################
  # dimension reduction
  transx = drtrans(x, y, yind, xnew, drmethod, r, ss, dr)
  
  x1 = transx $ x1
  #########################################
  
  p = ncol(x1) # dimensionality (after dimension reduction if applied)
  
  #########################################
  # decide whether the dimension reduction process chooses a null model
  cons = constant(x1) 
  ksind = all(cons == 1) # ksind == 1 means no kernel smoothing
  #########################################
  
  if(ksind == 1)
  {
    
    ssest = sup
    
    sup.sd = sqrt(tau - tau ^ 2) / (ftheta * sqrt(n1)) # supervised sd
    
    ss.sd = sup.sd # semi-supervised sd
    
  } else {
    
    x1 = x1[, !cons]
    
    #########################################
    # choose the bandwidth
    
    # np.bw = npregbw(xdat = x1, ydat = yind, bwmethod="cv.ls", nmulti = 1)
    # bw.opt = np.bw $ bw # optimal bandwidth for full data
    
    sink("aux")
    np.bw = npcdensbw(xdat = x1, ydat = as.factor(yind), nmulti = 1)
    sink(NULL)
    bw.opt = np.bw $ xbw # optimal bandwidth for full data
    
    
    nk = floor(n1 / k) # size of the fold in data splitting
    
    h = bw.opt * ( ( n1 / (n1 - nk) ) ^ ( 1 / (4 + p) ) ) 
    # optimal bandwidth adjusted for data splitting
    #########################################
    
    
    #########################################
    # estimate conditional expectations with data splitting
    
    sum1 = 0 # sum of fitted values of the labeled set
    sum2 = 0 # sum of fitted values of the unlabeled set
    phihat = numeric(n1) # estimator of F(theta | g(X))
    
    for(i in 1 : (k - 1))
    {
      ind = (nk * (i - 1) + 1) : (nk * i)
      ind1 = 1 : length(ind)
      
      xx1 = x[-ind, ]
      yy1 = y[-ind]
      yyind = yind[-ind]
      xx2 = rbind(x[ind, ], xnew)
      
      transx.ds = drtrans(xx1, yy1, yyind, xx2, drmethod, r, ss, dr)
      
      xx1 = transx.ds $ x1
      xx2 = transx.ds $ x2
      
      cons.ds = constant(xx1) 
      ksind.ds = all(cons.ds == 1) # ksind == 1 means no kernel smoothing
      
      if(ksind.ds == 1)
      {
        
        fit = rep( mean(yyind), nrow(xx2) )
        
      } else {
        
        xx1 = xx1[, !cons.ds]
        xx2 = xx2[, !cons.ds]
        
        sink("aux")
        np.reg = npreg(txdat = xx1, tydat = yyind, bws = h, exdat = xx2)
        sink(NULL)
        fit = np.reg $ mean
        
      }
      
      phihat[ind] = fit[ind1]
      
      sum1 = sum1 + sum(fit[ind1])
      sum2 = sum2 + sum(fit[-ind1])
      
    }
    
    
    ind = (nk * (k - 1) + 1) : n1
    ind1 = 1 : length(ind)
    
    xx1 = x[-ind, ]
    yy1 = y[-ind]
    yyind = yind[-ind]
    xx2 = rbind(x[ind, ], xnew)
    
    transx.ds = drtrans(xx1, yy1, yyind, xx2, drmethod, r, ss, dr)
    
    xx1 = transx.ds $ x1
    xx2 = transx.ds $ x2
    
    cons.ds = constant(xx1) 
    ksind.ds = all(cons.ds == 1) # ksind == 1 means no kernel smoothing
    
    if(ksind.ds == 1)
    {
      
      fit = rep( mean(yyind), nrow(xx2) )
      
    } else {
      
      xx1 = xx1[, !cons.ds]
      xx2 = xx2[, !cons.ds]
      
      sink("aux")
      np.reg = npreg(txdat = xx1, tydat = yyind, bws = h, exdat = xx2)
      sink(NULL)
      fit = np.reg $ mean
      
    }
    
    phihat[ind] = fit[ind1]
    
    sum1 = sum1 + sum(fit[ind1])
    sum2 = sum2 + sum(fit[-ind1])
    
    #########################################
    imp = sum1 * n2 / (n1 * n) - sum2 / (n * k)
    
    ssest = sup + ( imp - (mean(yind) - tau) ) / ftheta
    
    
    #########################################
    # variance of the ss estimator
    sigma = sqrt(n2 / n * mean((yind - phihat) ^ 2) + n1 / n * mean((yind - tau) ^ 2)) # good
    
    
    ss.sd = sigma / (ftheta * sqrt(n1)) # ss sd
    
    sup.sd = sqrt(tau - tau ^ 2) / (ftheta * sqrt(n1)) # supervised sd
    
    
    #########################################
    
  }
  
  res = list("ssest" = ssest, "supest" = sup, "sssd" = ss.sd, "supsd" = sup.sd)
  
  return(res)
  
}

#################################################################################
# computing entirely parametric imputation functions
# x is the covariate matrix of the labeled set.
# y is the response vector of the labeled set.
# yind is the indicator I(y1 < theta_initial).
# xnew is the covariate matrix of the unlabeled set.
# epimethod = "logistic", linear", "plogistic" or "plinear" means logistic, linear, 
# penalized logistic or penalized linear model for the entirely parametrically imputation.
# The output is the fitted values at xnew.
epifit = function(x, y, yind, xnew, epimethod)
{
  
  if(epimethod == "logistic")
  {
    
    mod = glmnet(x, yind, family = "binomial", lambda = 0)
    fit = predict(mod, newx =  xnew, type = "response")
    
  }
  
  if (epimethod == "linear")
  {
    
    mod = glmnet(x, y, family = "gaussian", lambda = 0)
    fit = predict(mod, newx =  xnew, type = "response")
  }
  
  if(epimethod == "plogistic")
  {
    
    mod = cv.glmnet(x, yind, family = "binomial")
    fit = predict(mod, newx =  xnew, s = "lambda.min", type = "response")
    
  }
  
  if (epimethod == "plinear")
  {
    
    mod = cv.glmnet(x, y, family = "gaussian")
    fit = predict(mod, newx =  xnew, s = "lambda.min", type = "response")
    
  }
  
  return(fit)
}

#################################################################################

#################################################################################
# supervised dimension reduction for semi-supervised data
# x is the covariate matrix of the labeled set.
# y is the response vector of the labeled set.
# yind is the indicator I(y1 < theta_initial).
# xnew is the covariate matrix of the unlabeled set.
# drmethod is the method for dimension reduction.
# drmethod = "linear" means using linear regression.
# drmethod = "logistic" means using logstic regression.
# drmethod = "save" means using SAVE on I(Y < theta).
# drmethod = "sir" means using SIR.
# drmethod = "save.con" means using SAVE on Y.
# drmethod = "plinear" means using penalized linear regression.
# drmethod = "plogistic" means using penalized logistic regression.
# drmethod = "psir" means using penalized SIR.
# r is the number of directions in dimension reduction.
# ss is the number of slices in sir.
# dr = T means applying dimension reduction via the semi-supervised SIR.
# There are two outputs: $x1 is the transformation of the labeled covariate matrix;
# $x2 is the transformation of the unlabaled covariate matrix.
drtrans = function(x, y, yind, xnew, drmethod = NULL, r = NULL, ss = NULL, dr = F)
{
  
  if(dr == T)
  {
    
    if(drmethod == "linear")
    {
      
      slm = glmnet(x, y, family = "gaussian",lambda = 0) 
      
      
      x1 = predict(slm, newx = x, type = "response")
      x2 = predict(slm, newx = xnew, type = "response")
      
    }
    
    if(drmethod == "logistic")
    {
      
      slm = glmnet(x, yind, family = "binomial",lambda = 0) 
      
      
      x1 = predict(slm, newx = x, type = "link")
      x2 = predict(slm, newx = xnew, type = "link")
      
    }
    
    if(drmethod == "save")
    {
      
      tm = drsave(x, yind, r)
      
      x1 = x %*% tm
      x2 = xnew %*% tm
      
    }
    
    if(drmethod == "sir")
    {
      
      tm = drsir(x, y, r, ss)
      
      x1 = x %*% tm
      x2 = xnew %*% tm
      
    }
    
    if(drmethod == "save.con")
    {
      
      tm = drsave.con(x, y, r, ss)
      
      x1 = x %*% tm
      x2 = xnew %*% tm
      
    }
    
    if(drmethod == "plinear")
    {
      
      slm = cv.glmnet(x, y, family = "gaussian") # 10-fold cv to choose lambda
      
      x1 = predict(slm, newx = x, s = "lambda.min", type = "response")
      x2 = predict(slm, newx = xnew, s = "lambda.min", type = "response")
      
    }
    
    if(drmethod == "plogistic")
    {
      
      slm = cv.glmnet(x, yind, family = "binomial") # 10-fold cv to choose lambda
      
      x1 = predict(slm, newx = x, s = "lambda.min", type = "link")
      x2 = predict(slm, newx = xnew, s = "lambda.min", type = "link")
      
    }
    
    if(drmethod == "psir")
    {
      
      tm = LassoSIR(X = x, Y = y, H = ss, no.dim = r) $ beta
      
      x1 = x %*% tm
      x2 = xnew %*% tm
    }
    
    
  } else {
    
    x1 = x
    x2 = xnew
    
  }
  
  if(is.vector(x1))
  {
    
    x1 = matrix(x1, ncol = 1)
    x2 = matrix(x2, ncol = 1)
    
  } 
  
  res = list("x1" = x1, "x2" = x2)
  
  return(res)
  
}
#################################################################################

#################################################################################
# dimension reduction on a continuous response via SAVE
# x is the covariate matrix. 
# y is the response vector. 
# r is the number of directions.
# ss is the number of slices.
# The output is a p * r transformation matrix whose columns are the directions 
# where p is column number of x.
drsave.con = function(x, y, r, ss)
{
  
  n = nrow(x)
  p = ncol(x)
  
  mu = apply(x, 2, mean)
  sigma = t(x) %*% x / (n-1) - mu %*% t(mu) * n / (n-1)
  sqrtinv = solve(sqrtm(sigma))
  
  z = (x - matrix(rep(mu, n), nrow = n, byrow = T)) %*% sqrtinv
  
  m = matrix(0, p, p)
  
  
  
  ns = floor(n / ss)
  
  yorder = order(y)
  
  for(i in 1 : (ss - 1))
  {
    
    n0 = ns
    
    con = yorder[((i - 1) * ns + 1) : (i * ns)]
    
    z0 = z[con, ]
    
    mu0 = apply(z0, 2, mean)
    sigma0 = t(z0) %*% z0 / (n0-1) - mu0 %*% t(mu0) * n0 / (n0-1)
    m0 = diag(p) - sigma0
    
    
    
    m = m + m0 %*% m0 * n0 / n
    
  }
  
  n0 = n - (ss - 1) * ns
  
  con = yorder[(n - n0 + 1) : n]
  
  z0 = z[con, ]
  
  mu0 = apply(z0, 2, mean)
  sigma0 = t(z0) %*% z0 / (n0-1) - mu0 %*% t(mu0) * n0 / (n0-1)
  m0 = diag(p) - sigma0
  
  
  
  m = m + m0 %*% m0 * n0 / n
  
  
  
  decom = eigen(m)
  location = order(abs(decom $ values), decreasing = T)[1 : r]
  
  omega = as.matrix(decom $ vectors[, location])
  
  res = sqrtinv %*% omega
  
  return(res)
}
#################################################################################


#################################################################################
# dimension reduction on a binary response via SAVE
# x is the covariate matrix. 
# y is the response vector. 
# r is the number of directions.
# The output is a p * r transformation matrix whose columns are the directions 
# where p is column number of x.
drsave = function(x, y, r)
{
  
  n = nrow(x)
  p = ncol(x)
  
  mu = apply(x, 2, mean)
  sigma = t(x) %*% x / (n-1) - mu %*% t(mu) * n / (n-1)
  sqrtinv = solve(sqrtm(sigma))
  
  z = (x - matrix(rep(mu, n), nrow = n, byrow = T)) %*% sqrtinv
  
  f = mean(y)
  
  z0 = z[y == 0, ]
  z1 = z[y == 1, ]
  n0 = nrow(z0)
  n1 = nrow(z1)
  
  mu0 = apply(z0, 2, mean)
  sigma0 = t(z0) %*% z0 / (n0-1) - mu0 %*% t(mu0) * n0 / (n0-1)
  mm0 = diag(p) - sigma0
  mu1 = apply(z1, 2, mean)
  sigma1 = t(z1) %*% z1 / (n1 - 1) - mu1 %*% t(mu1) * n1 / (n1-1)
  mm1 = diag(p) - sigma1
  
  omega = (1 - f) * (mm0 %*% mm0) + f * (mm1 %*% mm1)
  
  decom = eigen(omega)
  location = order(abs(decom $ values), decreasing = T)[1 : r]
  
  m = as.matrix(decom $ vectors[, location])
  
  res = sqrtinv %*% m
  
  return(res)
}
#################################################################################




#################################################################################
# dimension reduction via SIR
# x is the covariate matrix. 
# y is the response vector. 
# r is the number of directions.
# ss is the number of slices.
# smethod = eqsp means equal slice space. 
# smethod = "eqnb" means equal number of observations in each slice.
# The output is a p * r transformation matrix whose columns are the directions 
# where p is column number of x.
drsir = function(x, y, r, ss, smethod = "eqsp")
{
  
  n = nrow(x)
  p = ncol(x)
  
  mu = apply(x, 2, mean)
  sigma = t(x) %*% x / (n-1) - mu %*% t(mu) * n / (n-1)
  sqrtinv = solve(sqrtm(sigma))
  
  z = (x - matrix(rep(mu, n), nrow = n, byrow = T)) %*% sqrtinv
  
  m = matrix(0, p, p)
  
  if(smethod == "eqsp")
  {
    
    ymax = max(y)
    ymin = min(y)
    gap = (ymax - ymin) / ss
    
    
    con = (!(y > ymin + gap)) 
    n0 = sum(con)
    
    
    if (n0 > 0)
    {
      z0 = z[con, ]
      
      if (is.vector(z0))
      {
        
        m0 = z0
        
      } else
      {
        
        m0 = apply(z0, 2, mean)
        
      }
      
      m = m + m0 %*% t(m0) * n0 / n
    }
    
    
    for (i in 2 : ss)
    {
      con = ( (y > ymin + (i-1) * gap) & (!(y > ymin + i * gap)) )
      n0 = sum(con)
      
      if (n0 > 0)
      {
        z0 = z[con, ]
        
        if (is.vector(z0))
        {
          
          m0 = z0
          
        } else
        {
          
          m0 = apply(z0, 2, mean)
          
        }
        
        m = m + m0 %*% t(m0) * n0 / n
      }
      
    }
    
  }
  
  
  if(smethod == "eqnb")
  {
    
    ns = floor(n / ss)
    
    yorder = order(y)
    
    for(i in 1 : (ss - 1))
    {
      
      con = yorder[((i - 1) * ns + 1) : (i * ns)]
      
      z0 = z[con, ]
      
      m0 = apply(z0, 2, mean)
      
      m = m + m0 %*% t(m0) * ns / n
      
    }
    
    con = yorder[((ss - 1) * ns + 1) : n]
    
    z0 = z[con, ]
    
    m0 = apply(z0, 2, mean)
    
    m = m + m0 %*% t(m0) * (n - (ss - 1) * ns) / n
    
    
  }
  
  
  decom = eigen(m)
  location = order(abs(decom $ values), decreasing = T)[1 : r]
  
  omega = as.matrix(decom $ vectors[, location])
  
  res = sqrtinv %*% omega
  
  return(res)
}
#################################################################################


#################################################################################
# determine whether the columns of a matrix are constant
# The output is a vector whose jth component is TRUE if the jth column of x is constant.
constant = function(x)
{
  
  if(is.vector(x)) x = matrix(x, ncol = 1)
  
  p = ncol(x)
  
  dd = numeric(p)
  for(i in 1 : p) dd[i] = length( unique(x[, i]) )
  
  
  return(dd == 1)
  
}
#################################################################################


#################################################################################
# logit function
hf = function(x) 1 / (1 + exp(-x))
#################################################################################


#################################################################################
# regression functions
#########################
# Y is independent of X
cm1 = function(x, b)
{
  return(numeric(nrow(x)))
}
#########################


#########################
# linear model
cm2 = function(x, b)
{
  
  res = (x %*% b)[, 1]
  
  return(res)
}
#########################

#########################
# single index model
cm3 = function(x, b)
{
  
  q = sum(b != 0)
  
  res = ((x %*% b) + (x %*% b) ^ 2 / q)[, 1]
  
  return(res)
}
#########################

#########################
# double index model
# cm4 = function(x, b)
# {
#   p = ncol(x)
#   
#   delta = c(rep(0, p/2), rep(1, p/2))
#   
#   res = ((x %*% b) * (1 + x %*% delta))[, 1]
#   
#   return(res)
# }

cm4 = function(x, b)
{
  
  p = ncol(x)
  
  q = sum(b != 0)
  
  r = ceiling(q / 2)
  
  delta = c(rep(0, p - r), rep(1, r))
  
  res = ( (x %*% b) * (1 + 2 * x %*% delta / q) )[, 1]
  
  return(res)
}
#########################

#########################
# non-index model (quadratic model)
cm5 = function(x, b)
{
  
  res = (x %*% b)[, 1] + ( (x ^ 2) %*% b / 3 )[, 1]
  
  
  return(res)
}
#########################

#########################
# three index model
cm6 = function(x, b)
{
  p = ncol(x)
  
  delta = c(rep(0, p/2), rep(1, p/2))
  omega = rep(c(1, 0), p/2)
  
  res = ((x %*% b) * (1 + x %*% delta) + (x %*% omega) ^ 2)[, 1]
  
  return(res)
}
#########################
#################################################################################