# This is the program for the simulations of semi-supervised quantile estimation.

.libPaths("/home/grad/rondai/RLibrary")

#################################
# load packages and functions
suppressMessages({
  if (!require(doParallel)) install.packages('doParallel', repos =
                                               'https://cran.revolutionanalytics.com/')
  library(doParallel)
  
  
  library(MASS)
})

source("/home/grad/rondai/ssl/mq/ssl_functions.R")
#################################

registerDoParallel(detectCores()) # set multicores

s = 600 # number of replications

n1 = 200 # size of the labeled set
n2 = 5000 # size of the unlabeled set
n = n1 + n2 # total sample size 

tau = 0.5 # quantile level
k = 10 # number of folds in cross fitting

method = "ks" # method to estimate the outcome model
drmethod = "linear" # method for dimension reduction
epimethod = "logistic" # method for parametric regression

#################################
# number of slices for the sliced inverse regression
if(drmethod == "sir") ss = ceiling(n1 / 5)
if(drmethod == "psir") ss = ceiling(n1 / 75)
#################################

hd = 0 # high dimensional (p = 200 or 500) or not (p = 10 or 20)
sparsity = "high" # sparsity level in HD models: "high" means q = 5; "low" means q = ceiling(p ^ {1 / 2})
j1 = 1 # dimensionality of X
j2 = 4 # outcome model

#################################
# set the dimensionality of X
if(! hd) p = c(10, 20)[j1]
if(hd) p = c(200, 500)[j1]
#################################

b = rep(1, p) # parameter in the outcome model

#################################
# set the sparsity in the HD models
if(hd & sparsity == "high") b[-(1 : ceiling(sqrt(p)))] = 0
if(hd & sparsity == "low") b[-(1 : 5)] = 0
#################################

#################################
# set the outcome model
if(j2 == 1) cm = cm1 # Y is independent of X
if(j2 == 2) cm = cm2 # linear model
if(j2 == 3) cm = cm3 # single index model
if(j2 == 4) cm = cm4 # two indexes model
if(j2 == 5) cm = cm5 # non-index model (with the sum of quadratic forms)
#################################

#######################################
# True value of the parameter and components for computing 
# the oracle relative efficiency
set.seed(1)
n0 = 100000
x = mvrnorm(n0, numeric(p), diag(p))
ind = apply( x, 1, function(x) all(abs(x) < 5) )
x = x[ind, ]
ce = cm(x, b) # E(Y | X)
y = ce + rnorm(nrow(x)) # standard normal error term 
q0 = quantile(y, tau)

cvar = mean((pnorm(q0 - ce) - tau) ^ 2) # E[E{psi(Y,theta) | X} ^ 2]
#######################################

output = numeric(10)

#######################################
# calculate the estimators
ff = foreach(i = 1 : s, .combine = rbind, .errorhandling = "remove") %dopar%
  {
    set.seed(99 * i)
    
    x1 = mvrnorm(2 * n1, numeric(p), diag(p))
    ind = apply( x1, 1, function(x) all(abs(x) < 5) )
    x1 = x1[ind, ]
    x1 = x1[1 : n1, ]
    
    y1 = cm(x1, b) + rnorm(n1)
    
    x2 = mvrnorm(2 * n2, numeric(p), diag(p))
    ind = apply( x2, 1, function(x) all(abs(x) < 5) )
    x2 = x2[ind, ]
    x2 = x2[1 : n2, ]
    
    est = qss(x1, y1, x2, tau, method, drmethod, r = 2, ss, dr = T, epimethod, k)
    
    ss.est = est $ ssest
    ss.sd = est $ sssd
    ss.ci = c(ss.est - 1.96 * ss.sd, ss.est + 1.96 * ss.sd)
    
    sup.est = est $ supest
    sup.sd = est $ supsd
    sup.ci = c(sup.est - 1.96 * sup.sd, sup.est + 1.96 * sup.sd)
    
    output[1] = sup.est - q0
    output[2] = (sup.est - q0) ^ 2
    output[3] = sup.sd
    output[4] = 1.96 * sup.sd
    output[5] = !((q0 < sup.ci[1]) | (q0 > sup.ci[2]))
    
    output[6] = ss.est - q0
    output[7] = (ss.est - q0) ^ 2
    output[8] = ss.sd
    output[9] = 1.96 * ss.sd
    output[10] = !((q0 < ss.ci[1]) | (q0 > ss.ci[2]))
    
    
    output
    
  }    
#######################################
registerDoSEQ()

#######################################
# summarize the result
ff = na.omit(ff)

ff = ff[1 : min(500, nrow(ff)), ]


ore = 1 / (1 - (n2 / n) * cvar / (tau - tau ^ 2))

ave = apply(ff, 2, mean)

res.sup = c(sd(ff[, 1]), ave[1 : 5])
res.ss = c(sd(ff[, 6]), ave[6 : 10])

res = c(res.sup[3] / res.ss[3], ore, res.sup, res.ss)


names(res) = c("RE", "ORE", "Sup.ESE", "Sup.Bias", "Sup.MSE", "Sup.ASE",
               "Sup.CIL", "Sup.CR", "SS.ESE", "SS.Bias", "SS.MSE", "SS.ASE",
               "SS.CIL", "SS.CR")

nrow(ff)

res
#######################################




