# This is the program for the data analysis of semi-supervised quantile estimation.

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

dat.raw = read.csv("/home/grad/rondai/ssl/mq/NHEFS.csv") # load data

all.smokers = 1 # analyze all smokers (1) or those quitting smoking (0)
#################################
# filter the data
if(all.smokers == 1) dat = dat.raw
if(all.smokers == 0) dat = dat.raw[dat.raw $ qsmk == 0, ]
#################################

vname = c("wt82", "sex", "age", "race", "education","smokeintensity",
          "smokeyrs", "exercise", "active", "wt71", "cholesterol", "ht",
          "alcoholfreq", "sbp", "dbp", "price71", "price82", "tax71",
          "tax82", "asthma", "allergies") # names of the covariates


#################################
# preprocess the data so that they can be directly plugged in the function
ind = numeric(length(vname))
for(i in 1 : length(vname)) ind[i] = which(colnames(dat) == vname[i])
dat = dat[, ind]
dat = na.omit(dat)
dat = dat[dat $ alcoholfreq != 5, ]
x = as.matrix(dat[, -1])
y = dat[, 1]
#################################

s = 600 # number of replications

n1 = 200 # size of the labeled set
n = length(y) # total sample size 
n2 = n - n1 # size of the unlabeled set

tau = 0.5 # quantile level
k = 10 # number of folds in cross fitting
ss = ceiling(n1 / 5)


method = "ks" # method to estimate the outcome model
drmethod = "psir" # method for dimension reduction
epimethod = "plogistic" # method for parametric regression

#################################
# number of slices for the sliced inverse regression
if(drmethod == "sir") ss = ceiling(n1 / 5)
if(drmethod == "psir") ss = ceiling(n1 / 75)
#################################

q0 = quantile(y, tau) # gold standard estimator

output = numeric(10)

#######################################
# calculate the estimators
ff = foreach(i = 1 : s, .combine = rbind, .errorhandling = "remove") %dopar%
  {
    set.seed(888 * i)
    
    lab = sample(1 : n, n1)
    
    x1 = x[lab, ]
    y1 = y[lab]
    
    
    x2 = x[-lab, ]
    
    
    
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


ave = apply(ff, 2, mean)

res.sup = c(sd(ff[, 1]), ave[1 : 5])
res.ss = c(sd(ff[, 6]), ave[6 : 10])

res = c(res.sup[3] / res.ss[3], ave[1] + q0, res.sup, ave[6] + q0, res.ss)

names(res) = c("RE", "Sup.Est", "Sup.ESE", "Sup.Bias", "Sup.MSE", "Sup.ASE",
               "Sup.CIL", "Sup.CR", "SS.Est", "SS.ESE", "SS.Bias", "SS.MSE", "SS.ASE",
               "SS.CIL", "SS.CR")

nrow(ff)

res
#######################################





