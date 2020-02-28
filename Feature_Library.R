# Mass atrocity prediction, Juan Diego Vera, Thomas Merkh
rm(list=ls())
library(glmnet)
load("/home/tmerkh/prepared_data_final_15Oct2018.RData")
dat=as.data.frame(dat)

# We would like to predict the start of new mass killings
outcomenames=c("anymk.start.1", "anymk.start.2window")

predictornames = c("anymk.ongoing","anymk.ever",
          "reg.afr", "reg.eap", "reg.eur", "reg.mna", "reg.sca", #americas, "reg.amr" left out as baseline.
          "countryage.ln", "popsize.ln.combined", "imr.sqrt", "gdppcgrowth.combined",
          "ios.iccpr1","includesnonstate",
          "durable.ln","minorityrule", "elf.ethnic", "battledeaths.ln",
          "candidaterestriction", "partyban","judicialreform",
          "religiousfreedom", "pol_killing_approved",
          "freemove_men4","freemove_women4", "freediscussion",
          "social_inequality","even_civilrights","repress_civilsoc","social_power_dist",
          "tradeshare.ln.combined",
          "coup.try.5yr",
          "polity2.fl.2","polity2.fl.3")

## Omit incomplete entries, this will be the Y and X data.
yXtrain2017=na.omit(dat[,c("sftgcode","year", outcomenames, predictornames, "country_name","year")])
Xtrain2017=yXtrain2017[,predictornames]      # 33 Predictors 
ytrain2017_1yr=yXtrain2017[,outcomenames[1]] # single leaded year outcome - This is just anymk.start.1 in the data
ytrain2017_2yr=yXtrain2017[,outcomenames[2]] # two year window outcome    - This is anymk.start.2window
y = as.numeric(ytrain2017_1yr)

## Create Test data
# yXtest2017= na.omit(dat[dat$year==2017,c("sftgcode","year", outcomenames, predictornames, "country_name","year")])
# yXtest2017= na.omit(dat[dat$year==2017,c("sftgcode","COWcode",outcomenames,predictornames,"country_name","year")])
Xtest2017 = na.omit(dat[dat$year==2017,c("sftgcode","COWcode",predictornames,"country_name","year")])

## Function for creating the feature library
flib_function <- function(y, df) {
form <- y ~ .^2            # This will tell the next line to add all pairwise interactions between predictors
x0 <- model.matrix(form, df)  
#drop intercept
x0 <- x0[,-1]
x = as.matrix(df)
# We can add any function of the data that we wish, like taking the sine of the data values for example
# flib <- cbind(x0,sin(x))
flib <- cbind(x0)
}

## Create the feature libraries
flib <- as.matrix(flib_function(y, Xtrain2017))
flibtest <- as.matrix(flib_function(1:dim(Xtest2017)[1], Xtest2017[,predictornames]))

#make column names unique
colnames(flib) = make.names(make.unique(colnames(flib)), unique=TRUE)
colnames(flibtest) = make.names(make.unique(colnames(flibtest)), unique=TRUE)

## glmnet-Lasso for feature/model selection
## Should we be using family="binomial" to do the logistic regression because the response is binary

## Explaination of CV.GLMNET:
# cv.glmnet() performs cross-validation
# An n-fold CV will randomly divide your observations into n non-overlapping groups/folds of approx equal size. 
# The first fold will be used for validation set and the model is fit on 9 folds.
# In the case of lasso and ridge models, CV helps choose the value of the tuning parameter lambda. 

# This may take some time, 5 minutes, depending on nlambda and nfolds
glmnet1 <- cv.glmnet(flib, y, type.measure = 'mse', nfolds = 10, alpha = 1, family = "gaussian", nlambda = 10)  
# alpha == 1 is LASSO, nlambda == 100 (number of lambdas tested)
# glmnet1$lambda.min is the lambda that results in the lowest cross validation error

c <- coef(glmnet1, s = 'lambda.min', exact = TRUE)
inds <- which(c != 0)
variables <- row.names(c)[inds]
library(intrval)
variables <- variables[variables %ni% "(Intercept)"]
lassoXtrain2017 = flib[,variables]                    # The remaining features, selected from LASSO
lassoXtest2017  = flibtest[,variables]                # Only use these the relevant features in the test data

## Perform ridge regression on the selected features, plot, and do predictions
ridge.cv.2017.1yr = cv.glmnet(y = as.numeric(ytrain2017_1yr), x = as.matrix(lassoXtrain2017), alpha = 0, family = "binomial")
# plot(ridge.cv.2017.1yr)
ridge.coefsfull.2017.1yr = coef(ridge.cv.2017.1yr, s = "lambda.min")  # Get the model coefficients
ridge.predictions = signif(predict.cv.glmnet(ridge.cv.2017.1yr, newx = as.matrix(lassoXtest2017), s = "lambda.min", type = "response"),4)

## Perform elastic-net-logit
elastic.cv.out = cv.glmnet(y = as.numeric(ytrain2017_1yr), x = as.matrix(lassoXtrain2017), alpha = 0.5, family = "binomial")
elastic.min.out = coef(elastic.cv.out, s = elastic.cv.out$lambda.min)
elastic.train.min.predictions = predict.cv.glmnet(elastic.cv.out, newx = as.matrix(lassoXtrain2017), s = "lambda.min", type = "response")
elastic.test.min.predictions = predict.cv.glmnet(elastic.cv.out, newx = as.matrix(lassoXtest2017), s = "lambda.min", type = "response")

## KRLS -- use library(KRLS2)
# X=as.matrix(lassoXtrain2017)print("KRLS crashes your fossil of a computer")
# krls.out = KRLS::krls(y=as.numeric(ytrain2017_1yr), X=as.matrix(lassoXtrain2017))
# krls.train.predictions = krlsout$fitted
# krls.test.predictions = predict(krls.out, newdata = X=as.matrix(lassoXtest2017))$fit

## Gather results, the risks of mass killings in each country in 2017
Xtest2017$risk2017.1yr.EN = as.numeric(elastic.test.min.predictions)
Xtest2017$risk2017.1yr = as.numeric(ridge.predictions)

## Hand rolled performance measurement:
# Idea: We would like quantify the goodness of each model, where we penalize false positives less than false negatives.
# We can hand craft a loss function to "go easy" on models which predict mk's a year in advance too.

# Right now, this is testing the performance of each model on the training data.
# It would be interesting to loop over thresholds to test the robustness of one model over another.
alert_threshold <- 0.10
ridge.predictions.training = signif(predict.cv.glmnet(ridge.cv.2017.1yr, newx = as.matrix(lassoXtrain2017), s = "lambda.min", type = "response"),4)
y_predicted_ridge <- (as.numeric(ridge.predictions.training) > alert_threshold)*1
y_predicted_EN <- (as.numeric(elastic.train.min.predictions) > alert_threshold)*1
y_true <- ytrain2017_1yr

# y1 == binary decision whether to alert a warning (1), or do nothing (0), based on the model
# y2 == binary, 1 if a mass killing occured, 0 else.
compare_loss <- function(y1, y2) {
  if(y1 == y2){
    return(0)
  }
  # Make false positives a little bad
  else if (y1 == 1 && y2 == 0){
    return(1)
  }
  # Make false negatives really bad
  else if (y1 == 0 && y2  == 1){
    return(3)
  }
  else {
    print("Something went wrong!")
  }
}

test_perf <- function(y_predicted, y_true) {
  performance = 0
  d = length(y_predicted)
  for(i in 1:d){
    performance = performance + compare_loss(y_predicted[i], y_true[i])
  }
  return(performance/d)
}

Ridge_perf <- test_perf(y_predicted_ridge, y_true)
EN_perf    <- test_perf(y_predicted_EN, y_true)
print(Ridge_perf)
print(EN_perf)

## Write a CSV file containing the results
everything2017 = Xtest2017 %>%  select(country_name, sftgcode, COWcode, risk2017.1yr, risk2017.1yr.EN) %>% arrange(desc(risk2017.1yr))
colnames(everything2017) = c("country","SFTGcode","COW","risk_in_2017_ridge","risk_in_2017_EN")
View(everything2017)
write.csv(x = everything2017, file = "Risk.csv" )

## Could we get a print out of the features that LASSO chose?
results<-variables
write.table(results, paste0(getwd(), "/results/lassoresults.csv"), sep = ",", row.names = F, col.names = F, quote = F, append = F)