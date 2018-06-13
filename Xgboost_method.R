library(xgboost)
library(caret)
library(tidyverse)
library(dummies)

# Loading Data
credit_data = read.csv("application_train.csv")

# Dummy variables creation

dummy_NAME_CONTRACT_TYPE = data.frame(dummy(credit_data$NAME_CONTRACT_TYPE))

dummy_CODE_GENDER = data.frame(dummy(credit_data$CODE_GENDER))

dummy_FLAG_OWN_CAR = data.frame(dummy(credit_data$FLAG_OWN_CAR))

dummy_FLAG_OWN_REALTY = data.frame(dummy(credit_data$FLAG_OWN_REALTY)) 

dummy_NAME_TYPE_SUITE = data.frame(dummy(credit_data$NAME_TYPE_SUITE))

dummy_NAME_INCOME_TYPE = data.frame(dummy(credit_data$NAME_INCOME_TYPE)) 

dummy_NAME_EDUCATION_TYPE = data.frame(dummy(credit_data$NAME_EDUCATION_TYPE)) 

dummy_NAME_FAMILY_STATUS  = data.frame(dummy(credit_data$NAME_FAMILY_STATUS )) 

dummy_HOUSING_TYPE = data.frame(dummy(credit_data$NAME_HOUSING_TYPE))

dummy_WEEKDAY_APPR_PROCESS_START = data.frame(dummy(credit_data$WEEKDAY_APPR_PROCESS_START))

dummy_FONDKAPREMONT_MODE = data.frame(dummy(credit_data$FONDKAPREMONT_MODE))

dummy_HOUSETYPE_MODE = data.frame(dummy(credit_data$HOUSETYPE_MODE))

dummy_WALLSMATERIAL_MODE = data.frame(dummy(credit_data$WALLSMATERIAL_MODE))

dummy_EMERGENCYSTATE_MODE = data.frame(dummy(credit_data$EMERGENCYSTATE_MODE))



continuous_columns = c(                 
                       "CNT_CHILDREN", "AMT_INCOME_TOTAL"  ,         
                       "AMT_CREDIT"  ,  "AMT_ANNUITY"  ,   
                       "AMT_GOODS_PRICE", "REGION_POPULATION_RELATIVE" ,"DAYS_BIRTH"  ,
                       "DAYS_EMPLOYED"  ,  "DAYS_REGISTRATION",
                       "DAYS_ID_PUBLISH"  , "OWN_CAR_AGE",           
                       "FLAG_MOBIL" , "FLAG_EMP_PHONE"   ,           
                       "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE"  ,          
                       "FLAG_PHONE" , "FLAG_EMAIL", "CNT_FAM_MEMBERS",        
                       "REGION_RATING_CLIENT" , "REGION_RATING_CLIENT_W_CITY" , "HOUR_APPR_PROCESS_START"   ,  
                       "REG_REGION_NOT_LIVE_REGION" , "REG_REGION_NOT_WORK_REGION" , 
                       "LIVE_REGION_NOT_WORK_REGION" , "REG_CITY_NOT_LIVE_CITY"  ,
                       "REG_CITY_NOT_WORK_CITY" , "LIVE_CITY_NOT_WORK_CITY" ,"EXT_SOURCE_1" ,     
                       "EXT_SOURCE_2" ,"EXT_SOURCE_3",    
                       "APARTMENTS_AVG" ,"BASEMENTAREA_AVG" ,   
                       "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG",   
                       "COMMONAREA_AVG" , "ELEVATORS_AVG" , 
                       "ENTRANCES_AVG"  , "FLOORSMAX_AVG" ,       
                       "FLOORSMIN_AVG" ,  "LANDAREA_AVG",      
                       "LIVINGAPARTMENTS_AVG","LIVINGAREA_AVG",     
                       "NONLIVINGAPARTMENTS_AVG", "NONLIVINGAREA_AVG",    
                       "APARTMENTS_MODE","BASEMENTAREA_MODE",   
                       "YEARS_BEGINEXPLUATATION_MODE", "YEARS_BUILD_MODE",         
                       "COMMONAREA_MODE" ,"ELEVATORS_MODE" ,        
                       "ENTRANCES_MODE","FLOORSMAX_MODE",       
                       "FLOORSMIN_MODE","LANDAREA_MODE",      
                       "LIVINGAPARTMENTS_MODE","LIVINGAREA_MODE",     
                       "NONLIVINGAPARTMENTS_MODE","NONLIVINGAREA_MODE",    
                       "APARTMENTS_MEDI","BASEMENTAREA_MEDI",   
                       "YEARS_BEGINEXPLUATATION_MEDI" , "YEARS_BUILD_MEDI",          
                       "COMMONAREA_MEDI","ELEVATORS_MEDI",        
                       "ENTRANCES_MEDI","FLOORSMAX_MEDI",       
                       "FLOORSMIN_MEDI","LANDAREA_MEDI",      
                       "LIVINGAPARTMENTS_MEDI","LIVINGAREA_MEDI",     
                       "NONLIVINGAPARTMENTS_MEDI",  "NONLIVINGAREA_MEDI",    
                       "TOTALAREA_MODE", "OBS_30_CNT_SOCIAL_CIRCLE", 
                       "DEF_30_CNT_SOCIAL_CIRCLE"  ,   "OBS_60_CNT_SOCIAL_CIRCLE",
                       "DEF_60_CNT_SOCIAL_CIRCLE" ,    "DAYS_LAST_PHONE_CHANGE",
                       "FLAG_DOCUMENT_2","FLAG_DOCUMENT_3",
                       "FLAG_DOCUMENT_4","FLAG_DOCUMENT_5",
                       "FLAG_DOCUMENT_6","FLAG_DOCUMENT_7",
                       "FLAG_DOCUMENT_8","FLAG_DOCUMENT_9",
                       "FLAG_DOCUMENT_10","FLAG_DOCUMENT_11",
                       "FLAG_DOCUMENT_12","FLAG_DOCUMENT_13",
                       "FLAG_DOCUMENT_14","FLAG_DOCUMENT_15",
                       "FLAG_DOCUMENT_16","FLAG_DOCUMENT_17",
                       "FLAG_DOCUMENT_18","FLAG_DOCUMENT_19",
                       "FLAG_DOCUMENT_20","FLAG_DOCUMENT_21",
                       "AMT_REQ_CREDIT_BUREAU_HOUR" ,"AMT_REQ_CREDIT_BUREAU_DAY",
                       "AMT_REQ_CREDIT_BUREAU_WEEK","AMT_REQ_CREDIT_BUREAU_MON",
                       "AMT_REQ_CREDIT_BUREAU_QRT" ,"AMT_REQ_CREDIT_BUREAU_YEAR", "TARGET"  )

credit_continuous = credit_data[,continuous_columns]

credit_data_new =
  cbind(dummy_NAME_CONTRACT_TYPE, dummy_CODE_GENDER, dummy_FLAG_OWN_CAR, dummy_FLAG_OWN_REALTY,
        dummy_NAME_TYPE_SUITE, dummy_NAME_INCOME_TYPE, dummy_NAME_EDUCATION_TYPE, dummy_NAME_FAMILY_STATUS,
        dummy_HOUSING_TYPE, dummy_WEEKDAY_APPR_PROCESS_START, dummy_FONDKAPREMONT_MODE, dummy_HOUSETYPE_MODE,
        dummy_WALLSMATERIAL_MODE, dummy_EMERGENCYSTATE_MODE,credit_continuous)

# Setting seed for repeatability of results of train & test split
set.seed(123)
numrow = nrow(credit_data_new)
trnind = sample(1:numrow,size = as.integer(0.7*numrow))
train_data = credit_data_new[trnind,]
test_data = credit_data_new[-trnind,]

y = train_data$TARGET

#Xgboost classifier

xgb = xgboost(data = data.matrix(train_data[, -174]), label = y, eta = 0.04, max_depth = 2, 
              nrounds = 5000, subsample = 0.5, colsample_bytree = 0.5, seed = 1,
              eval_metric = "logloss",
              objective = "binary:logistic", nthread = 3)



# custom functions for calculation of precision and recall

frac_trzero = (table(train_data$TARGET)[[1]])/nrow(train_data)
frac_trone = (table(train_data$TARGET)[[2]])/nrow(train_data)

frac_tszero = (table(test_data$TARGET)[[1]])/nrow(train_data)
frac_tsone = (table(test_data$TARGET)[[2]])/nrow(train_data)

prec_zero = function(act, pred){
  tble = table(act, pred)
  return(round(tble[1,1]/(tble[1,1]+tble[2,1]), 4))
}

prec_one = function(act, pred){
  tble = table(act, pred)
  return(round(tble[2,2]/(tble[2,2]+tble[1,2]), 4))
}

recl_zero = function(act, pred){
  tble = table(act, pred)
  return(round(tble[1,1]/(tble[1,1]+tble[1,2]), 4))
}

recl_one = function(act, pred){
  tble = table(act, pred)
  return(round(tble[2,2]/(tble[2,2]+tble[2,1]), 4))
}

accrcy = function(act, pred){
  tble = table(act, pred)
  return(round((tble[1,1]+tble[2,2])/sum(tble), 4))
}

#XGBoost value prediction on train and test data

tr_y_pred_prob = predict(xgb, data.matrix(train_data[, -174]))
tr_y_pred = as.numeric(tr_y_pred_prob > 0.5)

ts_y_pred_prob = predict(xgb, data.matrix(test_data[, -174]))
ts_y_pred = as.numeric(ts_y_pred_prob > 0.5)

tr_y_act = train_data$TARGET
ts_y_act = test_data$TARGET

tr_tble = table(tr_y_act, tr_y_pred)

#XGBoost Metric prediction on Train Data
print(paste("XGBoost - Train Confusion Matrix"))
print(tr_tble)

tr_acc = accrcy(tr_y_act, tr_y_pred)

trprec_zero = prec_zero(tr_y_act, tr_y_pred)
trrecl_zero = recl_zero(tr_y_act, tr_y_pred)

trprec_one = prec_one(tr_y_act, tr_y_pred)
trrecl_one = recl_one(tr_y_act, tr_y_pred)

trprec_ovll = trprec_zero*frac_trzero + trprec_one*frac_trone
trrecl_ovll = trrecl_zero*frac_trzero + trrecl_one*frac_trone

print(paste("XGBoost train accuracy:", tr_acc))

print(paste("Xgboost - Train Classification report"))
print(paste("Zero Precision", trprec_zero, "Zero Recall", trrecl_zero))
print(paste("One Precision", trprec_one, "One Recall", trrecl_one))
print(paste("Overall Precision", round(trprec_ovll, 4), 
            "Overall Recall", round(trrecl_ovll, 4)))

#Xgboost metric prediction on test data 

ts_tble = table(ts_y_act, ts_y_pred)
print(paste("XGBoost - Test Confusion Matrix"))
print(ts_tble)

ts_acc = accrcy(ts_y_act, ts_y_pred)

tsprec_zero = prec_zero(ts_y_act, ts_y_pred)
tsrecl_zero = recl_zero(ts_y_act, ts_y_pred)

tsprec_one = prec_one(ts_y_act, ts_y_pred)
tsrecl_one = recl_one(ts_y_act, ts_y_pred)

tsprec_ovll = tsprec_zero*frac_tszero + tsprec_one*frac_tsone
tsrecl_ovll = tsrecl_zero*frac_tszero + tsrecl_one*frac_tsone

print(paste("XGBoost train accuracy:", ts_acc))

print(paste("Xgboost - Train Classification report"))
print(paste("Zero Precision", tsprec_zero, "Zero Recall", tsrecl_zero))
print(paste("One Precision", tsprec_one, "One Recall", tsrecl_one))
print(paste("Overall Precision", round(tsprec_ovll, 4), 
            "Overall Recall", round(tsrecl_ovll, 4)))




