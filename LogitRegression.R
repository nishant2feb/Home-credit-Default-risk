# Variable Importance
library(mctest)
library(dummies)
library(Information)
library(pROC)
library(dplyr)

credit_data = read.csv("application_train.csv")

# I.V Calculation

IV <- create_infotables(data=credit_data, y="TARGET", parallel=FALSE)
for (i in 1:length(colnames(credit_data))-1){
  seca = IV[[1]][i][1]
  sum(seca[[1]][5])
  print(paste(colnames(credit_data)[i],",IV_Value:",round(sum(seca[[1]][5]),4
  )))
}


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



continuous_columns = c("SK_ID_CURR" ,  "TARGET"   ,                 
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
                       "AMT_REQ_CREDIT_BUREAU_QRT" ,"AMT_REQ_CREDIT_BUREAU_YEAR"  )

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

remove_cols_extra_dummy =
  c("CODE_GENDER.XNA", "NAME_FAMILY_STATUS.Unknown")


# significant variables one by one
remove_cols_insig = c("SK_ID_CURR",
                    "CODE_GENDER",
                    "AMT_GOODS_PRICE",
                    "OWN_CAR_AGE",
                    "FLAG_EMP_PHONE",
                    "FLAG_WORK_PHONE",
                    "FLAG_CONT_MOBILE",
                    "FLAG_PHONE",
                    "OCCUPATION_TYPE",
                    "REGION_RATING_CLIENT",
                    "REGION_RATING_CLIENT_W_CITY",
                    "HOUR_APPR_PROCESS_START",
                    "REG_REGION_NOT_LIVE_REGION",
                    "REG_REGION_NOT_WORK_REGION",
                    "EMERGENCYSTATE_MODE ",
                    "OBS_30_CNT_SOCIAL_CIRCLE",
                    "DEF_30_CNT_SOCIAL_CIRCLE",
                    "OBS_60_CNT_SOCIAL_CIRCLE ",
                    "DAYS_LAST_PHONE_CHANGE",
                    "FLAG_DOCUMENT_3",
                    "FLAG_DOCUMENT_4",
                    "FLAG_DOCUMENT_5",
                    "FLAG_DOCUMENT_6",
                    "FLAG_DOCUMENT_7",
                    "FLAG_DOCUMENT_8",
                    "FLAG_DOCUMENT_9",
                    "FLAG_DOCUMENT_10",
                    "FLAG_DOCUMENT_11",
                    "FLAG_DOCUMENT_12",
                    "FLAG_DOCUMENT_13",
                    "FLAG_DOCUMENT_14",
                    "FLAG_DOCUMENT_15",
                    "FLAG_DOCUMENT_16",
                    "FLAG_DOCUMENT_17",
                    "FLAG_DOCUMENT_18",
                    "FLAG_DOCUMENT_19",
                    "FLAG_DOCUMENT_20" )

remove_cols = c(remove_cols_extra_dummy,remove_cols_insig)

glm_fit = glm(TARGET ~ .,family = "binomial",data =
                train_data[,!(names(train_data) %in% remove_cols)])

# Significance check - p_value summary(glm_fit)
# Multi collinearity check - VIF

remove_cols_vif = c(remove_cols,"TARGET")
vif_table = imcdiag(train_data[,!(names(train_data) %in%
                                    remove_cols_vif)],train_data$TARGET,detr=0.001, conf=0.99)
vif_table

# Predicting probabilities
train_data$glm_probs = predict(glm_fit,newdata = train_data,type =
                                 "response")
test_data$glm_probs = predict(glm_fit,newdata = test_data,type =
                                "response")
# Area under

ROC1 <- roc(as.factor(train_data$TARGET),train_data$glm_probs)
plot(ROC1, col = "blue")
print(paste("Area under the curve",round(auc(ROC1),4)))

# Actual prediction based on threshold tuning

threshold_vals = c(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9)

for (thld in threshold_vals){
  train_data$glm_pred = 0
  train_data$glm_pred[train_data$glm_probs>thld]=1
  tble = table(train_data$glm_pred,train_data$TARGET)
  acc = (tble[1,1]+tble[2,2])/sum(tble)
  print(paste("Threshold",thld,"Train accuracy",round(acc,4)))
}

# Best threshold from above search is 0.5 with accuracy as 0.7841

best_threshold = 0.5

# Train confusion matrix & accuracy

train_data$glm_pred = 0
train_data$glm_pred[train_data$glm_probs>best_threshold]=1
tble = table(train_data$glm_pred,train_data$TARGET)
acc = (tble[1,1]+tble[2,2])/sum(tble)
print(paste("Confusion Matrix - Train Data"))
print(tble) print(paste("Train accuracy",round(acc,4)))

# Test confusion matrix & accuracy

test_data$glm_pred = 0
test_data$glm_pred[test_data$glm_probs>best_threshold]=1
tble_test = table(test_data$glm_pred,test_data$TARGET)
acc_test = (tble_test[1,1]+tble_test[2,2])/sum(tble_test)
print(paste("Confusion Matrix - Test Data")) print(tble_test)
print(paste("Test accuracy",round(acc_test,4)))
