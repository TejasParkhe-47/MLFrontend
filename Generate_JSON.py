from Data_Pipeline import Generate_JSON
import pandas as pd

# 1. INPUT PAGE OF WEBSITE

dataset = "cleaned_data.csv"

n_estimator = 3
maxdepth = 10
randomstate = 42
testsize = 0.3

# 2. TRAINING PAGE OF WEBSITE

myObj = Generate_JSON(dataset,n_estimator,maxdepth,randomstate,testsize) # --> Generates a Python Object for Model Training

X_train, X_test, y_train, y_test = myObj.resample_split_data() # --> Resamples and then splits dataset

myObj.to_json(X_train,y_train) # --> Trains the model, tracks the flow of data of each tree in the model and stores (also generates) it in .csv and .json file

# 3. PREDICTION PAGE OF WEBSITE

Recency = 0
Income = 57091
MntGoldProds = 37
NumCatalogPurchases = 3
NumWebVisitsMonth = 5
MntFruits = 5
NumStorePurchases = 7
MntSweetProducts = 0
Year_Birth = 1967

input_value = [Recency,Income,MntGoldProds,NumCatalogPurchases,NumWebVisitsMonth,MntFruits,NumStorePurchases,MntSweetProducts,Year_Birth]

# (For Testing) print("Inputs : ",input_value,"\n")

predictions = myObj.model.predict([[Recency,Income,MntGoldProds,NumCatalogPurchases,NumWebVisitsMonth,MntFruits,NumStorePurchases,MntSweetProducts,Year_Birth]]) # --> Performs prediction
flow = myObj.output_flow # --> Stores the flow of input data of different trees of the trained model, in .csv and .json file

# (For Testing) (From here)
"""print("Decision Technique :\n")
for key,value in flow.items():
    z = 1
    print("-"*80)
    print("For Tree ",key,",")
    for val in value:
        print(z,".\t",f"{val[0]} {val[1]} {val[2]}\t--->\t{val[3]}")
        z += 1 
    print("\n")
    print("-"*80)"""
# (To here)

y_pred = myObj.predict_stats(X_test,y_test) # --> Defines model performance (accuracy and classification report)

flow_df = pd.DataFrame(flow)
flow_df.columns = [f"Tree {x+1}" for x in range(myObj.n_estimators)]
flow_df.to_json("Output/Prediction_flow.json",index=False)
# (For Testing) print("\nSaved the flow as 'Prediction_flow.json'")
flow_df.to_csv("Output/Prediction_flow.csv",index=False)
# (For Testing) print("Saved the flow as 'Prediction_flow.csv'")

final_prediction = predictions[0] # --> Final Prediction of the model

# (For Testing) print("\nPredicted Value is ",predictions[0])