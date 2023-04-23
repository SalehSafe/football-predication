import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score ,f1_score, recall_score ,classification_report
from sklearn.metrics._classification import UndefinedMetricWarning
from sklearn.svm import SVC
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore', category=UndefinedMetricWarning)


def root(team_code, opp_code, venue_code, hour, day_code):
    matches = pd.read_csv("data/matches.csv", index_col=0)
    del matches["comp"]
    del matches["notes"]
    matches["date"] = pd.to_datetime(matches["date"])
    matches["target"] = (matches["result"] == "W").astype("int")
    matches["venue_code"] = matches["venue"].astype("category").cat.codes
    matches["team_code"] = matches["team"].astype("category").cat.codes
    matches["opp_code"] = matches["opponent"].astype("category").cat.codes
    matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
    matches["day_code"] = matches["date"].dt.dayofweek

    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    kn = KNeighborsClassifier(n_neighbors=3)
    svm_linear = SVC(kernel='linear')
    svm_rbf = SVC(kernel='rbf')

    train = matches[matches["date"] < '2022-01-01']
    test = matches[matches["date"] >= '2022-01-01']

    predictors = ["team_code", "opp_code", "venue_code", "hour", "day_code"]

    rf.fit(train[predictors], train["target"])
    kn.fit(train[predictors], train["target"])
    svm_linear.fit(train[predictors], train["target"])
    svm_rbf.fit(train[predictors], train["target"])

    inputData = {"team_code": team_code, "opp_code": opp_code, "venue_code": venue_code, "hour": hour, "day_code": day_code}
    inputData = pd.DataFrame.from_dict(inputData, orient='index').T
    #Train Model
    preds_rf = rf.predict(test[predictors])
    preds_kn = kn.predict(test[predictors])
    preds_svm_linear = svm_linear.predict(test[predictors])
    preds_svm_rbf = svm_rbf.predict(test[predictors])
    #Accuracy
    acc_rf = accuracy_score(test["target"], preds_rf)
    acc_kn = accuracy_score(test["target"], preds_kn)
    acc_svm_linear = accuracy_score(test["target"], preds_svm_linear)
    acc_svm_rbf = accuracy_score(test["target"], preds_svm_rbf)
    #Precision
    prec_rf = precision_score(test["target"], preds_rf)
    prec_kn = precision_score(test["target"], preds_kn)
    prec_svm_linear = precision_score(test["target"], preds_svm_linear, zero_division=0)
    prec_svm_rbf = precision_score(test["target"], preds_svm_rbf, zero_division=0)
    #F1 Score
    f1_rf = f1_score(test["target"], preds_rf)
    f1_kn = f1_score(test["target"], preds_kn)
    f1_svm_linear = f1_score(test["target"], preds_svm_linear)
    f1_svm_rbf = f1_score(test["target"], preds_svm_rbf)
    #Recall
    rec_rf = recall_score(test["target"], preds_rf)
    rec_kn = recall_score(test["target"], preds_kn)
    rec_svm_linear = recall_score(test["target"], preds_svm_linear)
    rec_svm_rbf = recall_score(test["target"], preds_svm_rbf)
    
    ###############
    print('Random Forest:\n', classification_report(test["target"], preds_rf))
    print('K-Nearest Neighbors:\n', classification_report(test["target"], preds_kn))
    print('SVM (linear):\n', classification_report(test["target"], preds_svm_linear))
    print('SVM (RBF):\n', classification_report(test["target"], preds_svm_rbf))
    ########################
        # calculate classification reports
    rf_report = classification_report(test["target"], preds_rf)
    kn_report = classification_report(test["target"], preds_kn)
    svm_linear_report = classification_report(test["target"], preds_svm_linear)
    svm_rbf_report = classification_report(test["target"], preds_svm_rbf)

    # convert reports to tables
    rf_table = tabulate(rf_report.split("\n"), headers='keys', tablefmt='pipe')
    kn_table = tabulate(kn_report.split("\n"), headers='keys', tablefmt='pipe')
    svm_linear_table = tabulate(svm_linear_report.split("\n"), headers='keys', tablefmt='pipe')
    svm_rbf_table = tabulate(svm_rbf_report.split("\n"), headers='keys', tablefmt='pipe')

    # print tables
    print('Random Forest:\n', rf_table)
    print('K-Nearest Neighbors:\n', kn_table)
    print('SVM (linear):\n', svm_linear_table)
    print('SVM (RBF):\n', svm_rbf_table)

    

    table = [ ['Random Forest', acc_rf, prec_rf, f1_rf, rec_rf],
        ['KNN', acc_kn, prec_kn, f1_kn, rec_kn],
        ['SVM Linear', acc_svm_linear, prec_svm_linear, f1_svm_linear, rec_svm_linear],
        ['SVM RBF', acc_svm_rbf, prec_svm_rbf, f1_svm_rbf, rec_svm_rbf]
    ]

    headers = ['Model', 'Accuracy', 'Precision', 'F1 Score', 'Recall']

    print(tabulate(table, headers=headers))

root(1,2,1,15,5)

