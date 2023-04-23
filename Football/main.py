#python -m uvicorn main:app --reload
#http://127.0.0.1:8000/docs#/
 
from fastapi import FastAPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from fastapi.middleware.cors import CORSMiddleware 
app = FastAPI()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root():
    return {"message": "Hello World "}

@app.get("/team_code")
async def root( ): 
    matches = pd.read_csv("data/matches.csv", index_col=0) 
   
    dictionary=dict( enumerate(matches["team"].astype("category").cat.categories ) )
    team= []
    ids=[]
    for key , value in dictionary.items():
        team.append(value)
        ids.append(key) 

    return {"teams":team , "ids":ids , "dict" : dictionary}

@app.get("/opp_code")
async def root( ):
    matches = pd.read_csv("data/matches.csv", index_col=0) 
   
    dictionary=dict( enumerate(matches["opponent"].astype("category").cat.categories ) )
    opp = []
    id = []
    for key , value in dictionary.items():
        opp.append(value)
        id.append(key)
    return {"opp":opp , "id" :id , "dic": dictionary}

@app.get("/predictRandomForest")
async def root( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int):
    return predictRandomForest( team_code, opp_code, venue_code, hour, day_code)

def predictRandomForest( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int): 
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


    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
    train = matches[matches["date"] < '2022-01-01']
    test = matches[matches["date"] > '2022-01-01']

    predictors = ["team_code", "opp_code", "venue_code", "hour", "day_code"]

    rf.fit(train[predictors], train["target"]) 
 

    inputData={"team_code":team_code, "opp_code":opp_code, "venue_code":venue_code, "hour":hour, "day_code":day_code}
    inputData=pd.DataFrame.from_dict(inputData, orient='index').T

    #return inputData

    predcts = rf.predict(inputData)
    print(predcts)
    res=False
    if len(predcts)>0 and predcts[0]==1  :
        res=True
    return {"result": res}


@app.get("/predictKNearestNeighbor")
async def root( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int):
    return predictKNearestNeighbor( team_code, opp_code, venue_code, hour, day_code)

def predictKNearestNeighbor( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int): 
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


    
    from sklearn.neighbors import KNeighborsClassifier
    kn=KNeighborsClassifier(n_neighbors=3)
    
    train = matches[matches["date"] < '2022-01-01']
    test = matches[matches["date"] > '2022-01-01']

    predictors = ["team_code", "opp_code", "venue_code", "hour", "day_code"]

    kn.fit(train[predictors], train["target"]) 
 

    inputData={"team_code":team_code, "opp_code":opp_code, "venue_code":venue_code, "hour":hour, "day_code":day_code}
    inputData=pd.DataFrame.from_dict(inputData, orient='index').T

    #return inputData

    predcts = kn.predict(inputData)
    print(predcts)
    res=False
    if len(predcts)>0 and predcts[0]==1  :
        res=True
    return {"result": res}



#############


@app.get("/predictSVM")
async def root( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int):
    return predictSVM( team_code, opp_code, venue_code, hour, day_code)

def predictSVM( team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int): 
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

    
    from sklearn import svm
    clf = svm.SVC(kernel='linear')
    
    train = matches[matches["date"] < '2022-01-01']
    test = matches[matches["date"] > '2022-01-01']

    predictors = ["team_code", "opp_code", "venue_code", "hour", "day_code"]

    clf.fit(train[predictors], train["target"]) 

    inputData={"team_code":team_code, "opp_code":opp_code, "venue_code":venue_code, "hour":hour, "day_code":day_code}
    inputData=pd.DataFrame.from_dict(inputData, orient='index').T

    predcts = clf.predict(inputData)
    res=False
    if len(predcts)>0 and predcts[0]==1:
        res=True
    return {"result": res}


##############



@app.get("/predictSVMM")
async def root(team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int):
    return predictSVM(team_code, opp_code, venue_code, hour, day_code)

def predictSVMM(team_code:int, opp_code:int, venue_code:int, hour:int, day_code:int):
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
    from sklearn import svm
    clff = svm.SVC(kernel='rbf')
    train = matches[matches["date"] < '2022-01-01']
    test = matches[matches["date"] > '2022-01-01']

    predictors = ["team_code", "opp_code", "venue_code", "hour", "day_code"]
    clff.fit(train[predictors], train["target"])

    inputData = {"team_code":team_code, "opp_code":opp_code, "venue_code":venue_code, "hour":hour, "day_code":day_code}
    inputData = pd.DataFrame.from_dict(inputData, orient='index').T

    predcts = clff.predict(inputData)
    res = False
    if len(predcts) > 0 and predcts[0] == 1:
        res = True
    return {"result": res}

