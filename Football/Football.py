
import pandas as pd
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

##############################################

#rf.fit(train[predictors], train["target"])  
import torch
class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Net, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid() 
    def forward(self, x):
        hidden = self.fc1(x)
        hidden2= self.fc3(hidden)
        relu = self.relu(hidden2)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

training_input=train[predictors]
training_output=train["target"]
test_input=test[predictors]
test_output=test["target"]
#convert to tensors
training_input = torch.FloatTensor(training_input.values)
training_output = torch.FloatTensor(training_output.values)
test_input = torch.FloatTensor(test_input.values)
test_output = torch.FloatTensor(test_output.values)

input_size = training_input.size()[1] # number of features selected
hidden_size = 30 # number of nodes/neurons in the hidden layer
model = Net(input_size, hidden_size) # create the model
criterion = torch.nn.BCELoss() # works for binary classification
# without momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9) 
#with momentum parameter
optimizer = torch.optim.SGD(model.parameters(), lr = 0.9, momentum=0.2)


model.eval()
y_pred = model(test_input)
before_train = criterion(y_pred.squeeze(), test_output.squeeze())
print('Test loss before training' , before_train.item())


model.train()
epochs = 2000
errors = []
for epoch in range(epochs):
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(training_input)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), training_output)
    errors.append(loss.item())
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()


model.eval()
y_pred = model(test_input)
after_train = criterion(y_pred.squeeze(), test_output)
print('Test loss after Training' , after_train.item())

#######################
preds = rf.predict(test[predictors].values.squeeze()) 


import matplotlib.pyplot as plt
import numpy as np
def plotcharts(errors):
    errors = np.array(errors)
    plt.figure(figsize=(12, 5))
    graf02 = plt.subplot(1, 2, 1) # nrows, ncols, index
    graf02.set_title('Errors')
    plt.plot(errors, '-')
    plt.xlabel('Epochs')
    graf03 = plt.subplot(1, 2, 2)
    graf03.set_title('Tests')
    a = plt.plot(test_output.numpy(), 'yo', label='Real')
    plt.setp(a, markersize=10)
    a = plt.plot(y_pred.detach().numpy(), 'b+', label='Predicted')
    plt.setp(a, markersize=10)
    plt.legend(loc=7)
    plt.show()
plotcharts(errors)


#from sklearn.metrics import accuracy_score
#error = accuracy_score(test["target"], preds.squeeze())
#print(error)