import numpy as np

X_train = np.arange(1,101,1) 

print(X_train)

len_data = len(X_train)
test_predictions = []
first_eval_batch = np.array(X_train[int(len_data - 20):])
print(first_eval_batch)
current_batch = np.reshape(first_eval_batch,(1, 20, 1))
print(current_batch)
for i in range(5):
    pred = i
    test_predictions.append(pred)
    current_batch = np.append(current_batch[:,1:,:],[[[pred]]],axis=1)
    print(current_batch)

print(test_predictions)