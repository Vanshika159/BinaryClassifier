# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load the data
# data = pd.read_csv('asset_embedding_label.csv')

# # Preprocess the embeddings: remove '[' and ']' and split into individual columns
# data['embedding'] = data['embedding'].str.strip('[]')
# embeddings = data['embedding'].str.split(',', expand=True).astype(float)
# embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]

# # Combine embeddings with original data
# data = pd.concat([data, embeddings], axis=1)

# # Split the data into features (embeddings) and labels
# X = data.drop(columns=['asset_id', 'embedding', 'label'])
# y = data['label']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the classifier (logistic regression)
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = clf.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# Load the data
# data = pd.read_csv('asset_embedding_label.csv')

# # Preprocess the embeddings: remove '[' and ']' and split into individual columns
# data['embedding'] = data['embedding'].str.strip('[]')
# embeddings = data['embedding'].str.split(',', expand=True).astype(float)
# embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]

# # Combine embeddings with original data
# data = pd.concat([data, embeddings], axis=1)

# # Split the data into features (embeddings) and labels
# X = data.drop(columns=['asset_id', 'embedding', 'label'])
# y = data['label']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train the Random Forest classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=42)
# clf.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = clf.predict(X_test)

# # Evaluate the classifier
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load the data
# data = pd.read_csv('finaldata.csv')

# # Preprocess the embeddings: remove '[' and ']' and split into individual columns
# data['embedding'] = data['embedding'].str.strip('[]')
# embeddings = data['embedding'].str.split(',', expand=True).astype(float)
# embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]

# # Combine embeddings with original data
# data = pd.concat([data, embeddings], axis=1)

# # Split the data into features (embeddings) and labels
# X = data.drop(columns=['asset_id', 'embedding', 'label'])
# y = data['label']

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Calculate class weights
# if len(y_train[y_train == "Good"]) == 0:  # Check if there are any instances of the 'Good' class
#     class_weights = {"Bad": 1, "Good": 1}  # Assign equal weights if no instances of 'Good' class
# else:
#     class_weights = {"Bad": 1, "Good": len(y_train[y_train == "Bad"]) / len(y_train[y_train == "Good"])}

# # Initialize and train the classifier (logistic regression) with class weights
# clf = LogisticRegression(class_weight=class_weights)
# clf.fit(X_train, y_train)

# # Make predictions on the testing data
# y_pred = clf.predict(X_test)

# # Evaluate the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print("Classification Report:")
# print(classification_report(y_test, y_pred))

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load the data
# train_data = pd.read_csv('balanced_train_data.csv')
# val_data = pd.read_csv('balanced_val_data.csv')
# test_data = pd.read_csv('balanced_test_data.csv')

# # Function to preprocess embeddings
# def preprocess_embeddings(data):
#     data['embedding'] = data['embedding'].str.strip('[]')
#     embeddings = data['embedding'].str.split(',', expand=True).astype(float)
#     embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
#     data = pd.concat([data, embeddings], axis=1)
#     return data

# # Preprocess the embeddings for each dataset
# train_data = preprocess_embeddings(train_data)
# val_data = preprocess_embeddings(val_data)
# test_data = preprocess_embeddings(test_data)

# # Split the data into features (embeddings) and labels
# X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_train = train_data['label']
# X_val = val_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_val = val_data['label']
# X_test = test_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_test = test_data['label']

# # Calculate class weights for training data
# if len(y_train[y_train == "Good"]) == 0:  # Check if there are any instances of the 'Good' class
#     class_weights = {"Bad": 1, "Good": 1}  # Assign equal weights if no instances of 'Good' class
# else:
#     class_weights = {"Bad": 1, "Good": len(y_train[y_train == "Bad"]) / len(y_train[y_train == "Good"])}

# # Initialize and train the classifier (logistic regression) with class weights
# clf = LogisticRegression(class_weight=class_weights)
# clf.fit(X_train, y_train)

# # Validate the model on the validation data
# y_val_pred = clf.predict(X_val)
# val_accuracy = accuracy_score(y_val, y_val_pred)
# print("Validation Accuracy:", val_accuracy)
# print("Validation Classification Report:")
# print(classification_report(y_val, y_val_pred))

# # Make predictions on the testing data
# y_test_pred = clf.predict(X_test)

# # Evaluate the classifier on test data
# test_accuracy = accuracy_score(y_test, y_test_pred)
# print("Test Accuracy:", test_accuracy)
# print("Test Classification Report:")
# print(classification_report(y_test, y_test_pred))

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load the data
# train_data = pd.read_csv('balanced_train_data.csv')
# val_data = pd.read_csv('balanced_val_data.csv')
# test_data = pd.read_csv('balanced_test_data.csv')

# # Function to preprocess embeddings
# def preprocess_embeddings(data):
#     data['embedding'] = data['embedding'].str.strip('[]')
#     embeddings = data['embedding'].str.split(',', expand=True).astype(float)
#     embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
#     data = pd.concat([data, embeddings], axis=1)
#     return data

# # Preprocess the embeddings for each dataset
# train_data = preprocess_embeddings(train_data)
# val_data = preprocess_embeddings(val_data)
# test_data = preprocess_embeddings(test_data)

# # Initialize and train the initial model
# X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_train = train_data['label']
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Define the maximum number of iterations
# max_iterations = 5

# # Iterate for feedback mechanism
# for i in range(max_iterations):
#     # Validate the current model on the validation data
#     X_val = val_data.drop(columns=['asset_id', 'embedding', 'label'])
#     y_val_pred = clf.predict(X_val)
#     val_accuracy = accuracy_score(val_data['label'], y_val_pred)
#     print(f"Iteration {i+1} - Validation Accuracy: {val_accuracy:.4f}")

#     # Identify misclassified instances
#     misclassified_indices = np.where(y_val_pred != val_data['label'])[0]
#     misclassified_data = val_data.iloc[misclassified_indices]

#     # Augment the training data with misclassified instances
#     train_data = pd.concat([train_data, misclassified_data])

#     # Preprocess the augmented training data
#     X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
#     y_train = train_data['label']

#     # Retrain the model on the augmented training data
#     clf.fit(X_train, y_train)

# # Evaluate the final model on the test data
# X_test = test_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_test_pred = clf.predict(X_test)
# test_accuracy = accuracy_score(test_data['label'], y_test_pred)
# print(f"\nFinal Model - Test Accuracy: {test_accuracy:.4f}")
# print("Test Classification Report:")
# print(classification_report(test_data['label'], y_test_pred))

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.model_selection import train_test_split

# # Load the data
# train_data = pd.read_csv('balanced_train_data.csv')
# val_data = pd.read_csv('balanced_val_data.csv')
# test_data = pd.read_csv('balanced_test_data.csv')

# # Function to preprocess embeddings
# def preprocess_embeddings(data):
#     data['embedding'] = data['embedding'].str.strip('[]')
#     embeddings = data['embedding'].str.split(',', expand=True).astype(float)
#     embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
#     data = pd.concat([data, embeddings], axis=1)
#     return data

# # Preprocess the embeddings for each dataset
# train_data = preprocess_embeddings(train_data)
# val_data = preprocess_embeddings(val_data)
# test_data = preprocess_embeddings(test_data)

# # Split the data into features (embeddings) and labels
# X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_train = train_data['label']
# X_val = val_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_val = val_data['label']
# X_test = test_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_test = test_data['label']

# # Define the neural network architecture
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
#     Dropout(0.5),
#     Dense(32, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val))

# # Evaluate the model on validation data
# val_loss, val_accuracy = model.evaluate(X_val, y_val)
# print("Validation Accuracy:", val_accuracy)

# # Make predictions on test data
# y_pred = model.predict(X_test)
# y_pred_binary = (y_pred > 0.5).astype(int)

# # Evaluate the model on test data
# test_accuracy = accuracy_score(y_test, y_pred_binary)
# print("Test Accuracy:", test_accuracy)
# print("Test Classification Report:")
# print(classification_report(y_test, y_pred_binary))

# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, classification_report
# import numpy as np

# # Load the data
# train_data = pd.read_csv('balanced_train_data.csv')
# val_data = pd.read_csv('balanced_val_data.csv')
# test_data = pd.read_csv('balanced_test_data.csv')

# # Function to preprocess embeddings
# def preprocess_embeddings(data):
#     data['embedding'] = data['embedding'].str.strip('[]')
#     embeddings = data['embedding'].str.split(',', expand=True).astype(float)
#     embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
#     data = pd.concat([data, embeddings], axis=1)
#     return data

# # Preprocess the embeddings for each dataset
# train_data = preprocess_embeddings(train_data)
# val_data = preprocess_embeddings(val_data)
# test_data = preprocess_embeddings(test_data)

# # Initialize and train the initial model
# X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_train = train_data['label']
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Define the maximum number of iterations
# max_iterations = 1

# # Iterate for feedback mechanism
# for i in range(max_iterations):
#     # Validate the current model on the validation data
#     X_val = val_data.drop(columns=['asset_id', 'embedding', 'label'])
#     y_val_pred = clf.predict(X_val)
#     val_accuracy = accuracy_score(val_data['label'], y_val_pred)
#     print(f"Iteration {i+1} - Validation Accuracy: {val_accuracy:.4f}")

#     # Identify misclassified instances
#     misclassified_indices = np.where(y_val_pred != val_data['label'])[0]
#     misclassified_data = val_data.iloc[misclassified_indices]

#     # Augment the training data with misclassified instances
#     train_data = pd.concat([train_data, misclassified_data])

#     # Preprocess the augmented training data
#     X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
#     y_train = train_data['label']

#     # Retrain the model on the augmented training data
#     clf.fit(X_train, y_train)

# # Evaluate the final model on the test data
# X_test = test_data.drop(columns=['asset_id', 'embedding', 'label'])
# y_test_pred = clf.predict(X_test)
# test_accuracy = accuracy_score(test_data['label'], y_test_pred)
# print(f"\nFinal Model - Test Accuracy: {test_accuracy:.4f}")
# print("Test Classification Report:")
# print(classification_report(test_data['label'], y_test_pred))

# # Function to predict labels for new asset IDs and embeddings
# def predict_labels(embedding):
#     # Convert embedding to DataFrame
#     embedding_df = pd.DataFrame([embedding], columns=[f'embedding_{i}' for i in range(len(embedding))])
    
#     # Predict label for the provided embedding
#     prediction = clf.predict(embedding_df)
#     return prediction

# Example usage:
# embeddings = [0.040541068, -0.0307075, 0.021296972, 0.027257841, -0.0076071955, -0.026292356, 0.033102747, 0.037090436, -0.0348272, 0.0037086506, 0.0009882097, 0.0015057799, 0.006430078, 0.01672468, 0.023774125, -0.027146949, 0.0059834896, -0.0026960105, -0.039533164, 0.015627766, 0.015457696, 0.025990654, 0.0017614439, 0.020925572, -0.11193973, 0.012947003, 0.012427733, 0.024546808, 0.027411208, -0.036458757, -0.00012818366, 0.012969292, 0.0041907574, -0.048144367, -0.028232487, 0.04546448, -0.049274214, -0.040828977, -0.06278086, 0.015153377, -0.089371435, -0.0042588185, 0.029809028, 0.016988868, -0.0013940077, -0.03481686, -0.00066582684, -0.012788715, -0.030765876, 0.025724692, -0.00050413754, -0.014192769, 0.011475744, 0.029803861, -0.040116232, -0.005698901, 0.062045597, 0.022733394, 0.026820354, -0.054668684, -0.060214177, 0.049002856, -0.009874348, -0.0056474833, -0.0024306288, 0.0034962764, -0.038205132, 0.012214911, -0.056015693, 0.053354718, 0.04395954, -0.00078120455, 0.010492643, -0.02226719, 0.0029893313, 0.023461957, -0.01875834, -0.008780575, 0.0047884607, 0.033971705, -0.052390296, 0.014725463, -0.005468309, 0.006125121, -0.0010859029, 0.06933578, 0.009142289, -0.04408669, 0.022945827, -0.040220577, -0.063528426, 0.027660426, 0.019915612, -0.05096474, -0.010884878, 0.020769257, -0.059672453, 0.018173667, -0.019294351, 0.040016793, -0.02945161, -0.022582518, -0.02097029, 0.0076910253, 0.015973391, 0.041251674, 0.02719102, -0.029324505, -0.010432747, 0.01082992, -0.054838445, -0.052255493, 0.019729178, -0.052414607, 0.028379837, 0.016325342, -0.020324817, 0.026140928, -0.025122313, -0.025308926, -0.03979958, -0.010006106, 0.028657703, -0.008142901, 0.022516862, -0.038439408, -0.020671863, -0.011208359, 0.010295655, -0.025477398, -0.0021141574, 0.013860323, -0.012059158, -0.0020303552, -0.030374605, -0.04011975, -0.0024270185, -0.0458548, -0.013476967, 0.012120819, 0.013026343, 0.06661551, -0.008435029, -0.020958465, 0.030121293, 0.0074683377, -0.019458063, 0.038848475, 0.007328459, -0.04365263, -0.023754282, 0.03234221, 0.04842854, 0.011151304, -0.014039178, 0.018993396, 0.009493305, 0.02614626, 0.037705693, -0.07390752, -0.0013034676, 0.041453455, -0.015661098, 0.006673428, -0.010435855, 0.008237655, 0.02542318, 0.03771497, 0.005540775, -0.028736113, 0.012959322, -0.026270863, -0.0018464203, -0.039230388, 0.032623608, -0.016840411, 0.005388315, 0.03017295, 0.035711847, -0.0034760393, 0.01853221, 0.020185838, 0.018159831, -0.00971911, 0.015416015, 0.026525797, -0.02879559, -0.003895893, -0.014038448, -0.037866846, -0.02143974, 0.0074405656, -0.004555826, -0.009104697, -0.03660956, 0.0048282905, -0.023765339, -0.0045610564, 0.036875226, -0.020399913, -0.009116361, -0.0002360212, 0.00777838, -0.003668503, -0.03087244, 0.0057480293, 0.018895362, -0.013829789, -0.020827172, -0.0056606806, -0.01750908, -0.025831323, -0.010929922, 0.029631106, -0.019511946, 0.003397325, -0.03960847, 0.0029556854, 0.02283594, -0.03168456, -0.014722791, -0.039387554, -0.025385618, -0.052651145, 0.02958715, -0.014994652, -0.035541356, -0.004883714, -0.03190824, -0.028145637, 0.041095182, -0.04702363, -0.073798105, 0.018775864, 0.00025310184, -0.019253058, 0.0040180706, 0.0042221914, -0.0025631245, 0.024440957, -0.04249075, -0.019702291, -0.054496486, -0.015674748, 0.059055727, 0.025359854, 0.007219338, -0.009817566, -0.0049451995, -0.0009929696, -0.00018703216, 0.008154137, 0.021086296, -0.029156499, -0.034049407, 0.04080505, -0.0065692067, 0.042851932, -0.0330469, -0.012050417, -0.02244109, 0.03807859, 0.016145509, 0.025059974, 0.0014001267, 0.019986397, -0.051263664, 0.02568822, 0.0053579505, -0.03704916, 0.028009836, 0.019836731, 0.0058483556, -0.013710442, 0.012693764, -0.006268356, -0.020468041, -0.02585451, 0.028069435, 0.035364848, 0.016324729, 0.01845448, 0.009339704, 0.019308865, -0.037669804, -0.03322033, 0.032503176, 0.035429105, -0.03403478, 0.013081089, -0.01852951, 0.038807645, 0.015644647, -0.022637732, -0.06570909, -0.02426948, -0.020705385, 0.023321893, -0.0067266114, 0.007190065, 0.014765304, -0.01295661, -0.008442799, 0.027877837, -0.048498023, 0.038301714, -0.026883783, -0.00550287, -0.00672253, 0.00606462, 0.023562802, 0.010807827, 0.0076153963, 0.013295169, -0.013887188, -0.016499482, 0.007485938, 0.023356473, -0.020200241, -0.06525088, 0.023505304, -0.014936843, -0.0063970513, -0.003515449, 0.0040388955, 0.0079249, 0.037746552, 0.009482062, -0.02227448, -0.019726181, 0.012871626, -0.013511082, -0.05066847, -0.0042020436, -0.04052819, 0.038105726, 0.016580224, 0.05514516, 0.02185245, -0.01997804, 0.016999, -0.027659139, 0.0156155, 0.005400006, 0.017822048, 0.039259877, 0.006352428, -0.008928616, -0.0010454537, 0.05280413, -0.030856289, -0.011756829, 0.00467386, 0.015387858, -0.0066053513, -0.015615147, -0.03423979, -0.028388392, 0.019347686, -0.016644068, -0.009567023, 0.0497972, 0.049081653, -0.025075916, 0.027069084, 0.016835999, 0.016391622, -0.051176824, -0.0024348048, 0.013822388, 0.026403787, -0.044807713, -0.0143460715, -0.0081685325, 0.020023206, -0.022939641, -0.009206701, 0.016836276, 0.015155233, -0.026838413, -0.011049171, 0.020503063, 0.027135402, -0.03337939, -0.04097638, -0.0030015917, 0.026332451, 0.02564841, 0.001886909, 0.007926763, -0.001631763, 0.0016642401, -0.029114408, -0.015540845, -0.14459819, -0.03662567, 0.03479204, 0.015414515, 0.04743878, 0.0005080209, 0.027128994, 0.008432209, 0.033101685, 0.017933391, 0.019480748, 0.05888621, -0.03854462, -0.0065071527, 0.033286687, -0.013034145, -0.030970072, -0.014559153, -0.007837386, -0.049360804, 0.025632953, 0.016149512, -0.031217152, -0.015448839, 0.070442826, -0.036455724, 0.045063507, 0.004791387, -0.031530943, 0.038649336, 0.019767594, -0.026407154, -0.0018867857, 0.0029970282, 0.021002807, -0.045563497, 0.010659221, 0.0031009736, -0.0059569194, 0.0143751465, -0.018665174, 0.067521185, 0.005014353, -0.002979957, 0.04379257, 0.012177355, -0.006527739, 0.018487258, 0.01120483, 0.074073985, 0.037663214, -0.016932491, 0.0049755163, -0.0656218, 0.00073459034, -0.011336321, -0.07646063, 0.0076579987, -0.0016691244, 0.036980998, -0.02749976, -0.026484063, -0.019800492, -0.032264795, -0.014660792, 0.02637163, -0.0034170507, 0.0095906295, 0.052726448, 0.019954547, -0.023319725, 0.030315604, 0.0140194455, -0.010260672, 0.0139218215, -0.057512235, 0.011601633, 0.03867123, -0.020654341, 0.037509788, 0.02872701, 0.036066342, -0.04795764, 0.06810025, -0.0037064985, -0.024522949, -0.015616592, -0.026976997, -0.018157935, -0.0012799834, 0.0076537356, 0.0110298395, -0.009035977, 0.0037634368, 0.0295592, -0.0067530223, -0.0042803627, -0.02546707, -0.0039545237, 0.052813824, -0.013281071, -0.014535739, -0.009626291, -0.036529306, 0.006445654, -0.007068734, 0.009394143, -0.020933691, 0.008154856, 0.028310638, 0.11031132, 0.026359018, -0.009394655, -0.017402284, -0.023073519, -0.009727638, 0.015657205, -0.01526567, 0.010517338, 0.06139528, 0.029291121, -0.077318095, 0.004389274, -0.043941855, 0.004049557, 0.025842192, -0.013124194, -0.031816073, -0.03709196, -0.03418146, 0.020486586, 0.024187824, -0.00871832, 0.029928764, -0.023643762, -0.009048348, -0.016328068, -0.0073227044, 0.011539784, -0.036249258, -0.025884945, -0.023982543, 0.062044162, -0.008088005, 0.0025418205, 0.0015331021, 0.01762865, 0.042908385, 0.008615043, 0.021558594, -0.004753198, 0.01747323, -0.025238946, -0.01912941, 0.008553343, 0.01351871, -0.027646333, -0.026930034, 0.00114896, 0.020949826, -0.009558782, -0.018969474, -0.012548589, 0.04145748, 0.0018337811, 0.010628732, 0.0038000464, -0.032328364, -0.04782198, -0.042604346, 0.039088238, 0.002455158, -0.025592204, 0.07075937, -0.04872543, -0.0093910135, -0.014200529, -0.026394228, 0.009857043, 0.062540576, -0.029264878, -0.0025402862, -0.009158349, 0.05205947, 0.05281271, 0.03074877, 0.02582313, 0.0063064373, 0.014375104, -0.02509121, -0.0035944246, 0.00068419275, -0.052610297, -0.040297437, -0.0060330657, 0.06175878, -0.054826014, -0.08851845, 0.020625155, -0.05077129, -0.024601545, 0.008461269, -0.002085185, 0.0358292, 0.015996916, 0.00046112976, 0.0172589, -0.011484156, -0.008980821, 0.016202323, 0.0034945933, 0.0423459, 0.047494706, -0.012562987, -0.040189516, -0.015338991, 0.018034356, 0.014423248, -0.03742089, -0.0038246494, -0.0033597387, -0.004726087, 0.049447738, -0.011022315, 0.06705692, 0.0039879736, 0.0029805829, 0.013227397, -0.018330473, -0.0019073569, -0.028824938, 0.032196984, -0.04362213, -0.031251978, 0.09666608, 0.009985205, -0.023870315, 0.027818963, 0.021941375, 0.019741692, 0.03304609, 0.012869736, -0.051028837, -0.0049289484, 0.016825702, -0.004277516, -0.0035542105, -0.008675856, -0.028031627, 0.019520966, 0.01045282, -0.027211387, -0.06433881, 0.0027900087, -0.01570549, 0.010008658, -0.041666403, 0.03479991, -0.004760391, -0.044242006, 0.0015911164, 0.06475447, -0.009533104, -0.01955513, 0.034905896, -0.010581684, 0.008220829, 0.01138058, 0.045207784, 0.017507251, 0.041442417, 0.019145811, 0.041441474, 0.048268903, 0.019936409, -0.008096279, 0.008367168, 0.014939932, -0.027440514, 0.027161984, -0.050813206, 0.0016542327, -0.013092637, -0.057568867, 0.006930302, 0.034071617, 0.044785406, 0.022545086, -0.015407277, -0.03349838, -0.036507223, 0.008820299, -0.029089311, -0.011004161, -0.014353209, -0.025191441, -0.037634596, 0.0059589567, 0.04495838, -0.0012747527, -0.03847103, 0.013198911, 0.03988596, -0.0003496069, 0.02732047, 0.034021355, -0.011364139, 0.03719319, -0.022535136, 0.01986502, 0.015940698, -0.010066231, -0.022797972, -0.009240996, -0.04048648, 0.02079649, -0.008745447, 0.018112767, 0.031768538, 0.0055946675, -0.0023031437, -0.031409953, 0.027817093, 0.013004918, 0.09234271, -0.00061774784, -0.05903688, 0.03086869, 0.029764397, 0.010652088, 0.0027901235, 0.018904893, 0.0070432853, 0.020402672, -0.033695634, 0.013615456, 0.022816407, 0.004926714, -0.00014298195, 0.0016074027, 0.0037828672, -0.051653422, -0.00034559512, 0.0018663428, -0.009535552, -0.015389249, -0.0036452466, -0.015429427, 0.04201583, 0.013131038, -0.043440953, -0.027295519, 0.021591984, -0.03349263, 0.01961801, 0.069053024, 0.012928159, -0.010176952, 0.05216207, 0.013767246, 0.04131746, -0.021848565, 0.017204015, 0.065213, 0.057773255, -0.055545855, 0.0062373257, -0.00024551168, 0.019675417, -0.026511885, 0.05196268, -0.03620329, -0.015436157, -0.036079135, 0.015646433, -0.024711715, 0.005436034, -0.012579645, -0.046282705, 0.02087446, 0.0421136, -0.06903876, 0.04152242, -0.016564453, 0.014611678, 0.0100594, 0.014735513, 0.022889588, -0.024089068, 0.012893609, 0.05833508, -0.010150419, -0.011785434, -0.021448256, -0.02327026, 0.014090935, 0.009077095, -0.010149023, -0.025291534, -0.010587533, 0.017560182, -0.006405736, 0.036905047, 0.027908096, 0.0033218714, -0.024435395, 0.0016780371, -0.01307186, -0.100567706, -0.004936638, 0.041460894, -0.017135132, 0.012115581, 0.0016290869, 0.009801638, -0.033722464, 0.019877572, 0.12250045, 0.016732395, 0.024282644, 0.032758705, -0.035742555, 0.05020739, 0.005165736, 0.063848406, -0.024329074, -0.017210534, 0.012386832, 0.021127107, 0.04303185, -0.072796084, -0.014249712, 0.025806047, 0.011481433, 0.028670028, 0.01990554, -0.021337735, -0.016437398, -0.0064278175, 0.027434623, -0.0052419123, 0.035606425, 0.028666105, -0.00014907107, -0.013882972, -0.04560662, -0.0051339455, -0.003148985, -0.055730436, 0.012512475, 0.07873251, -0.011414685, -0.012312135, 0.015392275, -0.008986588, 0.027637407, -0.02597243, -0.039984487, 0.0184482, 0.032722015, -0.016408179, -0.014478641, -0.025670776, -0.019277545, 0.041742068, 0.022687335, -0.020260701, 0.06513304, -0.005715408, -0.017828597, 0.026281752, -0.013535554, 0.016598495, 0.0577567, -0.07462042, -0.0044097663, 0.013475843, -0.0058978847, -0.025734995, -0.06434178, 0.0078030666, -0.03965155, 0.016484184, 0.019607091, 0.021741489, 0.011535063, 0.003797268, -0.031188019, 0.0077519948, -0.013043601, 0.008419266, 0.003928617, -0.020023411, 0.0398022, -0.04922301, -0.04127909, 0.021522772, 0.007407185, 0.0100108925, -0.0059655574, 0.045729045, -0.01497908, -0.014697143, 0.034694634, -0.05706752, 0.029154634, 3.932293e-05, 0.023381792, -0.22440283, -0.07820403, -0.033392515, 0.014395354, 0.009430286, -0.020821933, 0.019136263, 0.010067312, 0.01526153, -0.032254748, -0.0123268, -0.037688024, 0.025441444, -0.038361806, -0.0019624357, 0.020508472, -6.808913e-05, -0.05036004, 0.00434819, -0.01747124, 0.0056510433, -0.015658664, 0.054700065, -0.015971126, 0.006570654, 0.050440338, 0.019631987, -0.022260398, -0.0053060637, -0.013242225, 0.030465519, 0.032013148, 0.050747782, 0.0463943, 0.035681397, 0.008802172, -0.0001936733, 0.024560627, 0.005241267, -0.044889614, -0.021833634, -0.03899516, 0.023405494, 0.0077185277, 0.0053615835, -0.041034967, 0.043659654, -0.029816825, -0.052683588, 0.027676346, 0.022679543, 0.009262736, 0.054732125, -0.009829513, 0.004714752, 0.028158396, -0.008351998, 0.005492833, -0.038397092, 0.013108384, -0.039760247, 0.011059095, -0.0021512378, -0.0079855565, -0.015674729, 0.022248589, -0.015300541, 0.05957034, -0.011902441, -0.03395982, -0.02483188, 0.023197718, -0.016269814, -0.004152574, -0.01235996, -0.013486399, -0.0431896, 0.0013995691, -0.010231473, 0.029090093, -0.007880283, -0.034850694, 0.041452464, -0.008120371, 0.06355331, -0.00980615, 0.030240187, -0.0077568456, -0.07394643, 0.023824638, -0.0015416599, -0.0013877234, -0.13735096, 0.02063073, -0.049115703, 0.02577475, 0.061214685, 0.04506453, 0.046971943, -0.06387699, -0.031716447, 0.032276183, -0.010016782, -0.019909613, -0.007986572, 0.026543641, -0.049306903, -0.012217192, -0.03739467, -0.041757435, -0.0041047474, -0.03437556, -0.018045431, -0.02365159, -0.013571658, 0.0065787146, 0.03283875, 0.057003524, 0.04131153, 0.015656212, -0.0028729774, -0.0020131157] # Example embeddings
# predictions = predict_labels(embeddings)
# print(predictions)

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Load the data
train_data = pd.read_csv('balanced_train_data.csv')
val_data = pd.read_csv('balanced_val_data.csv')
test_data = pd.read_csv('balanced_test_data.csv')

# Function to preprocess embeddings
def preprocess_embeddings(data):
    data['embedding'] = data['embedding'].str.strip('[]')
    embeddings = data['embedding'].str.split(',', expand=True).astype(float)
    embeddings.columns = [f'embedding_{i}' for i in range(embeddings.shape[1])]
    data = pd.concat([data, embeddings], axis=1)
    return data

# Preprocess the embeddings for each dataset
train_data = preprocess_embeddings(train_data)
val_data = preprocess_embeddings(val_data)
test_data = preprocess_embeddings(test_data)

# Initialize and train the initial model
X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
y_train = train_data['label']
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Define the maximum number of iterations
max_iterations = 1

# Iterate for feedback mechanism
for i in range(max_iterations):
    # Validate the current model on the validation data
    X_val = val_data.drop(columns=['asset_id', 'embedding', 'label'])
    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(val_data['label'], y_val_pred)
    print(f"Iteration {i+1} - Validation Accuracy: {val_accuracy:.4f}")

    # Identify misclassified instances
    misclassified_indices = np.where(y_val_pred != val_data['label'])[0]
    misclassified_data = val_data.iloc[misclassified_indices]

    # Augment the training data with misclassified instances
    train_data = pd.concat([train_data, misclassified_data])

    # Preprocess the augmented training data
    X_train = train_data.drop(columns=['asset_id', 'embedding', 'label'])
    y_train = train_data['label']

    # Retrain the model on the augmented training data
    clf.fit(X_train, y_train)

# Evaluate the final model on the test data
X_test = test_data.drop(columns=['asset_id', 'embedding', 'label'])
y_test_pred = clf.predict(X_test)
test_accuracy = accuracy_score(test_data['label'], y_test_pred)
print(f"\nFinal Model - Test Accuracy: {test_accuracy:.4f}")
print("Test Classification Report:")
print(classification_report(test_data['label'], y_test_pred))

# Create a DataFrame for predictions
predictions_df = pd.DataFrame({
    'asset_id': test_data['asset_id'],
    'model_prediction': y_test_pred,
    'actual_label': test_data['label']
})

# Save predictions to a text file
predictions_df.to_csv('predictions.txt', index=False, sep='\t')

# Function to predict labels for new asset IDs and embeddings
# def predict_labels(embedding):
#     # Convert embedding to DataFrame
#     embedding_df = pd.DataFrame([embedding], columns=[f'embedding_{i}' for i in range(len(embedding))])
    
#     # Predict label for the provided embedding
#     prediction = clf.predict(embedding_df)
#     return prediction

