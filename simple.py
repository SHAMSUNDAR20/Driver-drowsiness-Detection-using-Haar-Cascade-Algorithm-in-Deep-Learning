from keras.models import Sequential
from keras.layers import Conv2D , MaxPooling2D , Flatten , Dense
# creating model for driver drowsiness detection project:
model = Sequential()
 
model.add(Conv2D(64 , (3,3) , activation = 'relu' , input_shape= X.shape[1:]))
model.add(MaxPooling2D((1,1)))
 
model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((1,1)))
 
model.add(Conv2D(64 , (3,3) , activation = 'relu'))
model.add(MaxPooling2D((1,1)))
 
model.add(Flatten())
 
model.add(Dense(128, activation = 'relu'))
 
model.add(Dense(2, activation = 'softmax'))