import numpy as np
import pandas

def sigmoid(object):
    return 1/(1+np.exp(object))

def costFunction(Y,target):

    
    
    J_theta=(1/2)*np.sum(-target)
    return J_theta

def trainNeuralNetwork(dataset , labmda):

    Y=dataset["label"]
    column_name=dataset.ix[:,"pixel0":]
    column_name=column_name.values
    #column_name=np.reshape(column_name,(42000,784))



    index=2

    theta1=np.random.rand(785,25)
    theta2=np.random.rand(26,10)



    while(1):

        bias=1
        #  print x_train.describe()
        x_train= column_name[index]
        #x_train=x_train.as_matrix()
        #print type(x_train)
        x_train=np.append(bias,x_train)
        x_train=np.reshape(x_train,(785,1))

        x_train=np.transpose(x_train)


    
    
        

        #print x_train.shape()
        #print x_train.shape

        #print x_train

        a1=x_train
        #print x_train
        a2 = np.dot(x_train,theta1)
        z2=sigmoid(a2)
        #print z2.shape

        z2=np.append(bias,z2)

        a3=np.dot(z2,theta2)
        z3=sigmoid(a3)

        cost=costFunction(Y[index],z3)
        print cost
        

        
        break
        index+=1





dataset = pandas.read_csv("train.csv")

#print dataset.describe()

trainNeuralNetwork(dataset,2)



