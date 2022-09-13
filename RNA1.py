import mnist_loader
import Network
import pickle
training_data, validation_data , test_data = mnist_loader.load_data_wrapper()
training_data = list(training_data)
test_data = list(test_data)
net=Network.Network([784,30,10], cost=Network.CrossEntropyCost)
net.SGD( training_data[:1000], 71, 10, 3.5, evaluation_data=test_data, lmbda=0.1, monitor_evaluation_cost=True, monitor_evaluation_accuracy=True, monitor_training_cost=True, monitor_training_accuracy=True)
archivo = open("red_prueba1.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
#leer el archivo
archivo_lectura = open("red_prueba.pkl",'rb')
net = pickle.load(archivo_lectura)
archivo_lectura.close()
net.SGD( training_data, 71, 10, 3.5, test_data=test_data)
archivo = open("red_prueba.pkl",'wb')
pickle.dump(net,archivo)
archivo.close()
exit()
