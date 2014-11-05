#2014-11-04
#Red Neuronal Artificial MLP
#Mineria de Datos
#Universidad Nacional de Colombia Sede Manizales

#John Elkin Rendon Romero
#Mario Fernando Reyes Ojeda
#Santiago Cespedes Zapata
#Alejandra Quinonez

def feedforward(self, entradas):
    #Propaga las entradas desde la capa de entrada hasta la capa de salida
    longVector=len(x)
    x.shape=(longVector,1)
    self.entradas[0]=x
    self.salidas[0]=x
    for i in range (1, self.num_capas):
        self.entradas[i] = self.pesos[i-1].dot(self.salidas[i-1])+self.bias[i-1]
        self.salidas[i] = self.sigmoidea(self.entradas[i])
        
    return self.salidas[-1]
    
def actualizacion_pesos(self, x,y):
    #Actualiza la matriz de pesos para cada capa, basada en una sola entrada x y salida esperada y
    
    #Obtiene la salida de mover las entradas por toda la red
    salida = self.feedforward(x)
    #calcula el error para la salida
    self.errores[-1] = self.sigmoidea_prima(self.salidas[-1]*(salida-y))
    
    #Backprogation del error desde la capa n-1 hasta la capa inicial + 1
    n=self.num_capas-2
    
    for i in xrange(n,0,-1):
        #Calcula el error de la capa a partir de la anterior
        self.errores[i] = self.sigmoidea_prima(self.entradas[i])*self.pesos[i].T.dot(self.errores[i+1])
        #Calcula la nueva matriz de pesos para la capa i  partir del error i+1
        #Np.outer es una funcion de productos de vectores
        self.pesos[i] = self.pesos[i]-self.tasa_aprendizaje*np.outer(self.errores[i+1], self.salidas[i])
        
        #Se ajustan los umbrales de la capa i
        self.bias[i] = self.bias[i] - self.tasa_apredizaje*self.errores[i+1]
        
    #se realiza lo mismo para la primera matriz de pesos
    self.pesos[0] = self.pesos[0] - self.tasa_aprendizaje*np.outer(self.errores[1], self.salidas[0])
    self.bias[0] = self.bias[0] - self.tasa_aprendizaje*self.errores[1]
    
    
    
        
    

