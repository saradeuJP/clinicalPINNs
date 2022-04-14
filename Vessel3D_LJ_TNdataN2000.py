"""
@author: Maziar Raissi
"""

import tensorflow as tf
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import scipy.io
import time
import sys
import csv
import pyvista as pv
import PVGeo

from utilities import neural_net, Navier_Stokes_3D, \
                      tf_session, mean_squared_error, relative_error
    
class HFM(object):
    # notational conventions
    # _tf: placeholders for input/output data and points used to regress the equations
    # _pred: output of neural network
    # _eqns: points used to regress the equations
    # _data: input-output data
    # _star: preditions
    
    def __init__(self, t_data, x_data, y_data, z_data,
                       t_eqns, x_eqns, y_eqns, z_eqns,
                       u,v,w,p,
                       layers, batch_size,
                       Pec, Rey):
        
        # specs
        self.layers = layers
        self.batch_size = batch_size
        
        # flow properties
        self.Pec = Pec
        self.Rey = Rey
        
        # data
        [self.t_data, self.x_data, self.y_data, self.z_data] = [t_data, x_data, y_data, z_data]
        [self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns] = [t_eqns, x_eqns, y_eqns, z_eqns]
                
        self.u=u
        self.v=v
        self.w=w
        self.p=p    
        
        self.grad_lossv=[]
        self.grad_loss_pointv=[]
        self.grad_loss_governv=[]
        
        self.dict_grad_loss=self.generate_grad_dict(self.layers) 
        self.dict_grad_loss_point=self.generate_grad_dict(self.layers) 
        self.dict_grad_loss_govern=self.generate_grad_dict(self.layers) 
        
        # placeholders
        [self.u_tf, self.v_tf, self.w_tf, self.p_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_eqns_tf, self.x_eqns_tf, self.y_eqns_tf, self.z_eqns_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
        [self.t_data_tf, self.x_data_tf, self.y_data_tf, self.z_data_tf] = [tf.placeholder(tf.float32, shape=[None, 1]) for _ in range(4)]
    

        self.net_cuvwp = neural_net(self.t_eqns, self.x_eqns, self.y_eqns, self.z_eqns, layers = self.layers)
        
        self.weights=self.net_cuvwp.weights
         
        # physics "informed" neural networks
        [self.u_data_pred,
         self.v_data_pred,
         self.w_data_pred,
         self.p_data_pred] = self.net_cuvwp(self.t_data_tf,
                                            self.x_data_tf,
                                            self.y_data_tf,
                                            self.z_data_tf)    
        
        [self.u_eqns_pred,
         self.v_eqns_pred,
         self.w_eqns_pred,
         self.p_eqns_pred] = self.net_cuvwp(self.t_eqns_tf,
                                            self.x_eqns_tf,
                                            self.y_eqns_tf,
                                            self.z_eqns_tf)
        
        [self.e1_eqns_pred,
         self.e2_eqns_pred,
         self.e3_eqns_pred,
         self.e4_eqns_pred] = Navier_Stokes_3D(self.u_eqns_pred,
                                               self.v_eqns_pred,
                                               self.w_eqns_pred,
                                               self.p_eqns_pred,
                                               self.t_eqns_tf,
                                               self.x_eqns_tf,
                                               self.y_eqns_tf,
                                               self.z_eqns_tf,
                                               self.Pec,
                                               self.Rey)
        
        self.e1 = mean_squared_error(self.e1_eqns_pred, 0.0)
        self.e2 = mean_squared_error(self.e2_eqns_pred, 0.0)
        self.e3 = mean_squared_error(self.e3_eqns_pred, 0.0)
        self.e4 = mean_squared_error(self.e4_eqns_pred, 0.0)
        self.eu = mean_squared_error(self.u_data_pred, self.u_tf)
        self.ev = mean_squared_error(self.v_data_pred, self.v_tf)
        self.ew = mean_squared_error(self.w_data_pred, self.w_tf)
        self.ep = mean_squared_error(self.p_data_pred, self.p_tf)
        
        # loss
        self.loss = self.e1 + self.e2 + self.e3 + self.e4 + \
                    self.eu + self.ev + self.ew #+ self.ep
        
        # loss grad
        for i in range(len(self.layers)-1):
            self.grad_lossv.append(tf.gradients(self.loss,self.weights[i])[0])
            self.grad_loss_pointv.append(tf.gradients((self.eu + self.ev + self.ew),self.weights[i])[0])
            self.grad_loss_governv.append(tf.gradients((self.e1 + self.e2 + self.e3 + self.e4),self.weights[i])[0])
        
        # optimizers
        self.global_step = tf.Variable(0, trainable=False)
        self.starter_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.learning_rate = tf.train.exponential_decay(self.starter_learning_rate, self.global_step,
                                                        1000, 0.9, staircase=False)
        #self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        
        self.optimizer2 = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                               method = 'L-BFGS-B', 
                                                               options = {'maxiter': 50000,
                                                                          'maxfun': 50000,
                                                                          'maxcor': 50,
                                                                          'maxls': 50,
                                                                          'ftol' : 1.0 * np.finfo(float).eps})
        self.sess = tf_session()
    
    def callback(self, loss, loss_eqns, loss_data):
        #print('Loss: %.3e'% (loss))
        loss_valuev=np.asarray(loss)
        loss_value_pointv=np.asarray(loss_data)
        loss_value_governv=np.asarray(loss_eqns)
        
        loss_valuev=loss_valuev.reshape(1,-1)
        loss_value_pointv=loss_value_pointv.reshape(1,-1)
        loss_value_governv=loss_value_governv.reshape(1,-1)
        
        with open('loss_value.csv','ab') as f:               
          np.savetxt(f,loss_valuev)
        
        with open('loss_value_point_N_batch1000.csv','ab') as f:               
          np.savetxt(f,loss_value_pointv)  
          
        with open('loss_value_govern_N_batch1000.csv','ab') as f:               
          np.savetxt(f,loss_value_governv)

    
    def generate_grad_dict(self, layers):
        num = len(layers) - 1
        grad_dict = {}
        for i in range(num):
            grad_dict['layer_{}'.format(i + 1)] = []
        return grad_dict
    
    def train(self, total_time, learning_rate):

        N_eqns = self.t_eqns.shape[0]
        N_data = self.t_data.shape[0]
        
        start_time = time.time()
        running_time = 0
        it = 0
        
        #while running_time < total_time:
        while it < 25000:
            
            idx_data = np.random.choice(N_data, self.batch_size)
            idx_eqns = np.random.choice(N_eqns, self.batch_size)
            
            (t_data_batch,
             x_data_batch,
             y_data_batch,
             z_data_batch) = (self.t_data[idx_data,:],
                              self.x_data[idx_data,:],
                              self.y_data[idx_data,:],
                              self.z_data[idx_data,:])

            (t_eqns_batch,
             x_eqns_batch,
             y_eqns_batch,
             z_eqns_batch) = (self.t_eqns[idx_eqns,:],
                              self.x_eqns[idx_eqns,:],
                              self.y_eqns[idx_eqns,:],
                              self.z_eqns[idx_eqns,:])

            (u_batch,
             v_batch,
             w_batch,
             p_batch) = (self.u[idx_data,:],
                         self.v[idx_data,:],
                         self.w[idx_data,:],
                         self.p[idx_data,:])

            tf_dict = {self.t_data_tf: t_data_batch,
                       self.x_data_tf: x_data_batch,
                       self.y_data_tf: y_data_batch,
                       self.z_data_tf: z_data_batch,
                       self.t_eqns_tf: t_eqns_batch,
                       self.x_eqns_tf: x_eqns_batch,
                       self.y_eqns_tf: y_eqns_batch,
                       self.z_eqns_tf: z_eqns_batch,
                       self.u_tf: u_batch,
                       self.v_tf: v_batch,
                       self.w_tf: w_batch,
                       self.p_tf: p_batch,
                       self.starter_learning_rate: learning_rate}
            
            self.sess.run([self.train_op], tf_dict)
            
            # Print
        
            if it % 100 == 0:
                elapsed = time.time() - start_time
                running_time += elapsed/3600.0

                [loss_value,
     learning_rate_value,
     e1, e2, e3, e4, eu, ev, ew, ep] = self.sess.run([self.loss,
                                           self.learning_rate, 
                                           self.e1, self.e2, self.e3, self.e4,
                                           self.eu, self.ev, self.ew, self.ep], tf_dict)
                
                print('It: %d, Loss: %.3e, Time: %.2fs, Running Time: %.2fh, Learning Rate: %.1e'
                      %(it, loss_value, elapsed, running_time, learning_rate_value))
                
                print('e1: %.3e' %(e1))
                print('e2: %.3e' %(e2))
                print('e3: %.3e' %(e3))
                print('e4: %.3e' %(e4))  
                print('eu: %.3e' %(eu))   
                print('ev: %.3e' %(ev))  
                print('ew: %.3e' %(ew))  
                print('ep: %.3e' %(ep))    
                
                loss_valuev=np.asarray(loss_value)
                loss_value_pointv=np.asarray(eu + ev + ew)
                loss_value_governv=np.asarray(e1 + e2 + e3 + e4)
                learning_rate_value=np.asarray(learning_rate_value)
                
                loss_valuev=loss_valuev.reshape(1,-1)
                loss_value_pointv=loss_value_pointv.reshape(1,-1)
                loss_value_governv=loss_value_governv.reshape(1,-1)
                learning_rate_value=learning_rate_value.reshape(1,-1)
                
                with open('loss_value.csv','ab') as f:               
                  np.savetxt(f,loss_valuev)
                
                with open('loss_value_point_N_batch1000.csv','ab') as f:               
                  np.savetxt(f,loss_value_pointv)  
                  
                with open('loss_value_govern_N_batch1000.csv','ab') as f:               
                  np.savetxt(f,loss_value_governv)
                  
                with open('lr.csv','ab') as f:               
                  np.savetxt(f,learning_rate_value)
    
                sys.stdout.flush()
                start_time = time.time() 
                
                #if it % 500 == 0:
                #  for i in range(len(self.layers)-1):
                #    grad_loss=self.sess.run(self.grad_lossv[i],tf_dict)
                #    grad_loss_point=self.sess.run(self.grad_loss_pointv[i],tf_dict)
                #    grad_loss_govern=self.sess.run(self.grad_loss_governv[i],tf_dict)

                  
                #    self.dict_grad_loss['layer_'+str(i+1)].append(grad_loss.flatten())
                #    self.dict_grad_loss_point['layer_'+str(i+1)].append(grad_loss_point.flatten())
                #    self.dict_grad_loss_govern['layer_'+str(i+1)].append(grad_loss_govern.flatten())
                  
                #  scipy.io.savemat('loss_grad_it%s.mat' %(it), self.dict_grad_loss) 
                #  scipy.io.savemat('loss_grad_point_it%s.mat' %(it),self.dict_grad_loss_point)
                #  scipy.io.savemat('loss_grad_govern_it%s.mat' %(it),self.dict_grad_loss_govern) 
                

            it += 1
      
        
        self.optimizer2.minimize(self.sess,
                               feed_dict = tf_dict,
                               fetches = [self.loss, (self.e1 + self.e2 + self.e3 + self.e4), (self.eu + self.ev + self.ew)],
                               loss_callback = self.callback)   

    
    def predict(self, t_star, x_star, y_star, z_star):
        
        tf_dict = {self.t_data_tf: t_star, self.x_data_tf: x_star, self.y_data_tf: y_star, self.z_data_tf: z_star}
        
        u_star = self.sess.run(self.u_data_pred, tf_dict)
        v_star = self.sess.run(self.v_data_pred, tf_dict)
        w_star = self.sess.run(self.w_data_pred, tf_dict)
        p_star = self.sess.run(self.p_data_pred, tf_dict)
        
        return u_star, v_star, w_star, p_star
    
if __name__ == "__main__": 
      
    batch_size = 2000
    
    layers = [4] + 20*[5*50] + [4]
    
    # Load Data
    
    import csv
    import numpy as np
    import glob

    with open('../../Data/time.csv',"rt", encoding="UTF-8-sig") as f:
      reader = csv.reader(f)
      t=[]
      for r in reader:
        #print(r)
        t.append(float(r[0]))
    t_star=np.array(t)
    

    file_dir='../../Data/vesselLJ_sim_laminar_2BC'

    x_star = []
    y_star = [] # N x 1
    z_star = [] # N x 1

    U_star = [] # N x T
    V_star = [] # N x T
    W_star = [] # N x T
    P_star = []

    for j in range(0,1625,5):
      csv_file = f'{file_dir}/exportstep_{j}.csv'
      #print(csv_file)
      x=[]
      y=[]
      z=[]
      u=[]
      v=[]
      w=[]
      p=[]
      
        
      with open(csv_file) as f:
        reader = csv.reader(f)
        i=0
        for r in reader:
          i=i+1
          if i >=7:

            x.append(float(r[0]))
            y.append(float(r[1]))
            z.append(float(r[2]))
              
            u.append(float(r[3]))
            v.append(float(r[4]))
            w.append(float(r[5]))
            p.append(float(r[6]))
  
        x_star=np.concatenate([x_star,x],0)
        y_star=np.concatenate([y_star,y],0)
        z_star=np.concatenate([z_star,z],0)
        U_star=np.concatenate([U_star,u],0)
        V_star=np.concatenate([V_star,v],0)
        W_star=np.concatenate([W_star,w],0)
        P_star=np.concatenate([P_star,p],0)

         
    x_star=np.array(x_star)
    y_star=np.array(y_star)
    z_star=np.array(z_star)
    
    U_star=np.array(U_star)
    V_star=np.array(V_star)
    W_star=np.array(W_star)
    P_star=np.array(P_star)
   
  
    #t_star=t_star*0.125723/0.005
    #x_star=x_star/1000/0.005
    #y_star=y_star/1000/0.005
    #z_star=z_star/1000/0.005
    #U_star=U_star/0.125723
    #V_star=V_star/0.125723
    #W_star=W_star/0.125723
    #P_star=P_star/1100/0.125723/0.125723 
    Ntot = len(U_star)
    
    x_star=x_star.reshape(len(t_star),int(Ntot/len(t_star)))
    y_star=y_star.reshape(len(t_star),int(Ntot/len(t_star)))
    z_star=z_star.reshape(len(t_star),int(Ntot/len(t_star)))
    
    U_star=U_star.reshape(len(t_star),int(Ntot/len(t_star)))
    V_star=V_star.reshape(len(t_star),int(Ntot/len(t_star)))
    W_star=W_star.reshape(len(t_star),int(Ntot/len(t_star)))
    P_star=P_star.reshape(len(t_star),int(Ntot/len(t_star)))

    X_star=x_star.T
    Y_star=y_star.T
    Z_star=z_star.T
    
    U_star=U_star.T # N x T
    V_star=V_star.T
    W_star=W_star.T
    P_star=P_star.T
    
    T = t_star.shape[0]
    N = U_star.shape[0]

    
    # Rearrange Data 
    t_star = t_star.reshape(T,1)
    T_star = np.tile(t_star, (1,N)).T # N x T
        
    print(T)
    print(N)
    

    ######################################################################
######################## Noiseles Data###############################
######################################################################

    
    T_eqns = T
    N_eqns = N
        
    idx_t = np.concatenate([np.array([0]), np.random.choice(T-2, T_eqns-2, replace=False)+1, np.array([T-1])] )
    idx_x = np.random.choice(N, N_eqns, replace=False)
    t_eqns = T_star[:, idx_t][idx_x,:].flatten()[:,None]
    x_eqns = X_star[:, idx_t][idx_x,:].flatten()[:,None]
    y_eqns = Y_star[:, idx_t][idx_x,:].flatten()[:,None]
    z_eqns = Z_star[:, idx_t][idx_x,:].flatten()[:,None]

    #T_data = round(T/2)
    N_data = 2000
    
    
    t_data = []
    x_data = []
    y_data = []
    z_data = []
    u_data = []
    v_data = []
    w_data = []
    p_data = []
    

    X1 = T_star.T # T x N
    X2 = X_star.T
    X3 = Y_star.T
    X4 = Z_star.T
    X5 = U_star.T
    X6 = V_star.T
    X7 = W_star.T
    X8 = P_star.T
    
    X1 = X1[:,:,np.newaxis]
    X2 = X2[:,:,np.newaxis]
    X3 = X3[:,:,np.newaxis]
    X4 = X4[:,:,np.newaxis]
    X5 = X5[:,:,np.newaxis]
    X6 = X6[:,:,np.newaxis]
    X7 = X7[:,:,np.newaxis]
    X8 = X8[:,:,np.newaxis]
    
    X = np.concatenate((X1,X2,X3,X4,X5,X6,X7,X8), axis=2) # T x N x 8
    
    X = X.transpose (1,2,0) # N x 8 x T
    np.random.shuffle(X)
    X_data = X[:N_data,:,:] # N x 8 x T
    print(X)
        
    #X_data = X_data.transpose (2,1,0) # T x 8 x N
    #np.random.shuffle(X_data)
    #X_data = X_data[:T_data,:,:]
    #X_eqns = X[T_data:,:,:]
    
    #X_data = X_data.transpose (2,1,0) # T x 8 x N
    
    print(X_data.shape) 
    print(x_eqns.shape)
  
    X_data = np.array(X_data)

    t_data = X_data[:,0,:].flatten()[:,None]
    x_data = X_data[:,1,:].flatten()[:,None]
    y_data = X_data[:,2,:].flatten()[:,None]
    z_data = X_data[:,3,:].flatten()[:,None]
    u_data = X_data[:,4,:].flatten()[:,None]
    v_data = X_data[:,5,:].flatten()[:,None]
    w_data = X_data[:,6,:].flatten()[:,None]
    p_data = X_data[:,7,:].flatten()[:,None]

            
    # Training
    model = HFM(t_data, x_data, y_data, z_data,
                t_eqns, x_eqns, y_eqns, z_eqns,
                u_data, v_data, w_data, p_data,
                layers, batch_size,
                Pec = 1.0/0.0036, Rey = 1.0/0.0036)
    
    model.train(total_time = 70, learning_rate=1e-3)
    
    # Test Data
            
    ################# Save Data ###########################
 
    
    U_pred = 0*U_star
    V_pred = 0*V_star
    W_pred = 0*W_star
    P_pred = 0*P_star
    
    for snap in range(0,t_star.shape[0]):
        t_test = T_star[:,snap:snap+1]
        x_test = X_star[:,snap:snap+1]
        y_test = Y_star[:,snap:snap+1]
        z_test = Z_star[:,snap:snap+1]
        
        
        u_test = U_star[:,snap:snap+1]
        v_test = V_star[:,snap:snap+1]
        w_test = W_star[:,snap:snap+1]
        p_test = P_star[:,snap:snap+1]
        
        
        # Prediction
        u_pred, v_pred, w_pred, p_pred = model.predict(t_test, x_test, y_test, z_test)
        
        
        U_pred[:,snap:snap+1] = u_pred
        V_pred[:,snap:snap+1] = v_pred
        W_pred[:,snap:snap+1] = w_pred
        P_pred[:,snap:snap+1] = p_pred
    
        # Error
        
        error_u = relative_error(u_pred, u_test)
        error_v = relative_error(v_pred, v_test)
        error_w = relative_error(w_pred, w_test)
        error_p = relative_error(p_pred - np.mean(p_pred), p_test - np.mean(p_test))
    
        
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error w: %e' % (error_w))
        #print('Error p: %e' % (error_p))
        
    scipy.io.savemat('./Vessel3D_LJ_T_N25_no_noise_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'U_pred':U_pred, 'V_pred':V_pred, 'W_pred':W_pred, 'P_pred':P_pred})
    
    U_star=U_star.flatten()
    V_star=V_star.flatten()
    W_star=W_star.flatten()
    P_star=P_star.flatten()

    U_pred=U_pred.flatten()
    V_pred=V_pred.flatten()
    W_pred=W_pred.flatten()
    P_pred=P_pred.flatten()

    N = U_star.shape[0]

    MSEu=np.sqrt(np.sum(np.square(U_pred - U_star))/N)/np.maximum(U_star.max(),(-1*U_star).max())
    MSEv=np.sqrt(np.sum(np.square(V_pred - V_star))/N)/np.maximum(V_star.max(),(-1*V_star).max())
    MSEw=np.sqrt(np.sum(np.square(W_pred - W_star))/N)/np.maximum(W_star.max(),(-1*W_star).max())
    MSEp=np.sqrt(np.sum(np.square(P_pred - P_star))/N)/np.maximum(P_star.max(),(-1*P_star).max())

    print(MSEu)
    print(MSEv)
    print(MSEw)
    print(MSEp)
