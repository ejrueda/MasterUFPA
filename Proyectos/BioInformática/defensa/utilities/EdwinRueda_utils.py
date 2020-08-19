### for more documentation, you can read the "creating utils" notebook.
#data: 30/07/2020
#author: Edwin Jahir Rueda Rojas
#page: https://github.com/ejrueda
#email: ejrueda95g@gmail.com
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
tf.keras.backend.set_floatx('float64')
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split


#----------------------------------------------------------
#-------------- Scaler class ------------------------------
#----------------------------------------------------------
class scaler:
    """
    Scaler class allows scaler a dataframe without losing the dataframe index
    """
    def __init__(self, xmin, xmax):
        """
        minmax scaler from dataframe
        """
        self.xmin = xmin
        self.xmax = xmax
        self.min_data = False
        self.max_data = False
        self.flag = False
        
    def fit(self, X):
        self.min_data = np.min(X).values
        self.max_data = np.max(X).values
        self.flag = True
        
    def transform(self, X):
        assert self.flag, "Erro de treinamento, primeiro tem que treinar o Scaler, called .fit()"
        X_r = X.copy()
        X_r = ((X_r - self.min_data)/(self.max_data - self.min_data))*(self.xmax-self.xmin) + self.xmin
        return X_r
    
    def inverse_transform(self, X):
        assert self.flag, "Erro de treinamento, primeiro tem que treinar o Scaler, called .fit()"
        X_r = X.copy()
        X_r = ((X_r - self.xmin)*(self.max_data - self.min_data)/(self.xmax - self.xmin)) + self.min_data
        return X_r
    
#----------------------------------------------------------
#-------------- gan_utils class ---------------------------
#----------------------------------------------------------
class gan_utils:
    """
    gan_utils allows train a Generative Adversarial Network and shows its result
    Methods:
        - kl_divergence
        - binary_cross_entropy
        - train_step
        - train
        - plot results
    """
    def __init__(self):
        self.accumulated_gloss = []
        self.accumulated_dloss = []
        self.precision = []
        self.recall = []
        self.kl_d = []
        self.X_train = None
        self.G = None
        self.D = None
        self.noise_input = None
    
    @tf.function
    def kl_divergence(self, probability):
        """
        this function computes the kullback-leibler divergence from the probability
        of a discriminator network
        """
        probability = tf.clip_by_value(probability, 1e-5, 1-1e-5)
        return tf.reduce_mean(probability*tf.math.log(probability/(1-probability)))
    
    @tf.function
    def binary_cross_entropy(self, prediction, target):
        """
        compute the loss for binary clasification problems
        inputs:
            prediction: predicted class
            target: target class
        """
        prediction = tf.clip_by_value(prediction, 1e-5, 1-1e-5)
        return -tf.reduce_mean(target*tf.math.log(prediction) + (1-target)*tf.math.log(1-prediction))

    @tf.function
    def train_step(self, sample, batch_size, noise_input, optimizerG, optimizerD):
        """
        this function train a GAN architecture from a batch
        inputs:
            sample: batch from a tensorflow dataset
            batch_size: size of the batch
            noise_input: size of the noise vector to train de generator network
            optimizerG: an optimizer of tensorflow, this optimizer is used to update the gradients
                        of the Generator network.
            optimizerD: an optimizer of tensorflow, this optimizer is used to update the gradients
                        of the Discriminator network.
        """
        noise = tf.random.normal([batch_size, noise_input])
        with tf.GradientTape() as gG, tf.GradientTape() as gD:
            synthetic_data = self.G(noise, training=True)

            real_output = self.D(sample, training=True)
            fake_output = self.D(synthetic_data, training=True)

            gen_loss = self.binary_cross_entropy(fake_output, tf.ones_like(fake_output))
            dis_loss = .5*(self.binary_cross_entropy(real_output, tf.ones_like(real_output)) + self.binary_cross_entropy(fake_output,tf.zeros_like(fake_output)))

        g_generator = gG.gradient(gen_loss, self.G.trainable_variables)
        g_discriminator = gD.gradient(dis_loss, self.D.trainable_variables)

        optimizerG.apply_gradients(zip(g_generator, self.G.trainable_variables))
        optimizerD.apply_gradients(zip(g_discriminator, self.D.trainable_variables))

        return gen_loss, dis_loss
    
    def train(self, dataset, G, D, noise_input, epochs, batch_size, optimizerG, optimizerD, verbose=True):
        """
        This function train a GAN architecture.
        inputs:
            dataset: pandas dataframe to train the architecture.
            G: a generator network to build the architecture
            D: a discriminator network to build the architecture.
            noise_input: size of the noise vector.
            epochs: number of epochs to train the architecture.
            batch_size: size of the batch to train the architecture in each epoch.
            optimizerG: an optimizer of tensorflow, this optimizer is used to update the gradients
                        of the Generator network.
            optimizerD: an optimizer of tensorflow, this optimizer is used to update the gradients
                        of the Discriminator network.
            verbose: Bolean, default True. parameter to show the convergence process of the architecture
        Return:
            accumulated_gloss: list of the loss in each epoch of the generator network
            accumulated_dloss: list of the loss in each epoch of the discriminator network
        """
        #reset metrics
        self.accumulated_gloss = []
        self.accumulated_dloss = []
        self.precision = []
        self.recall = []
        self.kld = []
        batch_g_loss = []
        batch_d_loss = []
        self.G, self.D = G, D
        self.noise_input = noise_input
        self.X_train = dataset
        batches = tf.data.Dataset.from_tensor_slices(dataset).batch(batch_size)
        for epoch in range(epochs):
            t_i = time()
            for batch in batches:
                batch_g_loss, batch_d_loss = self.train_step(batch, batch_size, noise_input,
                                                             optimizerG, optimizerD)
            self.accumulated_dloss.append(tf.reduce_mean(batch_d_loss))
            self.accumulated_gloss.append(tf.reduce_mean(batch_g_loss))

            X_sint = G(tf.random_normal_initializer(mean=0.0, stddev=1)([batch_size,noise_input],
                                                                        dtype=tf.float64))
            X_comb = tf.concat([dataset, X_sint], axis=0)
            y_comb = tf.concat([tf.ones((dataset.shape[0],1)), np.zeros((X_sint.shape[0],1))], axis=0)
            y_proba = D(X_comb)
            y_predict = tf.reshape((tf.greater(y_proba, .5).numpy()*1), [-1])
            self.kl_d.append(self.kl_divergence(y_proba).numpy())
            self.precision.append(precision_score(y_comb.numpy(), y_predict.numpy()))
            self.recall.append(recall_score(y_comb.numpy(), y_predict.numpy()))
            t_f = time()
            if verbose:
                print("epochs[%d:%d] :: G_loss[%f] :: D_loss[%f] :: time:%f[s]"%(epoch, epochs,
                                                                                 self.accumulated_gloss[-1],
                                                                                 self.accumulated_dloss[-1],
                                                                                 t_f-t_i))
        return self.accumulated_gloss, self.accumulated_dloss
        
    def plot_results(self, syn_size):
        """
        this function shows a figure with the principal metrics to see the convergence process
        of the architecture
        input:
            syn_size: number of synthetic genes to be generated
        return:
            fig: a matplotlib figure with the main metrics to see the convergence process
                 of the architecture
        """
        fig = plt.figure(figsize=(15,10))
        for i in range(4):
            plt.subplot(2,2, i+1)
            if i==0:
                rr = range(0,len(self.accumulated_gloss), 3)
                short_g_loss = [self.accumulated_gloss[i] for i in rr]
                short_d_loss = [self.accumulated_dloss[i] for i in rr]
                plt.plot(rr, short_g_loss, label="Generator", color="#FFB248")
                plt.plot(rr, short_d_loss, label="Discriminator", color="#5F84B4")
                plt.xlabel("Epochs", fontsize=13)
                plt.ylabel("Loss", fontsize=13)
                plt.grid(axis="y")
                plt.legend();
            elif i==1:
                plt.title("Divergencia de Kullback-Leibler", fontsize=14)
                plt.plot(range(len(self.kl_d)), self.kl_d, linewidth=.5)
                plt.plot(range(len(self.kl_d)), np.zeros(len(self.kl_d)))
                plt.grid()
                plt.xlabel("Epochs");
                plt.ylabel("Divergencia de Kullback-Leibler");
            elif i==2:
                short_precision = [self.precision[v] for v in rr]
                plt.plot(rr, short_precision, linewidth=.8, label="precision")
                plt.grid(axis="y")
                plt.plot(rr, np.ones(len(short_precision))*.5,
                         label="target", linestyle="--")
                plt.legend();
            elif i==3:
                pca = PCA(n_components=2)
                X_real_pca = pca.fit_transform(self.X_train)
                noise = tf.random.normal([syn_size, self.noise_input])
                synthetics = self.G(noise)
                X_fake_pca = pca.transform(synthetics.numpy())
                plt.scatter(X_fake_pca[:,0], X_fake_pca[:,1], label="synthetic", alpha=.4, color="orange",
                           edgecolors="red");
                plt.scatter(X_real_pca[:,0], X_real_pca[:,1], label="real", marker="*", s=80, color="green",
                            edgecolors="black");
        return fig
    
    def get_metrics(self, num_iter):
        """
        this function returns the metrics obtained after
        training the architecture
        num_iter: number of iterations to compute the boxplot
        return: precision of the discriminator network, Kullback-Leibler divergence,
                loss of the generator network, loss of the discriminator network
        """
        precision_d = []
        kld_divergence = []
        g_loss = []
        d_loss = []
        for i in range(num_iter):
            noise = tf.random.normal([self.X_train.shape[0], self.noise_input])
            synthetic_samples = self.G(noise)
            X_comb = tf.concat([self.X_train, synthetic_samples], axis=0)
            y_comb = tf.concat([tf.ones((self.X_train.shape[0],1), dtype=tf.float64),
                                tf.zeros((synthetic_samples.shape[0],1), dtype=tf.float64)], axis=0)
            y_proba = self.D(X_comb)
            y_predict = tf.reshape((tf.greater(y_proba, .5).numpy()*1), [-1])
            precision_d.append(precision_score(y_comb.numpy(), y_predict))
            kld_divergence.append(self.kl_divergence(y_proba).numpy())
            g_loss.append(self.binary_cross_entropy(self.D(synthetic_samples),
                                                    tf.zeros(synthetic_samples.shape[0], dtype=tf.float64)).numpy())
            d_loss.append(self.binary_cross_entropy(y_proba, y_comb).numpy())
            
        return precision_d, kld_divergence, g_loss, d_loss
    
#----------------------------------------------------------
#-------------- bokeh_utils class -------------------------
#----------------------------------------------------------
class bokeh_utils:
    """
    this class contains functions to facilitate the use of the bokeh library
    methods:
        boxplot_values
    """
    def __init__(self):
        pass
    
    def boxtplot_values(self, v, min_value= False, max_value=False):  
        """
        This function returns the values to plot a boxplot in the bokeh library
        parameters: 
            v: array with the values to make a boxplot
            min_value: Bolean, default False. if the min_value is True, the min value of the boxplot
            would be the min value of the v.
            max_value: Bolean, default False. if the max_value is True, the max value of the boxplot
            would be the max value of the v.
        return: [lower, quantile25, quantile50, quantile75, upper] and outliers
        """
        q25 = np.quantile(v, q=.25, interpolation="midpoint")
        q50 = np.quantile(v, q=.5, interpolation="midpoint")
        q75 = np.quantile(v, q=.75, interpolation="midpoint")
        lower = q25 - 1.5*(q75-q25)
        upper = q75 + 1.5*(q75-q25)
        #outliers
        outliers = v[(v<lower)|(v>upper)]
        if min_value == True:
            if lower < min(v):
                lower = min(v)
        if max_value == True:
            if upper > max(v):
                upper = max(v)
        return [lower, q25, q50, q75, upper], outliers
    
#----------------------------------------------------------
#-------------------- SMOTE class -------------------------
#----------------------------------------------------------

class smote:
    """
    SMOTE: Synthetic Minority Over-sampling Technique
        Methods:
            - get_syn_samples
    """
    def __init__(self):
        self.neigh = None
    
    def get_syn_samples(self, T, N, k):
        """
        get synthetic samples with the SMOTE algorithm.
        Inputs:
            T: 2D-data array to be increase
            N: Amount of SMOTE N%.
            k: Number of nearest neighbors
        Ouput:
            (N/100)*T synthetic minority class samples
            (∗ If N is less than 100%, randomize the minority class samples as only a random
               percent of them will be SMOTEd. ∗)
        """
        if N < 100:
            idxs = np.random.choice(range(len(T)), size=int(len(T)*(N/100)), replace=False)
            T = T[idxs, :]
            N = 100
        
        N = int(N/100)*len(T)
        synthetic = np.ones((N, T.shape[1]))
        self.neigh = NearestNeighbors(n_neighbors = k)
        self.neigh.fit(T)
        #generating synthetic samples
        for i in range(N):
            #print(i)
            idx = np.random.randint(low=0, high=T.shape[0])
            T_sample = T[idx] #x_i random sample
            #print(T_sample.shape)
            #getting the k-nearest neighbors
            nn = self.neigh.kneighbors(T_sample.reshape(1,T.shape[1]), return_distance=False)
            #getting random k-neighbors index
            nn_index = np.random.choice(nn[0])
            while (T_sample == T[nn_index]).all():
                nn_index = np.random.choice(nn[0])
            dif = T_sample - T[nn_index]
            gap = np.random.normal()
            synthetic[i] = T_sample + gap*dif
    
        return synthetic
    
#-----------------------------------------------------
#----------- One-Class SVM Classifier ----------------
#-----------------------------------------------------
class ocsvm_utilities:
    """
    ocsvm_utilities class allows to train an one-class SVM with cross_validation
    and allows select the best initial parameters
    inputs:
        X: Dataframe with the RG to train the novelty detector. The index of this
           Dataframe must be the name of each gene.
        y: labels, for this aproach the target is a vector of ones with dimension
           equal to X.shape[0]
        test_size: the percentage of data to be used in the test step
        k_folds: the number of folds in the cross validation step
    """
    def __init__(self, X, y, test_size, k_folds):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.k_folds = k_folds
        self.cv_results = {}
        
    def ocsvm_score(self,estimator, X, y):
        """
        this function compute the recall score obtained in a one-class classifier.
        this function is used as an argument in cross validation
        inputs:
            estimator: trained estimator
            X: test data
            y: target
        output:
            recall socore
        """
        tp, fn = sum(estimator.predict(X)==1), sum(estimator.predict(X)!=1)
        return  tp/(tp + fn)
    
    def get_best_params(self, param_grid, n_iter=5):
        """
        This function compute a GridSearchCV for different training sets
        inputs:
            n_iter: number of iterations of the GridSearchCV in different training sets
                    ex: the training set is divided in 70% to train and 30% to test. For
                    default, this selection is made five times, then, the GridSearchCV is
                    used five times.
            param_grid: dictionary with the name and values of the parameter to change.
                        ex: {"nu": [.2, .5, .7]}
        return:
        
        """
        self.train_score, self.test_score = pd.DataFrame(), pd.DataFrame()
        self.train_score["best_nu"] = np.zeros(len(param_grid["nu"]))
        self.test_score["best_nu"] = np.zeros(len(param_grid["nu"]))
        #set index
        self.train_score = self.train_score.set_index(param_grid["nu"])
        self.test_score = self.test_score.set_index(param_grid["nu"])
        count_cv = 0
        for i in range(n_iter):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size)
            self.ocsvm = OneClassSVM(kernel="rbf", gamma="auto")
            self.gsCV = GridSearchCV(self.ocsvm, param_grid=param_grid, cv=self.k_folds,
                                     scoring=self.ocsvm_score, return_train_score=True, iid=False)
            self.gsCV.fit(X_train, y_train)
            #self.cv_results["iter_"+str(i)] = self.gsCV.cv_results_            
            for cv in range(self.k_folds):
                self.train_score["score_cv_"+str(count_cv)] = self.gsCV.cv_results_["split"+str(cv)+"_train_score"]
                self.test_score["score_cv_"+str(count_cv)] = self.gsCV.cv_results_["split"+str(cv)+"_test_score"]
                count_cv += 1
            self.train_score.loc[self.gsCV.best_params_["nu"], "best_nu"] += 1
            self.test_score.loc[self.gsCV.best_params_["nu"], "best_nu"] += 1
        return self.train_score, self.test_score
    
    def get_statistics(self, iterations):
        """
        this funtion computes the recall score n times in both training and test set.
        where n=iterations
        inputs:
            iterations: int. number of times that the classifier is trained and tested.
        outputs:
            train_recall_score = list with n score values in the training data. where n=iterations
            test_recall_score = list with n score values in the test data. where n=iterations
        """
        train_recall_score = []
        test_recall_score = []
        best_nu_param = self.test_score.where(self.test_score.best_nu==self.test_score.best_nu.max()).dropna().index[0]
        for i in range(iterations):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.test_size)
            clf = OneClassSVM(kernel="rbf", gamma="auto", nu=best_nu_param)
            clf.fit(X_train, y_train)
            train_recall_score.append(self.ocsvm_score(clf, X_train, y_train))
            test_recall_score.append(self.ocsvm_score(clf, X_test, y_test))
            
        return train_recall_score, test_recall_score