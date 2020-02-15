import numpy as np
import os, collections, time, json, natsort
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import GlobalAveragePooling2D, Dense, Dropout,Activation,Flatten
from kerasfactory.imagenet_utils import preprocess_input
from keras.layers import Input
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import TensorBoard, EarlyStopping
from tensorboard.plugins.pr_curve import summary as pr_summary
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from kerasfactory.convnetfactory import ConvNetFactory

class TrainService():
	def __init__(self, config_path):
		self.__config = json.loads(open(config_path).read())
		self.__image_input = Input(shape=(224, 224, 3))
		self.__num_classes = 4
		self.__hist = None

	def configure_model(self):
		train_network_params = self.__config['network']
		image_input = self.__image_input

		network_name = train_network_params['name']
		include_top = train_network_params['include_top']
				
		model = ConvNetFactory.build(network_name, include_top, image_input)
		return model

	def fine_tune_model(self, model):
		train_network_params = self.__config['network']
		activation = train_network_params['activation']
		loss_type = train_network_params['loss_type']
		optimizer = train_network_params['optimizer']
		metrics = train_network_params['metrics']
		image_input = self.__image_input
		num_classes = self.__num_classes

		last_layer = model.get_layer('fc2').output
		out = Dense(num_classes, activation=activation, name='output')(last_layer)
		custom_model = Model(image_input, out)
		custom_model.summary()

		for layer in custom_model.layers[:-1]:
			layer.trainable = False

		custom_model.layers[3].trainable
		print("[INFO] compiling model...")
		custom_model.compile(loss=loss_type,optimizer=optimizer,metrics=[metrics])

		return custom_model

	def train_model(self, custom_model, X_train, y_train, X_test, y_test):
		train_network_params = self.__config['network']
		batch_size_training = train_network_params['batch_size_training']
		batch_size_evaluate = train_network_params['batch_size_evaluate']
		epochs = train_network_params['epochs']
		verbose = train_network_params['verbose']
		tensorboard_params = self.__config['tensorboard']
		name_runs = tensorboard_params["name_runs"]

		t=time.time()
		tensorboard_name = "{0}-{1}".format(name_runs, t)
		tensorBoard = [VisualizeTensorBoard(log_dir="logs/{}".format(tensorboard_name), histogram_freq=1,  
          					write_graph=True, write_images=True), EarlyStopping(monitor='val_loss', patience=3)]

		print("[INFO] starting training...")
		print('Run `tensorboard --logdir="./logs"` to see the results')
		hist = custom_model.fit(X_train, y_train, batch_size=batch_size_training, epochs=epochs,
							verbose=verbose, validation_data=(X_test, y_test), callbacks=tensorBoard)
		self.__hist = hist
		print('Training time: %s' % (t - time.time()))

		print("[INFO] Evaluate model...")
		(loss, accuracy) = custom_model.evaluate(X_test, y_test, batch_size=batch_size_evaluate, verbose=verbose)
		print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

		return custom_model

	def save_model(self, model):
		train_network_params = self.__config['network']
		model_save_path = train_network_params["model_save_path"]
		print("[INFO] dumping architecture and weights to file...")
		model.save(model_save_path)

	def get_data(self):
		image_path = self.__config['data_path']
		data_path = image_path['path']
		num_classes = self.__num_classes

		img_data_list=[]

		img_list=os.listdir(data_path)
		img_list = natsort.natsorted(img_list,reverse=False)
		
		for img in img_list:
			print(img)
			img_path = data_path + '/'+ img
			img = image.load_img(img_path, target_size=(224, 224))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			x = preprocess_input(x)
			print('Input image shape:', x.shape)
			img_data_list.append(x)

		img_data = np.array(img_data_list)
		print (img_data.shape)
		img_data=np.rollaxis(img_data,1,0)
		print (img_data.shape)
		img_data=img_data[0]
		print (img_data.shape)

		num_of_samples = img_data.shape[0]
		labels = np.ones((num_of_samples,),dtype='int64')

		# labels[:24]=0
		# labels[24:48]=1
		# labels[48:72]=2
		# labels[72:96]=3

		labels[:50]=0
		labels[50:100]=1
		labels[100:150]=2

		print(collections.Counter(labels))

		Y = np_utils.to_categorical(labels, num_classes)
		x,y = shuffle(img_data,Y, random_state=2)
		X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

		return X_train, X_test, y_train, y_test

	def visualize_graphs_metrics(self):

		hist = self.__hist

		train_loss=hist.history['loss']
		val_loss=hist.history['val_loss']
		train_acc=hist.history['acc']
		val_acc=hist.history['val_acc']
		xc=range(12)

		plt.figure(1,figsize=(7,5))
		plt.plot(xc,train_loss)
		plt.plot(xc,val_loss)
		plt.xlabel('num of Epochs')
		plt.ylabel('loss')
		plt.title('train_loss vs val_loss')
		plt.grid(True)
		plt.legend(['train','val'])
		plt.style.use(['classic'])

		plt.figure(2,figsize=(7,5))
		plt.plot(xc,train_acc)
		plt.plot(xc,val_acc)
		plt.xlabel('num of Epochs')
		plt.ylabel('accuracy')
		plt.title('train_acc vs val_acc')
		plt.grid(True)
		plt.legend(['train','val'],loc=4)
		plt.style.use(['classic'])

		plt.show()

class VisualizeTensorBoard(TensorBoard):
    def __init__(self, *args, **kwargs):
        self.pr_curve = kwargs.pop('pr_curve', True)
        super(VisualizeTensorBoard, self).__init__(*args, **kwargs)

        global tf
        import tensorflow as tf

    def set_model(self, model):
        super(VisualizeTensorBoard, self).set_model(model)

        if self.pr_curve:
            predictions = self.model._feed_outputs[0]
            labels = tf.cast(self.model._feed_targets[0], tf.bool)
            self.pr_summary = pr_summary.op(name='pr_curve',
                                            predictions=predictions,
                                            labels=labels,
                                            display_name='Precision-Recall Curve')

    def on_epoch_end(self, epoch, logs=None):
        super(VisualizeTensorBoard, self).on_epoch_end(epoch, logs)

        if self.pr_curve and self.validation_data:
            tensors = self.model._feed_targets + self.model._feed_outputs
            predictions = self.model.predict(self.validation_data[:-2])
            val_data = [self.validation_data[-2], predictions]
            feed_dict = dict(zip(tensors, val_data))
            result = self.sess.run([self.pr_summary], feed_dict=feed_dict)
            self.writer.add_summary(result[0], epoch)
        self.writer.flush()