from kerasfactory.trainservice import TrainService

if __name__ == '__main__':
	path = 'networks_config/network_config.json'
	train = TrainService(path)
	X_train, X_test, y_train, y_test = train.get_data()
	model = train.configure_model()
	custom_model = train.fine_tune_model(model)
	final_custom_model = train.train_model(custom_model, X_train, y_train, X_test, y_test)
	train.save_model(final_custom_model)
	train.visualize_graphs_metrics()