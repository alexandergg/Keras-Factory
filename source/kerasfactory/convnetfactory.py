from keras.applications import ResNet50
from keras.applications import InceptionResNetV2
from keras.applications import VGG19
from keras.applications import InceptionV3
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201

class ConvNetFactory:
	def __init__(self):
		pass

	@staticmethod
	def build(name, *args, **kargs):
		mappings = {
			"ResNet50": ConvNetFactory.ResNet50,
			"DenseNet121": ConvNetFactory.DenseNet121,
			"DenseNet169": ConvNetFactory.DenseNet169,
			"DenseNet201": ConvNetFactory.DenseNet201,
			"VGG19": ConvNetFactory.VGG19,
			"InceptionResNetV2": ConvNetFactory.InceptionResNetV2,
			"InceptionV3": ConvNetFactory.InceptionV3,
		}

		builder = mappings.get(name, None)

		if builder is None:
			return None

		return builder(*args, **kargs)
	
	@staticmethod
	def ResNet50(include_top, image_input, **kwargs):
		model = ResNet50(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model
	
	@staticmethod
	def InceptionResNetV2(include_top, image_input, **kwargs):
		model = InceptionResNetV2(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model
	
	@staticmethod
	def InceptionV3(include_top, image_input, **kwargs):
		model = InceptionV3(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model

	@staticmethod
	def DenseNet121(include_top, image_input, **kwargs):
		model = DenseNet121(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model
	
	@staticmethod
	def DenseNet169(include_top, image_input, **kwargs):
		model = DenseNet169(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model
	
	@staticmethod
	def DenseNet201(include_top, image_input, **kwargs):
		model = DenseNet201(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model
		
	@staticmethod
	def VGG19(include_top, image_input, **kwargs):
		model = VGG19(input_tensor=image_input, include_top=include_top, weights='imagenet')
		return model