import os
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import PIL.Image
import time

# Set up matplotlib parameters
mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

# Function to convert a tensor to an image
def tensor_to_image(tensor):
    tensor = tensor * 255  # Scale the tensor values to 0-255
    tensor = np.array(tensor, dtype=np.uint8)  # Convert tensor to numpy array
    if np.ndim(tensor) > 3:  # If the tensor has more than 3 dimensions
        assert tensor.shape[0] == 1  # Ensure the first dimension is 1
        tensor = tensor[0]  # Remove the first dimension
    return PIL.Image.fromarray(tensor)  # Convert the numpy array to an image

# Load images from local files
content_path = 'content_image.jpg'
style_path = 'style_image.jpg'

# Function to load and preprocess an image
def load_img(path_to_img):
    max_dim = 512  # Maximum dimension for the image
    img = tf.io.read_file(path_to_img)  # Read the image file
    img = tf.image.decode_image(img, channels=3)  # Decode the image
    img = tf.image.convert_image_dtype(img, tf.float32)  # Convert the image to float32 type
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)  # Get the shape of the image
    long_dim = max(shape)  # Get the longer dimension
    scale = max_dim / long_dim  # Calculate the scale factor
    new_shape = tf.cast(shape * scale, tf.int32)  # Calculate the new shape
    img = tf.image.resize(img, new_shape)  # Resize the image
    img = img[tf.newaxis, :]  # Add a batch dimension
    return img  # Return the preprocessed image

# Function to display an image
def imshow(image, title=None):
    if len(image.shape) > 3:  # If the image has more than 3 dimensions
        image = tf.squeeze(image, axis=0)  # Remove the batch dimension
    plt.imshow(image)  # Display the image
    if title:  # If a title is provided
        plt.title(title)  # Set the title
    plt.axis('off')  # Turn off the axis
    plt.show()  # Show the image

# Load and display the content and style images
content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

# Load the VGG19 model
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

# Specify the content and style layers
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

# Function to create a VGG model that returns a list of intermediate output values
def vgg_layers(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False  # Set the model to non-trainable
    outputs = [vgg.get_layer(name).output for name in layer_names]  # Get the outputs of the specified layers
    model = tf.keras.Model([vgg.input], outputs)  # Create the model
    return model  # Return the model

# Extract style and content features from the style image
style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image * 255.0)  # Multiply by 255 to scale back to original range

# Print the shape, min, max, and mean of each style layer output
for name, output in zip(style_layers, style_outputs):
    print(name)
    print("  shape: ", output.numpy().shape)
    print("  min: ", output.numpy().min())
    print("  max: ", output.numpy().max())
    print("  mean: ", output.numpy().mean())
    print()

# Function to compute the Gram matrix
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)  # Compute the Gram matrix
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)  # Calculate the number of locations
    return result / num_locations  # Normalize the Gram matrix

# Custom model to extract style and content features
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)  # Create the VGG model with the specified layers
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False  # Set the VGG model to non-trainable

    def call(self, inputs):
        inputs = inputs * 255.0  # Scale the inputs to the original range
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)  # Preprocess the inputs
        outputs = self.vgg(preprocessed_input)  # Get the outputs of the VGG model
        style_outputs, content_outputs = (outputs[:self.num_style_layers], outputs[self.num_style_layers:])  # Split the outputs into style and content
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]  # Compute the Gram matrix for each style output
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}  # Create a dictionary of content outputs
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}  # Create a dictionary of style outputs
        return {'content': content_dict, 'style': style_dict}  # Return the dictionaries

# Create an instance of the custom model
extractor = StyleContentModel(style_layers, content_layers)

# Extract style and content features from the content image
results = extractor(content_image)

# Print the shape, min, max, and mean of each style and content output
print('Styles:')
for name, output in sorted(results['style'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())
    print()

print("Contents:")
for name, output in sorted(results['content'].items()):
    print("  ", name)
    print("    shape: ", output.numpy().shape)
    print("    min: ", output.numpy().min())
    print("    max: ", output.numpy().max())
    print("    mean: ", output.numpy().mean())

# Extract style and content targets from the style and content images
style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

# Initialize the image variable for optimization
image = tf.Variable(content_image)

# Define optimization parameters
opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
style_weight = 1e-2
content_weight = 1e4
total_variation_weight = 30

# Define the style content loss function
def style_content_loss(outputs):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_targets[name])**2) for name in style_outputs.keys()])  # Compute style loss
    style_loss *= style_weight / num_style_layers  # Scale style loss
    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_targets[name])**2) for name in content_outputs.keys()])  # Compute content loss
    content_loss *= content_weight / num_content_layers  # Scale content loss
    loss = style_loss + content_loss  # Total loss
    return loss

# Define the training step function
@tf.function()
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)  # Extract features from the image
        loss = style_content_loss(outputs)  # Compute the loss
        loss += total_variation_weight * tf.image.total_variation(image)  # Add total variation loss
    grad = tape.gradient(loss, image)  # Compute gradients
    opt.apply_gradients([(grad, image)])  # Apply gradients
    image.assign(tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0))  # Clip image values to [0, 1]

# Training parameters
epochs = 10
steps_per_epoch = 100
step = 0

# Training loop
start = time.time()  # Start time
for n in range(epochs):
    for m in range(steps_per_epoch):
        step += 1  # Increment step count
        train_step(image)  # Perform a training step
        if step % 10 == 0:  # Every 10 steps
            print(".", end='', flush=True)  # Print a dot
            plt.figure(figsize=(12, 12))  # Create a new figure
            imshow(image.numpy()[0], title="Train step: {}".format(step))  # Display the image with the step count
            plt.axis('off')  # Turn off the axis
            plt.show()  # Show the figure

end = time.time()  # End time
print("\nTotal time: {:.1f} seconds".format(end-start))  # Print total time

# Save the stylized image
file_name = 'stylized-image.png'
tensor_to_image(image).save(file_name)  # Save the image
print("Stylized image saved as:", file_name)  # Print the file name
