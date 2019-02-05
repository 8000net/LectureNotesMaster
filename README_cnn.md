# Visualizing Convolutional Neural Networks 


### What's in this Lecture
This Lecture overviews some useful strategies for visualizing what is learned by Convolutional Neural Networks (CNNs). The Jupyter notebook includes lecture notes with code examples of various modern techniques for CNN visualizations. The models/ and images/ directories include files needed to run the jupyter notebook.

## Lecture Demo
In the activations-demo/ folder, there are two examples of real-time interactive visualizations using the camera on a laptop. 

First, heatmap.py overlays a heatmap of majority class activation for frames seen by the camera while the majority class is printed to the console. To run this example, simply use `python heatmap.py`.

Second, activations.py shows the activations in each convolutional layer of VGG using frames taken from the laptop camera. It can be run using `python activations.py`. While the program is running, use the 'q' key to exit, and the 'w' and 'e' keys to switch between layers of VGG. 
