# An example of binary image classification 

Toy image classifier

## A random image generator written in F# 
- A dotnet project lives in ShapeGenerator sub-folder
- `dotnet run` will generate random images of circles
- `dotnet run r` will generate random images of rectangles

## A Python classifier
- Prior to running make suer to `pip3 install torch torchvision`
- Program `experiment.py` that trains a model using PyTorch
- Program `recognizer.py` runs the model on images in a folder `input_images`
- Achieves 95% accuracy
