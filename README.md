Hello world project deploy in ESP32 Board.

# How to Create a hello world project from TF Lite:

I follow the instruction in tensorflow guideline to create this project (including components folder, main folder).

## Clone tensorflow v2.2.0 

```
git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow
```

Switch to version 2.2.0 which is more stable for this project than newest versions.

```
git checkout v2.2.0
```

## Generate then example 

The examples can be generated by the following command:

```
make -f tensorflow/lite/micro/tools/make/Makefile TARGET=esp generate_hello_world_esp_project
```

## Install ESP IDF

You can follow the instruction to install ESP-IDF to build the project in README_ESP.md.

## Building the example

Go the directory of the example you generated before with command:

```
cd tensorflow/lite/micro/tools/make/gen/esp_xtensa-esp32/prj/hello_world/esp-idf
```

To build this, run:

```
idf.py build
```

## Load and run the example

To flash (replace `/dev/ttyUSB0` with the device serial port):
```
idf.py --port /dev/ttyUSB0 flash
```

Monitor the serial output:
```
idf.py --port /dev/ttyUSB0 monitor
```

Use `Ctrl+]` to exit.

The previous two commands can be combined:
```
idf.py --port /dev/ttyUSB0 flash monitor
```

# Train a sinusoid model and create a harmonic motion with LEDs

## Train a sinusoid model

You can train a sinusoid model by many ways and can easily import the model you trained into the project above.

I suggest you to see an example for training a sinusoid model and convert it using [TensorFlow Lite and xxd](https://colab.research.google.com/github/tensorflow/tensorflow/blob/master/tensorflow/lite/micro/examples/hello_world/train/train_hello_world_model.ipynb).

## Import your model into example project

Get your model.cc that you have from the previous step that we have convert a quantized model. Rename it into sine_model_data.cc, and paste it in **main** folder.

## Create a circuit board for stimulator a harmonic motion with LEDs

# Result

[![Watch the video](https://img.youtube.com/vi/6RySA9YDytg/maxresdefault.jpg)](https://youtu.be/6RySA9YDytg)
