/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "main_functions.h"

#include "constants.h"
#include "output_handler.h"
#include "sine_model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"

#define SEGMENT 30

gpio_num_t pinArray[6] = { GPIO_NUM_0, GPIO_NUM_4, GPIO_NUM_16, GPIO_NUM_17, GPIO_NUM_5, GPIO_NUM_18 };
float x_val = M_PI / 2;
float y_val;
float dt = 2 * M_PI / SEGMENT;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;

// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 2 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

void setupGPIO() {
  gpio_config_t gpioConfig;
  gpioConfig.pin_bit_mask = 0;
  for (int i = 0; i < 6; i++) {
    gpioConfig.pin_bit_mask |= (1 << pinArray[i]);
  }

  gpioConfig.mode = GPIO_MODE_OUTPUT;
  gpioConfig.pull_up_en = GPIO_PULLUP_DISABLE;
  gpioConfig.pull_down_en = GPIO_PULLDOWN_DISABLE;
  gpioConfig.intr_type = GPIO_INTR_DISABLE;
  gpio_config(&gpioConfig);
}

int chooseLED(float y) {
  if (-1 <= y && y < -0.66) return 0;
  if (-0.66 <= y && y < -0.33) return 1;
  if (-0.33 <= y && y < 0) return 2;
  if (0 <= y && y < 0.33) return 3;
  if (0.33 <= y && y < 0.66) return 4;
  if (0.66 <= y && y <= 1) return 5;
  return -1;
}

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(g_sine_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::ops::micro::AllOpsResolver resolver;

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }

  // Obtain pointers to the model's input and output tensors.
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Keep track of how many inferences we have performed.
  inference_count = 0;
  setupGPIO();
}

// The name of this function is important for Arduino compatibility.
void loop() {
  if (x_val > 2 * M_PI) x_val -= 2 * M_PI;
  input->data.f[0] = x_val;

  // Run inference, and report any error
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed on x_val: %f\n",
                         static_cast<double>(x_val));
    return;
  }

  // Read the predicted y value from the model's output tensor
  float y_val = output->data.f[0];

  // Output the results. A custom HandleOutput function can be implemented
  // for each supported hardware target.
  HandleOutput(error_reporter, x_val, y_val);

  for(int i = 0; i < 6; i++) {
    gpio_set_level(pinArray[i], 0);
  }
  int iLED = chooseLED(y_val);
  printf("Choose LED: %d\n", iLED);
  gpio_set_level(pinArray[iLED], 1);
  vTaskDelay((1000 / SEGMENT) / portTICK_PERIOD_MS);
  gpio_set_level(pinArray[iLED], 0);

  // Increment the inference_counter, and reset it if we have reached
  // the total number per cycle
  x_val += dt;
  inference_count += 1;
  if (inference_count >= kInferencesPerCycle) inference_count = 0;
}
