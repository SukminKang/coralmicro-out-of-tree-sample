// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdio>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/timer.h"
#include "libs/tensorflow/utils.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"


namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/mobilenet_v2_1.0_224_quant.tflite";

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

[[noreturn]] void Main() {
  printf("\r\nTest Start!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  // auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  // if (!tpu_context) {
  //   printf("ERROR: Failed to get EdgeTpu context\r\n");
  //   vTaskSuspend(nullptr);
  // }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<10> resolver;
  // resolver.AddCustom(kCustomOp, RegisterCustomOp());
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddPad();
  resolver.AddAdd();
  resolver.AddAveragePool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();  

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  auto* input_tensor = interpreter.input_tensor(0);
  for(int i =0; i < input_tensor->bytes; i++) {
    input_tensor->data.uint8[i] = 0;
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Failed to invoke\r\n");
    vTaskSuspend(nullptr);
  }

  while(1);
}

}  // namespace
}  // namespace coralmicro

extern "C" [[noreturn]] void app_main(void *param) {
  (void)param;
  coralmicro::Main();
}
