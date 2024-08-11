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
// constexpr char kModelPath[] =
//     "/models/01.tflite";
constexpr int kTopK = 5;
constexpr float kThreshold = 0.5;

// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
STATIC_TENSOR_ARENA_IN_SDRAM(temp, 1024 * 1024 * 3);

int cnt = 1;
void invoke(char kModelPath[]) {

  auto start = TimerMillis();
  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddCustom(kCustomOp, RegisterCustomOp());
  resolver.AddQuantize();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddPad();
  resolver.AddAdd();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddConcatenation();

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
  //printf("input size: %d\r\n", input_tensor->bytes);

  for(int i =0; i < input_tensor->bytes; i++) {
    input_tensor->data.uint8[i] = temp[i];
  }

  if (interpreter.Invoke() != kTfLiteOk) {
    printf("Failed to invoke\r\n");
    vTaskSuspend(nullptr);
  }

  auto* output = interpreter.output(0);
  //printf("output size: %d\r\n", output->bytes);
  for (int i = 0; i < output->bytes; i++) {
    temp[i] = output->data.uint8[i];
  }

  auto end = TimerMillis();
  printf(
      "Layer %d invoke time: %lums\r\n",
      cnt,
      static_cast<uint32_t>(end - start));
  //printf("%d tflite done!\r\n", cnt++);
}

[[noreturn]] void Main() {
  printf("\r\nTest Start!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  auto boot_start = TimerMillis();
  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }
  auto boot_end = TimerMillis();
  printf(
      "TPU boot time: %lums\r\n",
      static_cast<uint32_t>(boot_end - boot_start));

  // for (int i = 0; i < 1024 * 1024; i++) {
  //   temp[i] = i % 128;
  // }
  // memset(temp, 0, 1024 * 1024);

  // memset(temp, 0, 1024 * 1024);
  // invoke("/models/multi/inception_v2/1_1.tflite");

  memset(temp, 0, 1024 * 1024);
  invoke("/models/multi/inception_v2/5_1_edgetpu.tflite");
  memset(temp, 0, 1024 * 1024);
  invoke("/models/multi/inception_v2/5_2_edgetpu.tflite");
  memset(temp, 0, 1024 * 1024);
  invoke("/models/multi/inception_v2/5_3_edgetpu.tflite");

  // cpu full
  // memset(temp, 0, 1024 * 1024);
  // invoke("/models/mobilenet_v2_1.0_224_quant.tflite");
  
  // tpu full
  // memset(temp, 0, 1024 * 1024);
  // invoke("/models/tpu/mobilenet_v2_1.0_224_quant_edgetpu.tflite");

  // cpu models
  // invoke("/models/cpu/01.tflite");
  // invoke("/models/cpu/02.tflite");
  // invoke("/models/cpu/03.tflite");
  // invoke("/models/cpu/04.tflite");
  // invoke("/models/cpu/05.tflite");
  // invoke("/models/cpu/06.tflite");
  // invoke("/models/cpu/07.tflite");
  // invoke("/models/cpu/08.tflite");
  // invoke("/models/cpu/09.tflite");
  // invoke("/models/cpu/10.tflite");
  // invoke("/models/cpu/11.tflite");
  // invoke("/models/cpu/12.tflite");
  // invoke("/models/cpu/13.tflite");
  // invoke("/models/cpu/14.tflite");
  // invoke("/models/cpu/15.tflite");
  // invoke("/models/cpu/16.tflite");
  // invoke("/models/cpu/17.tflite");
  // invoke("/models/cpu/18.tflite");
  // invoke("/models/cpu/19.tflite");
  // invoke("/models/cpu/20.tflite");
  // invoke("/models/cpu/21.tflite");
  // invoke("/models/cpu/22.tflite");
  // invoke("/models/cpu/23.tflite");
  // invoke("/models/cpu/24.tflite");
  // invoke("/models/cpu/25.tflite");
  // invoke("/models/cpu/26.tflite");
  // invoke("/models/cpu/27.tflite");
  // invoke("/models/cpu/28.tflite");
  // invoke("/models/cpu/29.tflite");

  //tpu models
  // invoke("/models/tpu/01_edgetpu.tflite"); //NO
  // invoke("/models/tpu/02_edgetpu.tflite"); //NO
  // invoke("/models/tpu/03_edgetpu.tflite"); //NO
  // invoke("/models/tpu/04_edgetpu.tflite"); //NO
  // invoke("/models/tpu/05_edgetpu.tflite"); //YES
  // invoke("/models/tpu/06_edgetpu.tflite"); //NO
  // invoke("/models/tpu/07_edgetpu.tflite"); //NO
  // invoke("/models/tpu/08_edgetpu.tflite"); //NO
  // invoke("/models/tpu/09_edgetpu.tflite"); //YES
  // invoke("/models/tpu/10_edgetpu.tflite"); //NO
  // invoke("/models/tpu/11_edgetpu.tflite"); //YES
  // invoke("/models/tpu/12_edgetpu.tflite"); //NO
  // invoke("/models/tpu/13_edgetpu.tflite"); //YES
  // invoke("/models/tpu/14_edgetpu.tflite"); //NO
  // invoke("/models/tpu/15_edgetpu.tflite"); //YES
  // invoke("/models/tpu/16_edgetpu.tflite"); //YES
  // invoke("/models/tpu/17_edgetpu.tflite"); //NO
  // invoke("/models/tpu/18_edgetpu.tflite"); //YES
  // invoke("/models/tpu/19_edgetpu.tflite"); //YES
  // invoke("/models/tpu/20_edgetpu.tflite"); //YES
  // invoke("/models/tpu/21_edgetpu.tflite"); //YES
  // invoke("/models/tpu/22_edgetpu.tflite"); //YES
  // invoke("/models/tpu/23_edgetpu.tflite"); //YES
  // invoke("/models/tpu/24_edgetpu.tflite"); //YES
  // invoke("/models/tpu/25_edgetpu.tflite"); //YES
  // invoke("/models/tpu/26_edgetpu.tflite"); //YES
  // invoke("/models/tpu/27_edgetpu.tflite"); //YES
  // invoke("/models/tpu/28_edgetpu.tflite"); //YES
  // invoke("/models/tpu/29_edgetpu.tflite"); //NO

  //tpu + cpu models
  // invoke("/models/cpu/01.tflite");
  // invoke("/models/cpu/02.tflite");
  // invoke("/models/cpu/03.tflite");
  // invoke("/models/cpu/04.tflite");
  // invoke("/models/tpu/05_edgetpu.tflite"); //YES
  // invoke("/models/cpu/06.tflite");
  // invoke("/models/cpu/07.tflite");
  // invoke("/models/cpu/08.tflite");
  // invoke("/models/cpu/09.tflite");
  // invoke("/models/cpu/10.tflite");
  // invoke("/models/tpu/11_edgetpu.tflite"); //YES
  // invoke("/models/cpu/12.tflite");
  // invoke("/models/tpu/13_edgetpu.tflite"); //YES
  // invoke("/models/cpu/14.tflite");
  // invoke("/models/tpu/15_edgetpu.tflite"); //YES
  // invoke("/models/tpu/16_edgetpu.tflite"); //YES
  // invoke("/models/cpu/17.tflite");
  // invoke("/models/tpu/18_edgetpu.tflite"); //YES
  // invoke("/models/tpu/19_edgetpu.tflite"); //YES
  // invoke("/models/tpu/20_edgetpu.tflite"); //YES
  // invoke("/models/tpu/21_edgetpu.tflite"); //YES
  // invoke("/models/tpu/22_edgetpu.tflite"); //YES
  // invoke("/models/tpu/23_edgetpu.tflite"); //YES
  // invoke("/models/tpu/24_edgetpu.tflite"); //YES
  // invoke("/models/tpu/25_edgetpu.tflite"); //YES
  // invoke("/models/tpu/26_edgetpu.tflite"); //YES
  // invoke("/models/tpu/27_edgetpu.tflite"); //YES
  // invoke("/models/tpu/28_edgetpu.tflite"); //YES
  // invoke("/models/cpu/29.tflite");


  while(1);
}

}  // namespace
}  // namespace coralmicro

extern "C" [[noreturn]] void app_main(void *param) {
  (void)param;
  coralmicro::Main();
}
