# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# compile ASM with /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile C with /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile CXX with /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++
ASM_DEFINES = -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm4 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DLFS_THREADSAFE -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DMCMGR_HANDLE_EXCEPTIONS=1 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DUSB_HOST_CONFIG_DFU -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

ASM_INCLUDES = /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/. /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm4 /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/nxp/rt1176-sdk /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/common_task /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/lists /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/log /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/messaging /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/osa /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/phy /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/uart /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/FreeRTOS/. /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/FreeRTOS /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/Core/Include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/DSP/Include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/NN/Include /home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/littlefs

ASM_FLAGS = -O3 -DNDEBUG -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DNDEBUG

C_DEFINES = -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm4 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DLFS_THREADSAFE -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DMCMGR_HANDLE_EXCEPTIONS=1 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DUSB_HOST_CONFIG_DFU -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

C_INCLUDES = -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/. -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm4 -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/nxp/rt1176-sdk -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/FreeRTOS/. -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/FreeRTOS -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/littlefs

C_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/sukmin/Desktop/coralmicro-out-of-tree-sample= -std=gnu99 -g -Os -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DNDEBUG

CXX_DEFINES = -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm4 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DLFS_THREADSAFE -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DMCMGR_HANDLE_EXCEPTIONS=1 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DUSB_HOST_CONFIG_DFU -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

CXX_INCLUDES = -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/. -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm4 -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/nxp/rt1176-sdk -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/libs/FreeRTOS/. -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/FreeRTOS -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/sukmin/Desktop/coralmicro-out-of-tree-sample/coralmicro/third_party/nxp/rt1176-sdk/middleware/littlefs

CXX_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/sukmin/Desktop/coralmicro-out-of-tree-sample= -fno-rtti -fno-exceptions -g -Os -std=gnu++17 -mcpu=cortex-m4 -mfloat-abi=hard -mfpu=fpv4-sp-d16 -DNDEBUG

