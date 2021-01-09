#include <iostream>
#include <memory>

#include "edgetpu_c.h"

std::string ToString(edgetpu_device_type type) {
    switch (type) {
        case EDGETPU_APEX_PCI:
            return "PCI";
        case EDGETPU_APEX_USB:
            return "USB";
    }
    return "Unknown";
}

int main(int argc, char* argv[]) {
    size_t num_devices;
    std::unique_ptr<edgetpu_device, decltype(&edgetpu_free_devices)> devices(
    edgetpu_list_devices(&num_devices), &edgetpu_free_devices);

    for (size_t i = 0; i < num_devices; ++i) {
        const auto& device = devices.get()[i];
        std::cout << i << " " << ToString(device.type) << " " << device.path
                  << std::endl;
    }

    return 0;
}

