#include <iostream>

namespace aspl {
    namespace utils {

        void log_info(const char* msg) {
            std::cout << "[ASPL INFO] " << msg << std::endl;
        }

    }
}