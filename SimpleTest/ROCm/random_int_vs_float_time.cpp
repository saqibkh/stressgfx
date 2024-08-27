#include <iostream>
#include <random>
#include <chrono>

int main() {
    const size_t numSamples = 1'000'000;
    
    // Random number engine
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Integer distribution
    std::uniform_int_distribution<int> intDist(0, 100);
    
    // Float distribution
    std::uniform_real_distribution<float> floatDist(0.0f, 100.0f);
    
    // Measure integer generation time
    auto startInt = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numSamples; ++i) {
        volatile int randInt = intDist(gen); // volatile prevents optimization
    }
    auto endInt = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationInt = endInt - startInt;
    
    // Measure float generation time
    auto startFloat = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < numSamples; ++i) {
        volatile float randFloat = floatDist(gen); // volatile prevents optimization
    }
    auto endFloat = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> durationFloat = endFloat - startFloat;
    
    std::cout << "Integer generation time: " << durationInt.count() << " seconds\n";
    std::cout << "Float generation time: " << durationFloat.count() << " seconds\n";

    return 0;
}

