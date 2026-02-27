#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: generator.exe <size>" << std::endl;
        return 1;
    }
    
    int size = std::atoi(argv[1]);
    std::srand(std::time(nullptr));
    
    std::ofstream f1("data/matrix1.txt");
    f1 << size << "\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double val = ((double)std::rand() / RAND_MAX) * 20.0 - 10.0;
            f1 << val << " ";
        }
        f1 << "\n";
    }
    f1.close();
    
    std::ofstream f2("data/matrix2.txt");
    f2 << size << "\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            double val = ((double)std::rand() / RAND_MAX) * 20.0 - 10.0;
            f2 << val << " ";
        }
        f2 << "\n";
    }
    f2.close();
    
    std::cout << "Matrices generated successfully!" << std::endl;
    return 0;
}