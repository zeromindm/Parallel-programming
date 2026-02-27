#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: multiplier.exe <size>" << std::endl;
        return 1;
    }
    
    int size = std::atoi(argv[1]);
    
    std::ifstream f1("data/matrix1.txt");
    int n1;
    f1 >> n1;
    std::vector<std::vector<double>> A(size, std::vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            f1 >> A[i][j];
    f1.close();
    
    std::ifstream f2("data/matrix2.txt");
    int n2;
    f2 >> n2;
    std::vector<std::vector<double>> B(size, std::vector<double>(size));
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            f2 >> B[i][j];
    f2.close();
    
    std::vector<std::vector<double>> C(size, std::vector<double>(size, 0.0));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < size; i++)
        for (int k = 0; k < size; k++)
            for (int j = 0; j < size; j++)
                C[i][j] += A[i][k] * B[k][j];
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    std::cout << elapsed.count() << std::endl;
    
    std::ofstream fout("results/result.txt");
    fout << size << "\n";
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++)
            fout << C[i][j] << " ";
        fout << "\n";
    }
    fout.close();
    
    return 0;
}