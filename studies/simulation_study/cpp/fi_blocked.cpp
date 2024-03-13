#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <string>

using namespace std;

double log_true_c(int ppb, int block, double v) {
    return (ppb * block / 2.0) * (log(v) - log(1 + v));
}

double g(vector<double>& theta, int ppb, double v) {
    double like = 0.0;
    for (int i = 0; i < ppb; ++i) {
        like -= theta[i] * theta[i] / (2.0 * v);
    }
    double prior = 0.0;
    for (int i = 0; i < ppb; ++i) {
        prior += -0.5 * log(2 * M_PI) - 0.5 * theta[i] * theta[i];
    }
    return like + prior;
}

vector<double> generate_target_sample(int n, int ppb, double v) {
    vector<double> target_sample(n * ppb);
    for (int i = 0; i < n * ppb; ++i) {
        target_sample[i] = sqrt(v / (v + 1)) * ((double)rand() / RAND_MAX);
    }
    return target_sample;
}

double compute_density(vector<double>& target_sample, int n, int ppb, int R) {
    double density = 0.0;
    for (int i = 0; i < n; ++i) {
        double product = 1.0;
        for (int j = 0; j < ppb; ++j) {
            product *= sin(R * target_sample[i * ppb + j]) / target_sample[i * ppb + j];
        }
        density += product;
    }
    return density / (n * pow(M_PI, ppb));
}

void create_histogram_plot(string label) {
    // Generate GNU Plotutils command file
    string gnufname = label + ".gnu";
    ofstream cmdFile(gnufname);
    if (!cmdFile.is_open()) {
        cerr << "Unable to open file for writing." << endl;
        return;
    }
    cmdFile << "set term png\n";
    cmdFile << "set output '"<< label << ".png'\n";
    cmdFile << "set title 'Histogram of Simulation Results'\n";
    cmdFile << "set xlabel 'LnZ Bias'\n";
    cmdFile << "set ylabel 'Frequency'\n";
    cmdFile << "binwidth=0.1\n";
    cmdFile << "bin(x,width)=width*floor(x/width) + binwidth/2.0\n";
    cmdFile << "plot '"<< label << ".txt' using (bin($1,binwidth)):(1.0) smooth freq with boxes\n";
    cmdFile.close();

    // Execute GNU Plotutils to create plot
    string cmd = "gnuplot " + gnufname;
    system(cmd.c_str());
    cout << "Histogram saved" << endl;
}

int main(int argc, char *argv[]) {
    int N = 10000000;
    double V = 1.0;
    int R = 100;
    int PPB = 10;
    int BLOCK = 5;
    const int ITERATIONS = 300;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--N") {
            N = stoi(argv[++i]);
        } else if (arg == "--V") {
            V = stod(argv[++i]);
        } else if (arg == "--R") {
            R = stoi(argv[++i]);
        } else if (arg == "--PPB") {
            PPB = stoi(argv[++i]);
        } else if (arg == "--BLOCK") {
            BLOCK = stoi(argv[++i]);
        }
    }

    cout << "N: " << N << ", V: " << V << ", R: " << R << ", PPB: " << PPB << ", BLOCK: " << BLOCK << endl;
    srand(42);

    double ltrue_c = log_true_c(PPB, BLOCK, V);

    vector<double> density_results(ITERATIONS);
    vector<double> simulation_results(ITERATIONS);

    cout << "Progress: ";
    for (int i = 0; i < ITERATIONS; ++i) {
        vector<double> post_dens_i(BLOCK);
        for (int j = 0; j < BLOCK; ++j) {
            vector<double> target_sample = generate_target_sample(N, PPB, V);
            post_dens_i[j] = compute_density(target_sample, N, PPB, R);
        }
        double post_dens = 0.0;
        for (int j = 0; j < BLOCK; ++j) {
            post_dens += log(fabs(post_dens_i[j]));
        }
        density_results[i] = post_dens;

        vector<double> theta(PPB, 0.0);
        double lpriorlike = g(theta, PPB, V);
        simulation_results[i] = lpriorlike - post_dens;

        // Print progress bar
        if ((i + 1) % (ITERATIONS / 30) == 0) {
            cout << "\rProgress: " << ((i + 1) * 100) / ITERATIONS << "%";
            cout.flush();
        }
    }
    cout << "\rProgress: 100%" << endl;

    double mean_simulation_results = 0.0;
    for (int i = 0; i < ITERATIONS; ++i) {
        mean_simulation_results += simulation_results[i];
    }
    mean_simulation_results /= ITERATIONS;

    cout << "Mean Simulation Results: " << mean_simulation_results << endl;
    cout << "True C: " << ltrue_c << endl;
    cout << "Difference (Simulation Results - True C): " << mean_simulation_results - ltrue_c << endl;

    string label = "simulation_results_ppb" + to_string(PPB) + "_block" + to_string(BLOCK) + "_n" + to_string(N) + "_r" + to_string(R) + "_v" + to_string(V);
    ofstream outFile(label + ".txt");
    if (outFile.is_open()) {
        for (int i = 0; i < ITERATIONS; ++i) {
            outFile << simulation_results[i]-ltrue_c << endl;
        }
        outFile.close();
        cout << "Simulation results saved to simulation_results.txt" << endl;
    } else {
        cerr << "Unable to open file for writing." << endl;
    }
    create_histogram_plot(label);

    return 0;
}
