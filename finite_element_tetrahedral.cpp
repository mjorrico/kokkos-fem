#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>

#include <Kokkos_Core.hpp>

#define Number float
#define MemLayout Kokkos::LayoutLeft

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#define ExecSpace Kokkos::Cuda
#elif defined(KOKKOS_ENABLE_OPENMP)
#define MemSpace Kokkos::HostSpace
#define ExecSpace Kokkos::OpenMP
#else
#define MemSpace Kokkos::HostSpace
#define ExecSpace Kokkos::Serial
#endif

void check_args(int argc, char* argv[], long& n_min, long& n_max, long& repeat);

void benchmark(long N, long repeat);

template <typename JacobianType, typename AType, typename RP>
void calculate(const JacobianType& J, AType& A, const long N);

int main(int argc, char* argv[]) {
    long n_min = 100;
    long n_max = -1;
    long repeat = -1;
    check_args(argc, argv, n_min, n_max, repeat);

    Kokkos::initialize(argc, argv);
    {
        for (long n = n_min; n < n_max; n = n * 1.1 + 1) {
            benchmark(n, repeat);
        }
    }
    Kokkos::finalize();

    return 0;
}


template <typename JacobianType, typename AType, typename RP>
void calculate(const JacobianType& J, AType& A, const long N) {
    Kokkos::parallel_for("calculate", RP(0, N), KOKKOS_LAMBDA(long i) {
        Number C0 = J(i, 1, 1) * J(i, 2, 2) - J(i, 1, 2) * J(i, 2, 1); // 3 FLOP
        Number C1 = J(i, 1, 2) * J(i, 2, 0) - J(i, 1, 0) * J(i, 2, 2); // 3 FLOP
        Number C2 = J(i, 1, 0) * J(i, 2, 1) - J(i, 1, 1) * J(i, 2, 0); // 3 FLOP

        Number inv_J_det = J(i, 0, 0) * C0 + J(i, 0, 1) * C1 + J(i, 0, 2) * C2; // 5 FLOP
        Number d = (1. / 6.) / inv_J_det; // 2 FLOP

        Number G0 = d * (J(i, 0, 0) * J(i, 0, 0) + J(i, 1, 0) * J(i, 1, 0) + J(i, 2, 0) * J(i, 2, 0)); // 6 FLOP
        Number G1 = d * (J(i, 0, 0) * J(i, 0, 1) + J(i, 1, 0) * J(i, 1, 1) + J(i, 2, 0) * J(i, 2, 1)); // 6 FLOP
        Number G2 = d * (J(i, 0, 0) * J(i, 0, 2) + J(i, 1, 0) * J(i, 1, 2) + J(i, 2, 0) * J(i, 2, 2)); // 6 FLOP
        Number G3 = d * (J(i, 0, 1) * J(i, 0, 1) + J(i, 1, 1) * J(i, 1, 1) + J(i, 2, 1) * J(i, 2, 1)); // 6 FLOP
        Number G4 = d * (J(i, 0, 1) * J(i, 0, 2) + J(i, 1, 1) * J(i, 1, 2) + J(i, 2, 1) * J(i, 2, 2)); // 6 FLOP
        Number G5 = d * (J(i, 0, 2) * J(i, 0, 2) + J(i, 1, 2) * J(i, 1, 2) + J(i, 2, 2) * J(i, 2, 2)); // 6 FLOP // 52 FLOP

        A(i, 0, 0) = G0; // taken care of by memory controller
        // A(i, 0, 1) = A(i, 1, 0) = G1; // FIX THIS LATER, DON'T USE DOUBLE =
        A(i, 0, 1) = G1;
        A(i, 1, 0) = G1;
        A(i, 0, 2) = G2;
        A(i, 2, 0) = G2;
        A(i, 0, 3) = -G0 - G1 - G2; // 2 FLOP
        A(i, 3, 0) = -G0 - G1 - G2; // 2 FLOP
        A(i, 1, 1) = G3;
        A(i, 1, 2) = G4;
        A(i, 2, 1) = G4;
        A(i, 1, 3) = -G1 - G3 - G4; // 2 FLOP
        A(i, 3, 1) = -G1 - G3 - G4; // 2 FLOP
        A(i, 2, 2) = G5;
        A(i, 2, 3) = -G2 - G4 - G5; // 2 FLOP
        A(i, 3, 2) = -G2 - G4 - G5; // 2 FLOP
        A(i, 3, 3) = G0 + 2 * G1 + 2 * G2 + G3 + 2 * G4 + G5; // 8 FLOP
        // TOTAL: 72 FLOP
    });
}


void benchmark(long N, long repeat) {
    repeat = (repeat > 0) ? repeat : std::max(6UL, 10000000UL / N);

    typedef Kokkos::RangePolicy<ExecSpace> RP;
    typedef Kokkos::View<Number* [3][3], MemLayout, MemSpace> Jacobian_t;
    typedef Kokkos::View<Number* [4][4], MemLayout, MemSpace> A_t;

    Jacobian_t jacobian("Jacobian", N);
    A_t A("A", N);

    Jacobian_t::HostMirror h_jacobian = Kokkos::create_mirror_view(jacobian);
    A_t::HostMirror h_A = Kokkos::create_mirror_view(A);

    for (long n = 0; n < N; n++) {
        h_jacobian(n, 0, 0) = 3;
        h_jacobian(n, 0, 1) = 1;
        h_jacobian(n, 0, 2) = 1;
        h_jacobian(n, 1, 0) = 1;
        h_jacobian(n, 1, 1) = 3;
        h_jacobian(n, 1, 2) = 1;
        h_jacobian(n, 2, 0) = 1;
        h_jacobian(n, 2, 1) = 1;
        h_jacobian(n, 2, 2) = 3;
    }


    // HOST TO DEVICE
    unsigned long repeat_h2d = 1;
    Kokkos::Timer timer;
    for (unsigned long i = 0; i < repeat_h2d; i++) {
        Kokkos::deep_copy(jacobian, h_jacobian);
    }
    Kokkos::fence();
    double T_host_to_device = timer.seconds() / repeat_h2d;


    // CALCULATE
    timer.reset();
    for (unsigned long i = 0; i < repeat; i++) {
        calculate<Jacobian_t, A_t, RP>(jacobian, A, N);
    }
    Kokkos::fence();
    double T_calculation = timer.seconds() / (double)repeat;


    // DEVICE TO HOST
    unsigned long repeat_d2h = 1;
    timer.reset();
    for (unsigned long i = 0; i < repeat_d2h; i++) {
        Kokkos::deep_copy(h_A, A);
    }
    Kokkos::fence();
    double T_device_to_host = timer.seconds() / repeat_d2h;

    // int p = 1e6;
    // for (int k = p; k < p + 10; k++) {
    //     std::cout << k << std::endl;
    //     for (int i = 0; i < 4; i++) {
    //         for (int j = 0; j < 4; j++) {
    //             std::cout << std::setw(11) << h_A(k, i, j) << " ";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << std::endl;
    // }

    double flop = 72 * N * 1e-9; // num of GFLOP
    double GFLOPS = flop / T_calculation; // GFLOPS (this is per second)
    double memory_in = 9 * (N * sizeof(Number) * 1e-9); // GB
    double bw_in = memory_in / T_host_to_device; // GB/s
    double memory_out = 16 * (N * sizeof(Number) * 1e-9); // GB
    double bw_out = memory_out / T_device_to_host; // GB/s
    double memory_io = (double)25 * ((double)N * (double)sizeof(Number) * (double)1e-9); // GB
    double bw_io = memory_io / T_calculation; // GB/s
    double mupd = 16 * N * 1e-6 / T_calculation; // MUPD

    std::cout << "Finite Element Tetahedral size: " << std::setw(10) << N << " ("
        << std::setw(12) << flop << " GFLOP, "
        << std::setw(12) << memory_in << " GB input, "
        << std::setw(12) << memory_out << " GB output) repeats: "
        << std::setw(6) << repeat << " "
        << std::setw(12) << T_calculation << " seconds or "
        << std::setw(10) << bw_io << " GB/s "
        << std::setw(10) << mupd << " MUPD/S "
        << std::setw(10) << GFLOPS << " GFLOP/s " 
        << "| BW in: "
        << std::setw(10) << bw_in << " GB/s, BW out: "
        << std::setw(10) << bw_out << " GB/s"
        << std::endl;
}


void check_args(int argc, char* argv[], long& n_min, long& n_max, long& repeat) {
    if (argc % 2 == 0) {
        std::cout << "Error, expected odd number of common line arguments" << std::endl;
        std::cout << "Expected line of the form" << std::endl;
        std::cout << "-min 100 -max 1000 -repeat 20" << std::endl;
        std::abort();
    }

    for (int i = 1; i < argc; i += 2) {
        std::string opt = argv[i];
        if (opt == "-min") {
            n_min = static_cast<long>(std::stod(argv[i + 1]));
        }
        else if (opt == "-max") {
            n_max = static_cast<long>(std::stod(argv[i + 1]));
        }
        else if (opt == "-repeat") {
            repeat = static_cast<long>(std::stod(argv[i + 1]));
        }
        else {
            std::cout << "Unknown option " << opt << " - ignored!" << std::endl;
        }
    }

    if (n_min < 1) {
        std::cout << "Expected positive size for -min argument, got " << n_min << std::endl;
        std::abort();
    }

    if (n_max < n_min) n_max = n_min + 1;

    if (std::is_same<Number, float>::value) {
        std::cout << "datatype: float | ";
    } else if (std::is_same<Number, double>::value) {
        std::cout << "datatype: double | ";
    } else {
        std::cout << "datatype: unknown | ";
    }

    if (std::is_same<MemLayout, Kokkos::LayoutLeft>::value) {
        std::cout << "mem_layout: Kokkos::LayoutLeft" << std::endl;
    } else if (std::is_same<MemLayout, Kokkos::LayoutRight>::value) {
        std::cout << "mem_layout: Kokkos::LayoutRight" << std::endl;
    } else {
        std::cout << "mem_layout: unknown" << std::endl;
    }
}
