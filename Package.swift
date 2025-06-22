// swift-tools-version: 6.1
import Foundation
import PackageDescription

let packageDir = URL(fileURLWithPath: #file).deletingLastPathComponent().path
#if os(Windows)
    let cuPath: String = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.5"
    let cuLibPath = "-L\(cuPath)\\lib\\x64"
    let cuIncludePath = "-I\(cuPath)\\include"
    
#elseif os(Linux)
    let cuPath = ProcessInfo.processInfo.environment["CUDA_HOME"] ?? "/usr/local/cuda"
    let cuLibPath = "-L\(cuPath)/lib64"
    let cuIncludePath = "-I\(cuPath)/include"
#else
    fatalError("OS not supported \(os)")
#endif
let package = Package(
    name: "swiftCUDNN",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "swiftCUDNN",
            targets: ["swiftCUDNN"]),
        .library(
            name: "cudnnFrontend",
            targets: ["cudnnFrontend"]),
    ],
    targets: [
        .target(
        name: "cudnnFrontend",
        publicHeadersPath: "include",
        cxxSettings: [
            .headerSearchPath(cuIncludePath),
            .headerSearchPath("include"),
            // .define("CUDNN_FRONTEND_SKIP_JSON_LIB", to: "1"),
            // .define("_GLIBCXX_USE_CXX11_ABI", to: "0"),
            .define("CUDNN_VERSION", to: "9200"),
            .unsafeFlags([
                    "-std=c++17",
                    "-Wno-attributes",
                    "-Wno-unused-function",
                    "-Wno-deprecated-declarations",
                    "-Wno-error"
                ])


        ],
        linkerSettings: [
            .unsafeFlags([
                cuLibPath,
            ]),
            .linkedLibrary("cudnn"),
            .linkedLibrary("cudart"),
            .linkedLibrary("cuda"),
            .linkedLibrary("cublas"),
            .linkedLibrary("cublasLt")

        ]
        ),
        .target(
            name: "swiftCUDNN",
            dependencies: ["cudnnFrontend"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                .unsafeFlags(["-Xcc", cuIncludePath, "-Xlinker", cuLibPath]),

            ],
        ),
        .testTarget(
            name: "swiftCUDNNTests",
            dependencies: ["swiftCUDNN"],
            swiftSettings: [
                .interoperabilityMode(.Cxx),
                // .unsafeFlags(["-Xcc", cuIncludePath, "-Xlinker", cuLibPath]),

            ],
        ),
    ],
    cxxLanguageStandard: .cxx17
)
