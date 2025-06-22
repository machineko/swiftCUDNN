import Testing
@testable import swiftCUDNN
import cudnnFrontend
import CxxStdlib

@Test func example() async throws {
    cudaSetDevice(0)
    // var graph: cudnnBackendDescriptor_t = .init(bitPattern: 0)
    let tensor = createTensor(dimensions: [1,3,24,24], dataType: .init(2))
    let dims: [Int64] = [2, 2]
    let strides: [Int64] = [2, 1]

    // Set properties using the builder
    let aDesc = cudnn_frontend.createTensorDesc("elo", [2,2], [2,1], .init(rawValue: 2)!)

    var graph = cudnn_frontend.graph.Graph()
    let A = graph.tensor(aDesc)
    var errCode = graph.validate()
    var handle: cudnnHandle_t?
    checkCudnnStatus(cudnnCreate(&handle), "cudnnCreate")

    #expect(errCode.code == .OK)

}
