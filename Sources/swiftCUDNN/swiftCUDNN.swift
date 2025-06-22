import cudnnFrontend

public func checkCudnnStatus(_ status: cudnnStatus_t, _ message: String = "") {
    if status.rawValue != CUDNN_STATUS_SUCCESS.rawValue {
        let errorString = String(cString: cudnnGetErrorString(status))
        fatalError("cuDNN error \(message): \(errorString)")
    }
}

func createTensor(dimensions: [Int64], dataType: cudnnDataType_t) -> cudnnBackendDescriptor_t? {
    var descriptor: cudnnBackendDescriptor_t?

    // Create tensor descriptor
    let status = cudnnBackendCreateDescriptor(
        CUDNN_BACKEND_TENSOR_DESCRIPTOR,
        &descriptor
    )
    checkCudnnStatus(status, "creating tensor descriptor")

    guard let desc = descriptor else { return nil }

    // Set data type
    var dataTypeValue = dataType
    checkCudnnStatus(
        cudnnBackendSetAttribute(
            desc,
            CUDNN_ATTR_TENSOR_DATA_TYPE,
            CUDNN_TYPE_DATA_TYPE,
            1,
            &dataTypeValue
        ),
        "setting data type"
    )

    // Set dimensions
    var dims = dimensions
    checkCudnnStatus(
        cudnnBackendSetAttribute(
            desc,
            CUDNN_ATTR_TENSOR_DIMENSIONS,
            CUDNN_TYPE_INT64,
            Int64(dimensions.count),
            &dims
        ),
        "setting dimensions"
    )

    // Calculate strides (row-major)
    var strides = [Int64](repeating: 1, count: dimensions.count)
    for i in (0..<dimensions.count-1).reversed() {
        strides[i] = strides[i+1] * dimensions[i+1]
    }

    checkCudnnStatus(
        cudnnBackendSetAttribute(
            desc,
            CUDNN_ATTR_TENSOR_STRIDES,
            CUDNN_TYPE_INT64,
            Int64(strides.count),
            &strides
        ),
        "setting strides"
    )

    // Set unique ID
    var uid = Int64(desc.hashValue)
    checkCudnnStatus(
        cudnnBackendSetAttribute(
            desc,
            CUDNN_ATTR_TENSOR_UNIQUE_ID,
            CUDNN_TYPE_INT64,
            1,
            &uid
        ),
        "setting unique ID"
    )

    // Set alignment
    var alignment = Int64(16)
    checkCudnnStatus(
        cudnnBackendSetAttribute(
            desc,
            CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,
            CUDNN_TYPE_INT64,
            1,
            &alignment
        ),
        "setting alignment"
    )

    // Finalize the descriptor
    checkCudnnStatus(
        cudnnBackendFinalize(desc),
        "finalizing tensor"
    )

    return desc
}

let MNIST_IMAGE_SIZE = 784  // 28x28 flattened
let HIDDEN_SIZE = 128
let NUM_CLASSES = 10
let BATCH_SIZE = 32

class MNISTLinearLayer {
    var handle: cudnnHandle_t?
    var inputTensor: cudnnBackendDescriptor_t?
    var weightTensor: cudnnBackendDescriptor_t?
    var biasTensor: cudnnBackendDescriptor_t?
    var outputTensor: cudnnBackendDescriptor_t?

    init() {
        // Create cuDNN handle
        checkCudnnStatus(cudnnCreate(&handle), "creating handle")
    }

    deinit {
        // Clean up descriptors
        if let input = inputTensor { cudnnBackendDestroyDescriptor(input) }
        if let weight = weightTensor { cudnnBackendDestroyDescriptor(weight) }
        if let bias = biasTensor { cudnnBackendDestroyDescriptor(bias) }
        if let output = outputTensor { cudnnBackendDestroyDescriptor(output) }
        if let h = handle { cudnnDestroy(h) }
    }

    func setupTensors() {
        // Input tensor: [batch_size, input_features]
        inputTensor = createTensor(
            dimensions: [Int64(BATCH_SIZE), Int64(MNIST_IMAGE_SIZE)],
            dataType: CUDNN_DATA_FLOAT
        )

        // Weight tensor: [output_features, input_features]
        weightTensor = createTensor(
            dimensions: [Int64(HIDDEN_SIZE), Int64(MNIST_IMAGE_SIZE)],
            dataType: CUDNN_DATA_FLOAT
        )

        // Bias tensor: [output_features]
        biasTensor = createTensor(
            dimensions: [Int64(HIDDEN_SIZE)],
            dataType: CUDNN_DATA_FLOAT
        )

        // Output tensor: [batch_size, output_features]
        outputTensor = createTensor(
            dimensions: [Int64(BATCH_SIZE), Int64(HIDDEN_SIZE)],
            dataType: CUDNN_DATA_FLOAT
        )
    }

    // Create a matmul operation descriptor
    func createMatmulOperation() -> cudnnBackendDescriptor_t? {
        var matmulDesc: cudnnBackendDescriptor_t?

        // Create matmul descriptor
        checkCudnnStatus(
            cudnnBackendCreateDescriptor(
                CUDNN_BACKEND_MATMUL_DESCRIPTOR,
                &matmulDesc
            ),
            "creating matmul descriptor"
        )

        guard let desc = matmulDesc else { return nil }

        // Set compute type
        var computeType = CUDNN_DATA_FLOAT
        checkCudnnStatus(
            cudnnBackendSetAttribute(
                desc,
                CUDNN_ATTR_MATMUL_COMP_TYPE,
                CUDNN_TYPE_DATA_TYPE,
                1,
                &computeType
            ),
            "setting compute type"
        )

        // Finalize
        checkCudnnStatus(
            cudnnBackendFinalize(desc),
            "finalizing matmul"
        )

        return desc
    }
}