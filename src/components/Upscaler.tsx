/* eslint-disable @typescript-eslint/no-explicit-any */
// File: src/components/Upscaler.jsx

import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/webgpu";

// Ensure ONNX Runtime Web WASM files are served correctly
// import wasmPath from "onnxruntime-web/dist/ort-wasm.wasm?url";
// import wasmThreadedPath from "onnxruntime-web/dist/ort-wasm-threaded.wasm?url";
// import wasmSimdPath from "onnxruntime-web/dist/ort-wasm-simd.wasm?url";

ort.env.wasm.wasmPaths = "/onnxruntime-web/";

class UpscalerONNXRuntimeWeb {
  private session: ort.InferenceSession | null;
  private inputName: string;
  private outputName: string;
  private executionProvider: "webgpu" | "wasm" | "cpu";
  private expectedInputShape: number[];

  constructor() {
    this.session = null;
    this.inputName = "input"; // Set to match the model's input name
    this.outputName = "output"; // Set to match the model's output name
    this.executionProvider = "webgpu";
    this.expectedInputShape = [1, 3, -1, -1]; // Default shape, will be updated after model load
  }

  async loadModel(modelUrl: string): Promise<void> {
    const providers = ["webgpu", "wasm", "cpu"];
    for (const provider of providers) {
      try {
        await this.tryLoadModel(
          modelUrl,
          provider as "webgpu" | "wasm" | "cpu"
        );
        return;
      } catch (error) {
        console.warn(`Failed to load model with ${provider}:`, error);
      }
    }
    throw new Error("Failed to load model with any available provider");
  }

  private async tryLoadModel(
    modelUrl: string,
    provider: "webgpu" | "wasm" | "cpu"
  ): Promise<void> {
    const options: ort.InferenceSession.SessionOptions = {
      executionProviders: [
        {
          name: "webgpu",
          preferredLayout: "NCHW",
        },
      ],
      graphOptimizationLevel: "all",
      enableProfiling: false,
      enableMemPattern: false,
      enableCpuMemArena: false,
      logSeverityLevel: 3 /*disable logs like 
         [W:onnxruntime:, session_state.cc:1166 VerifyEachNodeIsAssignedToAnEp] Some nodes were not assigned to the preferred execution providers which may or may not have an negative impact on performance. e.g. ORT explicitly assigns shape related ops to CPU to improve perf.
         [W:onnxruntime:, session_state.cc:1168 VerifyEachNodeIsAssignedToAnEp] Rerunning with verbose output on a non-minimal build will show node assignments.
         */,
      extra: {
        session: {
          disable_prepacking: "1",
          use_device_allocator_for_initializers: "1",
          use_ort_model_bytes_directly: "1",
          use_ort_model_bytes_for_initializers: "1",
          disable_cpu_ep_fallback: "0",
        },
      },
      freeDimensionOverrides: { batch_size: 1 },
    };

    if (provider === "wasm") {
      ort.env.wasm.numThreads = 4;
      ort.env.wasm.simd = true;
    }

    this.session = await ort.InferenceSession.create(modelUrl, options);

    console.log("Model Inputs:", this.session.inputNames);
    console.log("Model Outputs:", this.session.outputNames);
    this.executionProvider = provider;
    console.log(`Model loaded successfully with ${provider}`);
  }

  private getNextProvider(): "webgpu" | "wasm" | "cpu" | null {
    const providers = ["webgpu", "wasm", "cpu"];
    const currentIndex = providers.indexOf(this.executionProvider);
    return (
      (providers[currentIndex + 1] as "webgpu" | "wasm" | "cpu" | null) || null
    );
  }

  async upscale(imageData: ImageData): Promise<ImageData> {
    if (!this.session) {
      throw new Error("Model not loaded");
    }

    const tensor = this.preprocessImage(imageData);
    const feeds: Record<string, ort.Tensor> = { [this.inputName]: tensor };

    try {
      console.log(`Running inference with ${this.executionProvider}`);
      const outputMap = await this.session.run(feeds);
      const outputTensor = outputMap[this.outputName] as ort.Tensor;
      console.log("Output tensor shape:", outputTensor.dims);
      return this.postprocessOutput(
        outputTensor,
        imageData.width,
        imageData.height
      );
    } catch (error) {
      console.error(
        `Error during upscale with ${this.executionProvider}:`,
        error
      );

      if (this.executionProvider !== "cpu") {
        console.log("Attempting to fall back to next provider");
        const nextProvider = this.getNextProvider();
        if (nextProvider) {
          // We need to get the modelUrl here. Since we don't store it, we'll need to pass it from the component.
          // For now, let's throw an error indicating that we can't fallback automatically.
          throw new Error(
            "Fallback required. Please reload the model with a different provider."
          );
        }
      }
      throw error;
    }
  }

  private preprocessImage(imageData: ImageData): ort.Tensor {
    const { data, width: W, height: H } = imageData;
    const float32Array = new Float32Array(W * H * 3);

    for (let c = 0; c < 3; c++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          const sourceIdx = (h * W + w) * 4 + c;
          const targetIdx = c * H * W + h * W + w;
          float32Array[targetIdx] = data[sourceIdx] / 255;
        }
      }
    }

    return new ort.Tensor("float32", float32Array, [1, 3, H, W]);
  }

  private postprocessOutput(
    outputTensor: ort.Tensor,
    originalWidth: number,
    originalHeight: number
  ): ImageData {
    const data = outputTensor.data as Float32Array;
    console.log("Last 100 values of data:", data.slice(-100));
    const [, , height, width] = outputTensor.dims;

    console.log("Output tensor shape:", outputTensor.dims);
    console.log("Output data length:", data.length);
    console.log("Original dimensions:", originalWidth, "x", originalHeight);

    console.log("Output tensor shape:", outputTensor.dims);
    console.log("Output data length:", data.length);
    console.log("Original dimensions:", originalWidth, "x", originalHeight);
    console.log("New dimensions:", width, "x", height);

    // Calculate the scale factor
    const scaleX = width / originalWidth;
    const scaleY = height / originalHeight;

    if (Math.abs(scaleX - scaleY) > 0.01) {
      console.warn(
        `Uneven scaling factors: ${scaleX.toFixed(
          2
        )}x horizontally, ${scaleY.toFixed(2)}x vertically`
      );
    }

    const imageData = new ImageData(width, height);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const imageDataIndex = (y * width + x) * 4;
        for (let c = 0; c < 3; c++) {
          // NCHW format
          const tensorIndex = c * height * width + y * width + x;
          const value = data[tensorIndex];
          // console.log(`Value:`, value);
          imageData.data[imageDataIndex + c] = Math.max(
            0,
            Math.min(255, Math.round(value * 255))
          );
        }
        imageData.data[imageDataIndex + 3] = 255; // Alpha channel
      }
    }

    return imageData;
  }
}

interface UpscalerProps {
  modelUrl: string;
}

const Upscaler: React.FC<UpscalerProps> = ({ modelUrl }) => {
  const [upscaler, setUpscaler] = useState<UpscalerONNXRuntimeWeb | null>(null);
  const [inputImage, setInputImage] = useState<HTMLImageElement | null>(null);
  const [outputImage, setOutputImage] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [currentProvider, setCurrentProvider] = useState<
    "webgpu" | "wasm" | "cpu"
  >("webgpu");

  const inputCanvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const initUpscaler = async () => {
      try {
        setIsLoading(true);
        const newUpscaler = new UpscalerONNXRuntimeWeb();
        await newUpscaler.loadModel(modelUrl);
        setUpscaler(newUpscaler);
        setError(null);
      } catch (err) {
        setError(
          `Failed to load the model: ${
            err instanceof Error ? err.message : String(err)
          }`
        );
      } finally {
        setIsLoading(false);
      }
    };

    initUpscaler();
  }, [modelUrl, currentProvider]);

  const handleImageUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();

    reader.onload = (e: ProgressEvent<FileReader>) => {
      const img = new Image();
      img.onload = () => {
        setInputImage(img);
        const canvas = inputCanvasRef.current;
        if (canvas) {
          canvas.width = img.width;
          canvas.height = img.height;
          const ctx = canvas.getContext("2d");
          ctx?.drawImage(img, 0, 0);
        }
      };
      img.src = e.target?.result as string;
    };

    reader.readAsDataURL(file);
  };

  const handleUpscale = async () => {
    if (
      !upscaler ||
      !inputImage ||
      !inputCanvasRef.current ||
      !outputCanvasRef.current
    )
      return;

    try {
      setIsLoading(true);
      setError(null);

      const inputCanvas = inputCanvasRef.current;
      const ctx = inputCanvas.getContext("2d");
      if (!ctx) throw new Error("Failed to get canvas context");

      const imageData = ctx.getImageData(
        0,
        0,
        inputCanvas.width,
        inputCanvas.height
      );

      const upscaledImageData = await upscaler.upscale(imageData);

      const outputCanvas = outputCanvasRef.current;
      outputCanvas.width = upscaledImageData.width;
      outputCanvas.height = upscaledImageData.height;
      const outputCtx = outputCanvas.getContext("2d");
      if (!outputCtx) throw new Error("Failed to get output canvas context");

      outputCtx.putImageData(upscaledImageData, 0, 0);

      setOutputImage(outputCanvas.toDataURL());
    } catch (err) {
      if (err instanceof Error && err.message.includes("Fallback required")) {
        const nextProvider = getNextProvider(currentProvider);
        if (nextProvider) {
          setCurrentProvider(nextProvider);
          setError(
            `Falling back to ${nextProvider}. Please try upscaling again.`
          );
        } else {
          setError("Failed to upscale with all available providers.");
        }
      } else {
        setError(
          `Failed to upscale image: ${
            err instanceof Error ? err.message : String(err)
          }`
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  const getNextProvider = (
    current: "webgpu" | "wasm" | "cpu"
  ): "webgpu" | "wasm" | "cpu" | null => {
    const providers = ["webgpu", "wasm", "cpu"];
    const currentIndex = providers.indexOf(current);
    return (
      (providers[currentIndex + 1] as "webgpu" | "wasm" | "cpu" | null) || null
    );
  };

  return (
    <div>
      <input
        type='file'
        accept='image/*'
        onChange={handleImageUpload}
        disabled={isLoading}
      />
      <button
        onClick={handleUpscale}
        disabled={!inputImage || !upscaler || isLoading}
      >
        Upscale
      </button>
      {isLoading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}
      <div>
        <h3>Input Image</h3>
        <canvas ref={inputCanvasRef} />
      </div>
      <div>
        <h3>Output Image</h3>
        <canvas ref={outputCanvasRef} />
      </div>
    </div>
  );
};

export default Upscaler;
