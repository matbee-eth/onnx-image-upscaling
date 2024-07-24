from onnxruntime.quantization import QuantFormat, QuantType, quantize_dynamic
import argparse
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_model", required=True, help="input model")
    parser.add_argument("--output_model", required=True, help="output model")
    parser.add_argument(
        "--quant_format",
        default=QuantFormat.QDQ,
        type=QuantFormat.from_string,
        choices=list(QuantFormat),
    )
    parser.add_argument("--per_channel", default=False, type=bool)
    args = parser.parse_args()
    return args
  
def main():
    args = get_args()
    input_model_path = args.input_model
    output_model_path = args.output_model

    # Calibrate and quantize model
    # Turn off model optimization during quantization
    # quantize_static(
    #     input_model_path,
    #     output_model_path,
    #     dr,
    #     quant_format=args.quant_format,
    #     per_channel=args.per_channel,
    #     weight_type=QuantType.QInt8,
    # )
    quantize_dynamic(input_model_path,
                  output_model_path,
                  weight_type=QuantType.QInt8)
    print("Calibrated and quantized model saved.")

if __name__ == "__main__":
    main()