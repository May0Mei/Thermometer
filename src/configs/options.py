import argparse

def str2bool(string):
    return string.lower() in ['yes', 'true', 't', 1]

def process_args():
    parser = argparse.ArgumentParser(description='configuration')
    parser.register('type', 'bool', str2bool)
    # general options
    parser.add_argument('--root_path', type=str, help='the root path where the model and data are stored')
    parser.add_argument('--training_seed', type=int, default=0, help='seed to ensure consistency')
    parser.add_argument('--seed_list', nargs='*', dest='seed_list', help='list of seeds for evaluation',type=int)
    # dataset options
    parser.add_argument('--benchmark', type=str, default='mmlu',
                        help='the benchmark of datasets for training the thermometer model',
                        choices=['bigbench', 'mmlu', 'mrqa'])
    parser.add_argument('--test_dataset', type=str, help='the target datasets to test the performance of thermometer model')
    parser.add_argument('--dataset', type=str, default='anatomy', help='the dataset name for loading data')
    parser.add_argument('--dataset_split', type=str, default='train', help='train | val | test')
    # Thermometer hyper-parameter
    parser.add_argument('--thermometer_input_size', type=int, default=2048)
    parser.add_argument('--thermometer_hidden_size', type=int, default=256)
    parser.add_argument('--thermometer_lr', type=float, default=1e-3)
    parser.add_argument('--lambda_reg', type=float, default=1e-2, help="hyper-parameter for regularizer")
    parser.add_argument('--Gamma_k', type=float, default=1.25, help="Gamma regularizer scale parameter")
    parser.add_argument('--Gamma_theta', type=float, default=4.0, help="Gamma regularizer shape parameter ")
    parser.add_argument('--steps', type=int, default=5000, help="the number of steps of gradient update")
    # base LLM model options
    parser.add_argument('--model_type', default='encoder_decoder', help='the model type',
                        choices=['encoder_decoder', 'decoder_only'])
    parser.add_argument('--model_name', default='t5-base', help='the base LLM',
                        choices = ['Llama-2-7b-chat-hf', 'flan-t5-base', 'flan-t5-large','flan-t5-xl', 'flan-t5-xxl',
                                  'Llama-2-7b-chat-hf'])
    parser.add_argument('--HG_token', type=str, help='the access token on huggingface')
    parser.add_argument('--max_source_length', type=int, default=256, help="for encoder-decoder model")
    parser.add_argument('--max_target_length', type=int, default=128, help="for encoder-decoder model")
    parser.add_argument('--max_length', type=int, default=2048, help="for decoder-only model")
    parser.add_argument("--inference_batch_size", default=2, type=int,  help='the batch_size to inference base LLM')
    # optimization options
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument("--eval_batch_size", default=128, type=int)

    args = parser.parse_args()
    return args


