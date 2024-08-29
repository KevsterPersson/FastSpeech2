import numpy as np
import torch
import yaml
import onnx

from model import FastSpeech2
from synthesize import preprocess_english

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def read_config(p_config_path, m_config_path, t_config_path):
    with open(p_config_path, 'r') as f:
        preprocess_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(m_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(t_config_path, 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config, train_config)

    return configs


def preprocessing_model(preprocess_config, model_config):
    model = FastSpeech2(preprocess_config, model_config).to(device)

    model_path = "output/ckpt/LJSpeech/900000.pth.tar"
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt["model"])

    model.eval()
    model.requires_grad_ = False

    return model


def to_device(data, device):
    (ids, raw_texts, speakers, texts, src_lens, max_src_len) = data

    speakers = torch.tensor(speakers).to(device)
    texts = torch.tensor(texts).to(device)
    src_lens = torch.tensor(src_lens).to(device)

    return ids, raw_texts, speakers, texts, src_lens, max_src_len


def preprocessing_text(raw_text, speaker_id, preprocess_config):
    ids = raw_texts = [raw_text[:100]]
    speakers = torch.tensor([speaker_id])
    texts = torch.tensor([preprocess_english(raw_text, preprocess_config)])
    text_lens = torch.tensor([len(texts[0])])
    batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    for batch in batchs:
        batch = to_device(batch, device)

    return batch


def fs2onnx(model, dummy_input, input_names, output_names, output_path):
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=11,
        input_names=input_names,
        output_names=output_names,
    )


'''
            dynamic_axes={
            "texts": {0: "batch_size"},
            "speakers": {0: "batch_size"},
            "src_lens": {0: "batch_size"},
            "max_src_len": {0: "batch_size"}
        }
'''


if __name__ == '__main__':
    p_config_path = 'config/LJSpeech/preprocess.yaml'
    m_config_path = 'config/LJSpeech/model.yaml'
    t_config_path = 'config/LJSpeech/train.yaml'

    configs = read_config(p_config_path, m_config_path, t_config_path)
    preprocess_config, model_config, train_config = configs

    model = preprocessing_model(preprocess_config, model_config)

    raw_text = "hello world"
    speaker_id = 0
    batch = preprocessing_text(raw_text, speaker_id, preprocess_config)
    ids, raw_texts, speakers, texts, src_lens, max_src_len = batch

    dummy_input = (speakers,
                   texts,
                   src_lens,
                   max_src_len,
                   )
    input_names = ['speakers',
                   'texts',
                   'src_lens',
                   'max_src_len',
                    ]
    output_names = ["output",
                    "postnet_output",
                    "p_predictions",
                    "e_predictions",
                    "log_d_predictions",
                    "d_rounded",
                    "src_masks",
                    "mel_masks",
                    "src_lens",
                    "mel_lens"
                    ]
    output_path = "fs2onnx.onnx"
    fs2onnx(model, dummy_input, input_names, output_names, output_path)
