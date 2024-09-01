from typing import Tuple, Iterator

import fire
import torch
import numpy as np

from transformers import AutoTokenizer, AutoModelForCausalLM

from language_modeling_is_compression import arithmetic_coder, constants, utils


def compress(
    data,
    model,
) -> Tuple[bytes, int]:
    """Compresses the `data` using arithmetic coding and a pretrained model.

    Args:
      data: The data to be compressed.
      model: The pretrained model.

    Returns:
      The compressed data.
    """
    sequence_array = data[0].cpu().numpy()

    log_probs = list()
    for i in range(1, len(sequence_array)):
        inputs = torch.tensor([sequence_array[:i]], dtype=data.dtype).to(next(model.parameters()).device)
        subsequence_probs = model(
            inputs
        ).logits
        probs = torch.nn.functional.softmax(subsequence_probs, dim=-1)
        log_probs.append(probs[0, -1].detach().cpu().numpy())
    probs = np.vstack(log_probs)
    # probs = np.exp(log_probs)

    output = list()
    encoder = arithmetic_coder.Encoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        output_fn=output.append,
    )
    for pdf, symbol in zip(probs, sequence_array):
        encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
    encoder.terminate()

    compressed_bits = ''.join(map(str, output))
    compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

    return compressed_bytes, num_padded_bits

def decompress(
    data,
    num_padded_bits: int,
    model,
    uncompressed_length: int,
) -> bytes:
    """Decompresses the `data` using arithmetic coding and a pretrained model.

    See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

    Args:
      data: The data to be decompressed.
      model: The pretrained model.

    Returns:
      The decompressed data.
    """

    data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

    # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
    # from the compressed input and returns `None` when the input is exhausted.
    def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
        try:
            return int(next(bit_sequence))
        except StopIteration:
            return None

    decoder = arithmetic_coder.Decoder(
        base=constants.ARITHMETIC_CODER_BASE,
        precision=constants.ARITHMETIC_CODER_PRECISION,
        input_fn=_input_fn,
    )
    # We need a dummy token because the language model right-shifts the sequence
    # by one when computing the conditional probabilities. Concretely, at every
    # step, we need the `pdf` of the next token given all currently decompressed
    # tokens, but without a dummy token, the last `pdf` would be that of the last
    # already decompressed token. The value of the dummy token is irrelevant.
    sequence_array = np.array([[2]], dtype=np.int64)

    for idx in range(uncompressed_length-1):
        dummy_input_ids = torch.tensor(sequence_array, dtype=torch.int64).to(next(model.parameters()).device)
        subsequence_probs = model(
            dummy_input_ids
        ).logits
        probs = torch.nn.functional.softmax(subsequence_probs, dim=-1)
        probs = probs[0, -1].detach().cpu().numpy()
        token = decoder.decode(
            utils.normalize_pdf_for_arithmetic_coding(probs)
        )
        sequence_array = np.concatenate((sequence_array, [[token]]), axis=1)

    # Remove the dummy token and convert to bytes.
    return sequence_array[0,:]

def main(
        text: str = "Hello, my name is Niels",
        model_name_or_path: str = "facebook/opt-350m",
        device: str = "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    input_ids = tokenizer([text], return_tensors="pt").to(device).input_ids
    print(input_ids)
    compressed, num_padded_bits = compress(input_ids, model)
    print(compressed.hex())
    decompressed = decompress(
            compressed,
            num_padded_bits,
            model,
            len(input_ids[0])
        )
    print(decompressed)
    reconstructed = tokenizer.decode(decompressed)
    print(reconstructed)



if __name__ == "__main__":
    fire.Fire(main)