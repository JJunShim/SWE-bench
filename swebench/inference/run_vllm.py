import json
import logging
import tiktoken
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from datasets import load_from_disk, load_dataset
from tqdm.auto import tqdm

from swebench.inference.make_datasets.utils import extract_diff

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CONFIG = {
    "api_key": "EMPTY",
    "api_base": "http://localhost:{port}/v1",
    "model": "microsoft/Phi-4-reasoning-plus",
    "bench": "SWE-bench/SWE-bench",
    "temperature": 0.8,
    "top_p": 0.95,
    "max_tokens": 32768
}


def initialize_client(base) -> OpenAI:
    """OpenAI 클라이언트를 초기화합니다."""
    return OpenAI(
        base_url=base,
        api_key=CONFIG["api_key"],
        timeout=1800
    )


def count_tokens(text: str, encoder: str = "o200k_base") -> int:
    encoding = tiktoken.get_encoding(encoder)
    return len(encoding.encode(text))


def get_output_file(
    output_dir,
    model_name_or_path,
    dataset_path,
    split,
    temperature,
    top_p,
    min_len,
    max_len,
    shard_id,
    num_shards,
):
    suffix = ""
    if min_len is not None:
        suffix += f"__min-{min_len}"
    if max_len is not None:
        suffix += f"__max-{max_len}"
    if shard_id is not None and num_shards is not None:
        suffix += f"__shard-{shard_id}-{num_shards}"
    if Path(dataset_path).exists():
        dset_nickname = Path(dataset_path).name + "__" + split
    else:
        dset_nickname = dataset_path.replace("/", "__") + "__" + split
    if Path(model_name_or_path).exists():
        if "checkpoint" in Path(model_name_or_path).name:
            model_nickname = (
                Path(model_name_or_path).parent.name
                + "__"
                + Path(model_name_or_path).name
            )
        else:
            model_nickname = Path(model_name_or_path).name
    else:
        model_nickname = model_name_or_path.replace("/", "__")
    output_file = Path(
        output_dir,
        dset_nickname
        + "__"
        + model_nickname
        + "__temp-"
        + str(temperature)
        + "__top-p-"
        + str(top_p)
        + suffix
        + ".jsonl",
    )
    if not output_file.parent.exists():
        output_file.parent.mkdir(
            parents=True, exist_ok=True
        )  # exists_ok=True for parallel
    return output_file


def load_data(
    dataset_path,
    split,
):
    logger.info(f"Loading dataset from {dataset_path}")
    if not Path(dataset_path).exists():
        dataset = load_dataset(dataset_path, split=split)
    elif Path(dataset_path, split).exists():
        dataset = load_from_disk(Path(dataset_path) / split)
    else:
        dataset = load_dataset(dataset_path)[split]

    return dataset


def generate(
    client,
    dataset,
    temperature,
    top_p,
    fileobj,
    model_name_or_path,
    max_len,
):
    for ix, instance in enumerate(tqdm(dataset, desc="Generating patches")):
        max_tokens = (
            max_len if max_len is not None else CONFIG["max_tokens"]
        ) - int(count_tokens(instance["text"])*1.2)

        if max_tokens <= 0:
            logger.warning(
                f"Skipping instance {ix} with text length {len(instance['text'])} "
                + f"as it exceeds the maximum token limit of {max_len}."
            )
            content = None
            diff = None
        else:
            start = datetime.now()
            fail_count = 0
            while True:
                try:
                    output = client.chat.completions.create(
                        model=model_name_or_path,
                        reasoning_effort="high",
                        messages=[
                            {"role": "user", "content": instance["text"]}],
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens
                    )
                    break
                except Exception as e:
                    logger.exception(e)
                    print(f"failed on {ix}")
                    fail_count += 1
                    if fail_count >= 3:
                        raise ValueError("too many failures")

            total_len = output.usage.total_tokens
            new_len = output.usage.completion_tokens
            logger.info(
                f"Generated {new_len} tokens ({total_len} total) in {(datetime.now() - start).total_seconds()} "
                + f"seconds (speed: {new_len / (datetime.now() - start).total_seconds()} tps)"
            )

            content = output.choices[0].message.content
            logger.info(content[:200])
            diff = extract_diff(content)

        res = {
            "instance_id": instance["instance_id"],
            "full_output": content,
            "model_patch": diff,
            "model_name_or_path": model_name_or_path,
        }
        print(json.dumps(res), file=fileobj, flush=True)


def main(
    model_name_or_path,
    dataset_path,
    split,
    index,
    temperature,
    top_p,
    output_dir,
    min_len,
    max_len,
    shard_id,
    num_shards,
):
    if shard_id is not None and num_shards is None:
        raise ValueError("num_shards must be specified with shard_id")
    if shard_id is None and num_shards is not None:
        raise ValueError("shard_id must be specified with num_shards")

    output_file = get_output_file(
        output_dir=output_dir,
        model_name_or_path=model_name_or_path,
        dataset_path=dataset_path,
        split=split,
        temperature=temperature,
        top_p=top_p,
        min_len=min_len,
        max_len=max_len,
        shard_id=shard_id,
        num_shards=num_shards,
    )
    logger.warning(f"output_file: {output_file}")

    base = CONFIG["api_base"].format(port=8000)
    client = initialize_client(base)
    dataset = load_data(
        dataset_path=dataset_path,
        split=split
    )
    subset = dataset.select(range(index, dataset.num_rows))

    with open(output_file, "a") as f:
        generate(
            client=client,
            dataset=subset,
            temperature=temperature,
            top_p=top_p,
            fileobj=f,
            model_name_or_path=model_name_or_path,
            max_len=max_len,
        )
    logger.info("Done")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model or hf model name",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset or hf dataset name",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to use"
    )
    parser.add_argument("--index", type=int, default=0, help="데이터셋 처리 시작 인덱스")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--min_len",
        type=int,
        default=None,
        help="Minimum length of input sequences to include",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=None,
        help="Maximum length of input sequences to include",
    )
    parser.add_argument(
        "--shard_id", type=int, default=None, help="ID of the shard to load"
    )
    parser.add_argument(
        "--num_shards", type=int, default=None, help="Total number of shards"
    )
    args = parser.parse_args()
    main(**vars(args))
