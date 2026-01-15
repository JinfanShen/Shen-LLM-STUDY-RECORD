import json
from datasets import DownloadManager, DatasetInfo
import datasets as ds

class CMCRC2018TRIAL(ds.GeneratorBasedBuilder):
    def _info(self) ->DatasetInfo:
        """info
            对数据的字段进行定义    
        """
        return ds.DatasetInfo( 
            description="CMCRC2018 TRIAL",
            features=ds.Features({
                "id": ds.Value("string"),
                "context": ds.Value("string"),
                "question": ds.Value("string"),
                "answers": ds.features.Sequence(
                    {
                        "text": ds.Value("string"),
                        "answer_start": ds.Value("int32")
                    }
                )
            })
        )
    
    def _split_generators(self, dl_manager:DownloadManager):
        return [ds.SplitGenerator(name=ds.Split.TRAIN, gen_kwargs={"filepath": "./cmrc2018_trial.json"})]
    
    def _generate_examples(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
            for example in data['data']:
                for paragraph in example['paragraphs']:
                    context = paragraph['context']
                    for qa in paragraph['qas']:
                        question = qa['question'].strip()
                        id_ = qa['id']
                        answer_starts = [answer['answer_start'] for answer in qa['answers']]
                        answers = [answer['text'].strip() for answer in qa['answers']]

                        yield id_, {
                            "id": id_,
                            "context": context,
                            "question": question,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers
                            }
                        }


def load_cmrc2018(filepath):
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    for example in data['data']:
        for paragraph in example['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                examples.append({
                    "id": qa['id'],
                    "context": context,
                    "question": qa['question'].strip(),
                    "answers": {
                        "text": [a['text'].strip() for a in qa['answers']],
                        "answer_start": [a['answer_start'] for a in qa['answers']]
                    }
                })
    
    return ds.Dataset.from_list(examples)

if __name__ == "__main__":
    import os
    dataset = load_cmrc2018(os.path.dirname(os.path.abspath(__file__)) + "/cmrc2018_trial.json")
    print(dataset)
