import os
import json
import jsonlines
import random
from string import punctuation

class ODQATextData:
    def __init__(self, split, args, eval_only=False, traindata_obj=None):
        self.dir_path = args.dataset_dir
        self.split = split
        self.load()
        self.build_ref()
        if not eval_only:
            self.traindata_obj = traindata_obj
            flattened_gpt3_gen_filename = args.flattened_gen_data
            retrieve_filename = args.retrieve_filename
            clusters_filename = args.clusters_filename
            clusters_retrieve_filename = args.clusters_retrieve_filename
            fixed_sample_file = args.fixed_sample_file
            self.fixed_samples_func(flattened_gpt3_gen_filename, retrieve_filename, clusters_filename, clusters_retrieve_filename, fixed_sample_file)

    def load(self):
        filename = os.path.join(self.dir_path, self.split + '.jsonl')
        self.data = []
        with open(filename) as f:
            for item in jsonlines.Reader(f):
                self.data.append(item)

    def get_by_idx(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def build_ref(self):
        self.ref = []
        for i in range(len(self)):
            item = self.get_by_idx(i)
            self.ref.append({'answers': {'text': item['answer']}, 'id': i})

    def compute_metric(self, raw_pred, num=-1):
        """
        Nhận vào dự đoán thô, sau đó tính toán các chỉ số F1 và EM
        """
        preds = [{'prediction_text': text, 'id': i} for i, text in enumerate(raw_pred)]
        if num < 0:
            res, rw, all_em, all_f1 = self._compute_metric(preds, self.ref)
        else:
            res, rw, all_em, all_f1 = self._compute_metric(preds, self.ref[:num])
        return res, rw, all_em, all_f1

    def _compute_metric(self, preds, references):
        """
        Hàm thực hiện tính toán các chỉ số F1 và Exact Match
        preds: Dự đoán từ mô hình
        references: Dữ liệu tham chiếu
        """
        # Tính toán F1 và Exact Match
        f1_score = 0  # Tính F1 ở đây
        em_score = 0  # Tính EM ở đây
        # Cần thực hiện các phép tính cho F1 và Exact Match
        return {"f1_score": f1_score, "em_score": em_score}, None, None, None

    def fixed_samples_func(self, flattened_gpt3_gen_filename, retrieve_filename, clusters_filename,
                           clusters_retrieve_filename, fixed_sample_file=None):
        if fixed_sample_file is not None:
            with open(fixed_sample_file) as f:
                self.fixed_samples = json.load(f)
        else:
            self.fixed_samples = []
        with open(flattened_gpt3_gen_filename) as f:
            self.flattened_fixed_samples_by_gpt3 = json.load(f)
        with open(clusters_filename) as f:
            self.clusters_res = json.load(f)
        with open(clusters_retrieve_filename) as f:
            self.clusters_retrieve_res = json.load(f)

        if retrieve_filename is not None:
            with open(retrieve_filename) as f:
                self.retrieve_res = json.load(f)
        else:
            assert self.clusters_retrieve_res is not None

    def recall_select_answer_template(self, num_sample, source='traindata', sid=-1, realqid=-1, seed=-1,
                                      with_restrict="inst", instruction_way=-1, demo_way=-1, in_cot_way=False,
                                      with_short_question_marker=False, with_short_answer_marker=False):
        """
        Một ví dụ phương thức để chọn câu trả lời dựa trên yêu cầu của bạn
        :param num_sample: Số lượng mẫu cần chọn
        :param source: Nguồn dữ liệu (ví dụ: 'traindata', 'gpt3gen')
        :param seed: Seed cho việc chọn mẫu ngẫu nhiên
        :param instruction_way: Cách viết hướng dẫn (ví dụ: 0, 1, 2...)
        :param demo_way: Cách xử lý dữ liệu (ví dụ: 4)
        :param in_cot_way: Cách sử dụng chuỗi văn bản (có hay không)
        :return: Output câu trả lời đã chọn
        """

        instruction_pool = {
            0: "Given a question, first write a short passage that can answer it, then choose the evidence sentence from the passage, and finally answer the question based on this sentence.",
            1: "Given a question, recite a short passage about it, then answer it.",
            2: "Given a question, recite a piece of evidence about it, then answer it.",
            -1: "Answer the questions."
        }
        instruction = instruction_pool.get(instruction_way, "")

        # Lựa chọn câu trả lời từ dữ liệu đã cho
        output = instruction + '\n' if instruction else ''

        # Chọn mẫu ngẫu nhiên từ dữ liệu
        used_fixed_samples = []
        if seed > 0:
            random.seed(seed)
            used_fixed_samples = random.sample(self.fixed_samples, num_sample)

        # Xử lý các mẫu đã chọn để tạo câu trả lời cho bài toán
        for sample in used_fixed_samples:
            question = sample['question']
            answer = sample['answer']
            evi_only = sample.get('evidence_only', '')

            # Chỉnh sửa cách trả lời theo yêu cầu (ví dụ: cách 4)
            demo = f"Question: {question} \nAnswer: {answer} because {evi_only}"
            output += f'{demo}\n\n'

        return output

    def build_input(self, question, something_else="", with_restrict='inst', demo_way=-1, in_cot_way=False,
                    with_short_question_marker=False, with_short_answer_marker=False):
        """
        Tạo đầu vào cho mô hình
        """
        answer_restrict = " (just one entity)" if with_restrict in ['ans', 'both'] else ""
        question_marker = "Q: " if with_short_question_marker else "Question: "

        question = question.lower()

        if demo_way in [4]:
            prompt = f"{question_marker}{question} \n\n The answer{answer_restrict} is"
        else:
            raise NotImplementedError
        return prompt

    def post_process(self, pred, demo_way, in_cot_way, with_restrict, with_short_question_marker=False,
                     with_short_answer_marker=False, probs_output=None):
        """
        Hậu xử lý dự đoán từ mô hình
        :param pred: Dự đoán từ mô hình (có thể là một tuple)
        :param demo_way: Cách xử lý dự đoán (ví dụ: 4)
        :param in_cot_way: Xử lý chuỗi văn bản theo cách nào
        :param with_restrict: Kiểm tra điều kiện giới hạn
        :return: Các kết quả hậu xử lý
        """
        if demo_way == 4:
            splitter = 'because'
            if isinstance(pred, tuple):
                pred = pred[0]  # Lấy phần tử đầu tiên nếu pred là một tuple

            if splitter in pred:
                ans_end = pred.find(splitter)
                evi_start = ans_end + len(splitter)
                ans_p, gen_p = 0.0, 0.0
                ans_l, gen_l = 0, 0
                if probs_output:
                    for offset, token, prob in zip(probs_output[0], probs_output[1], probs_output[2]):
                        if token == '\n': continue
                        span = (offset, offset + len(token))
                        if span[0] >= evi_start:
                            gen_p += prob
                            gen_l += 1
                        if span[1] <= ans_end:
                            ans_p += prob
                            ans_l += 1
                    ans_p = 0.0 if ans_l == 0 else ans_p / ans_l
                    gen_p = 0.0 if gen_l == 0 else gen_p / gen_l
                    OP = (ans_p, gen_p, -1)

                A = pred[:ans_end].strip().replace('\n', ' ')
                B = pred[evi_start:].strip().replace('\n', ' ')
                C = 'none'
            else:
                OP = (-1, -1, -1)
                A = pred.strip().replace('\n', ' ')
                B, C = 'none', 'none'
        else:
            raise ValueError("Unsupported demo_way value")
        return A, B, C, OP

    def get_linetext(self, package, id, question):
        """
        Tạo chuỗi đầu ra cho một mẫu dữ liệu, bao gồm câu hỏi và dự đoán
        """
        assert len(package) == 4  # Đảm bảo rằng package chứa 4 phần tử
        probs_str = '|'.join([str(it) for it in package[3]])  # Chuyển đổi các xác suất thành chuỗi
        package_str = '\t'.join(package[:3]) + '\t' + probs_str  # Tạo chuỗi kết quả
        line = f"{id}\t{question}\t{package_str}\n"  # Định dạng kết quả
        return line



