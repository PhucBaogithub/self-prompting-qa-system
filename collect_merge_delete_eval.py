import os
import shutil
import subprocess

def collect_merge_delete(dir, num=10, ifdelete=False):
    subs_dir = os.path.join(dir, 'subs')
    alls = [''] * num
    
    # Kiểm tra sự tồn tại của thư mục 'subs'
    if not os.path.exists(subs_dir):
        print(f"Warning: The 'subs' directory does not exist in {dir}")
        return  # Thoát ra nếu không có thư mục 'subs'

    # Hợp nhất các file con trong thư mục 'subs'
    for fn in os.listdir(subs_dir):
        ffn = os.path.join(subs_dir, fn)
        try:
            with open(ffn) as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {ffn}: {e}")
            continue

        for l in lines:
            ll = l.strip()
            id = int(ll.split('\t')[0])
            if alls[id] != '':
                raise ValueError(f"Duplicate entry found for id {id} in file {fn}")
            alls[id] = ll

    # Kiểm tra xem có id nào bị thiếu không
    for idx, span in enumerate(alls):
        if span == '':
            raise ValueError(f"Missing value for id {idx}")

    # Lưu kết quả vào file 'full_preds.txt'
    fin_dir = os.path.join(dir, 'full_preds.txt')
    try:
        with open(fin_dir, 'w') as f:
            f.write('\n'.join(alls))
        print(f"File full_preds.txt created at {fin_dir}")
    except Exception as e:
        print(f"Error writing to {fin_dir}: {e}")
        return

    # Nếu yêu cầu, xóa thư mục 'subs'
    if ifdelete:
        shutil.rmtree(subs_dir)
        print(f"Deleted the 'subs' directory at {subs_dir}")

    # Gọi hàm đánh giá mô hình sau khi merge thành công
    eval_results = evaluate_predictions(dir)
    return eval_results


def evaluate_predictions(dir):
    """
    Function to evaluate the predictions using F1-Score and Exact Match (EM).
    Assumes the presence of a reference file (e.g., 'reference.json') and predictions in 'full_preds.txt'.
    """
    # Define paths for the ground truth and predictions
    pred_file = os.path.join(dir, 'full_preds.txt')
    reference_file = os.path.join(dir, 'reference.json')  # Update this path as needed

    # Kiểm tra sự tồn tại của file dự đoán và tham chiếu
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file {pred_file} not found!")
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file {reference_file} not found!")

    # Gọi script đánh giá và lấy kết quả
    result = subprocess.run(
        ["python", "squad_evaluate.py", "--taskname", "your_task_name", "--pred_filename", pred_file, "--reference_filename", reference_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    
    # Kiểm tra lỗi trong quá trình gọi subprocess
    if result.returncode != 0:
        print(f"Error in subprocess: {result.stderr.decode('utf-8')}")
        return

    # Parse kết quả đầu ra từ script đánh giá
    output = result.stdout.decode('utf-8')
    f1_score = None
    em_score = None
    
    for line in output.splitlines():
        if "F1" in line:
            f1_score = float(line.split(":")[1].strip())
        elif "EM" in line:
            em_score = float(line.split(":")[1].strip())
    
    print(f"F1 Score: {f1_score}")
    print(f"Exact Match (EM): {em_score}")
    return f1_score, em_score


if __name__ == '__main__':
    for subdir in os.listdir('./outputs'):
        fsubdir = os.path.join('./outputs', subdir)
        for subsubdir in os.listdir(fsubdir):
            fsubsubdir = os.path.join(fsubdir, subsubdir)
            if 'subs' in os.listdir(fsubsubdir) and 'full_preds.txt' not in os.listdir(fsubsubdir):
                # Thực hiện merge và đánh giá
                try:
                    eval_results = collect_merge_delete(dir=fsubsubdir, ifdelete=True, num=10)
                    print('Merge success!')
                    print(f'Evaluation Results: {eval_results}')
                    print('-----')
                except Exception as e:
                    print(f"Error during merging and evaluation in {subsubdir}: {e}")
