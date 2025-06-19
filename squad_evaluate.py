import os
import shutil
import subprocess


def collect_merge_delete(dir, num=10, ifdelete=False):
    """
    Merge all prediction files in 'subs' directory into a single file 'full_preds.txt'
    and call the evaluation function.
    """
    subs_dir = os.path.join(dir, 'subs')
    alls = [''] * num

    # Merge all the prediction files from 'subs' directory
    for fn in os.listdir(subs_dir):
        ffn = os.path.join(subs_dir, fn)
        try:
            with open(ffn) as f:
                lines = f.readlines()
            for l in lines:
                ll = l.strip()
                id = int(ll.split('\t')[0])
                if alls[id] != '':
                    raise ValueError(f"Duplicate entry found for id {id} in file {fn}")
                alls[id] = ll
        except Exception as e:
            print(f"Error reading file {ffn}: {e}")
            continue

    # Ensure no empty slots in the merged predictions
    for idx, span in enumerate(alls):
        if span == '':
            raise ValueError(f"Missing prediction for index {idx}")

    # Write merged predictions to 'full_preds.txt'
    fin_dir = os.path.join(dir, 'full_preds.txt')
    try:
        with open(fin_dir, 'w') as f:
            f.write('\n'.join(alls))
    except Exception as e:
        print(f"Error writing to {fin_dir}: {e}")
        raise

    # Optionally delete the 'subs' directory
    if ifdelete:
        try:
            shutil.rmtree(subs_dir)
            print(f"Deleted 'subs' directory: {subs_dir}")
        except Exception as e:
            print(f"Error deleting 'subs' directory: {e}")
    
    # Call the evaluation function
    eval_results = evaluate_predictions(dir)
    return eval_results


def evaluate_predictions(dir):
    """
    Evaluate the predictions using F1-Score and Exact Match (EM).
    Assumes the presence of a reference file (e.g., 'reference.json') and predictions in 'full_preds.txt'.
    """
    # Define paths for the predictions and reference files
    pred_file = os.path.join(dir, 'full_preds.txt')
    reference_file = os.path.join(dir, 'reference.json')  # Update this path as needed

    # Check if the necessary files exist
    if not os.path.exists(pred_file):
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    print(f"Evaluating predictions: {pred_file}")

    # Run the evaluation script (assuming you have `squad_evaluate.py` for this)
    try:
        result = subprocess.run(
            ["python", "squad_evaluate.py", "--taskname", "your_task_name", "--pred_filename", pred_file, "--reference_filename", reference_file],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
    except Exception as e:
        print(f"Error during subprocess execution: {e}")
        return None
    
    # Check for any errors in stderr
    error_output = result.stderr.decode('utf-8')
    if error_output:
        print(f"Error: {error_output}")
        return None

    # Parse the output from the evaluation script
    output = result.stdout.decode('utf-8')
    print(f"Output from squad_evaluate:\n{output}")
    
    # Assuming output contains F1 and EM scores, extract them (modify this based on actual output)
    f1_score = None
    em_score = None
    
    for line in output.splitlines():
        if "F1" in line:
            f1_score = float(line.split(":")[1].strip())
        elif "EM" in line:
            em_score = float(line.split(":")[1].strip())
    
    # Check if we have extracted the scores
    if f1_score is not None and em_score is not None:
        print(f"F1 Score: {f1_score}")
        print(f"Exact Match (EM): {em_score}")
    else:
        print("Failed to extract F1 and EM scores from the evaluation output.")
    
    return f1_score, em_score


if __name__ == '__main__':
    # Iterate through the output directories to process and evaluate
    for subdir in os.listdir('./outputs'):
        fsubdir = os.path.join('./outputs', subdir)
        for subsubdir in os.listdir(fsubdir):
            fsubsubdir = os.path.join(fsubdir, subsubdir)
            if 'subs' in os.listdir(fsubsubdir) and 'full_preds.txt' not in os.listdir(fsubsubdir):
                # Attempt to merge predictions and evaluate
                try:
                    eval_results = collect_merge_delete(dir=fsubsubdir, ifdelete=True, num=10)
                    print('Merge success!')
                    print(f'Evaluation Results: {eval_results}')
                    print('-----')
                except Exception as e:
                    print(f"Error: {e}")
