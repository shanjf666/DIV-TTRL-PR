import json
import argparse
from collections import Counter

def strip_string(string):
    if string is None:
        return ""
    string = str(string)
    string = string.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "").replace(" ", "")
    if string == "0.5":
        string = "\\frac{1}{2}"
    return string

def main():
    parser = argparse.ArgumentParser(description="Compare Student (Original) vs Teacher (Refined) pseudo-labels.")
    parser.add_argument("--input_file", type=str, required=True, help="Input JSONL from teacher_voting_experiment.py")
    args = parser.parse_args()

    total = 0
    reprompted = 0
    diff_count = 0
    both_correct = 0
    rescued = 0 # Student Wrong -> Teacher Correct
    harmed = 0  # Student Correct -> Teacher Wrong
    still_wrong = 0

    examples = []

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            total += 1
            
            if not item.get("was_reprompted", False):
                continue
            
            reprompted += 1
            
            # Use normalized versions for comparison
            student_ans = strip_string(item.get("maj_answer", item.get("sc_answer", "")))
            teacher_ans = strip_string(item.get("teacher_maj_answer", ""))
            gt_ans = strip_string(item.get("gt", item.get("answer", "")))

            if student_ans != teacher_ans:
                diff_count += 1
                if len(examples) < 5:
                    examples.append({
                        "id": item.get("idx", "N/A"),
                        "problem": item.get("problem", "")[:200] + "...",
                        "student": student_ans,
                        "teacher": teacher_ans,
                        "gt": gt_ans
                    })

            # Accuracy analysis
            student_correct = (student_ans == gt_ans) if gt_ans else False
            teacher_correct = (teacher_ans == gt_ans) if gt_ans else False

            if student_correct and teacher_correct: both_correct += 1
            elif not student_correct and teacher_correct: rescued += 1
            elif student_correct and not teacher_correct: harmed += 1
            else: still_wrong += 1

    print("\n" + "="*60)
    print("STUDENT VS TEACHER COMPARISON")
    print("="*60)
    print(f"Total problems           : {total}")
    print(f"Reprompted (Low SC < 0.3): {reprompted}")
    print(f"Label Changes            : {diff_count} ({diff_count/reprompted:.1%})")
    
    if gt_ans: # If GT is available in the file
        print("-" * 30)
        print(f"Rescued (✗ -> ✓)         : {rescued}")
        print(f"Harmed  (✓ -> ✗)         : {harmed}")
        print(f"Net Gain                 : {rescued - harmed:+d}")
        print(f"Teacher Final Accuracy   : {(both_correct + rescued)/reprompted:.1%}")
    
    if examples:
        print("\n" + "-"*30)
        print("EXAMPLES OF CHANGES")
        print("-" * 30)
        for ex in examples:
            print(f"ID {ex['id']}: {ex['problem']}")
            print(f"  Student: {ex['student']}")
            print(f"  Teacher: {ex['teacher']}")
            if ex['gt']: print(f"  Ground Truth: {ex['gt']}")
            print("-" * 10)
    print("="*60)

if __name__ == "__main__":
    main()
