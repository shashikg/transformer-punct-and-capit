import argparse
import string
from tqdm.auto import tqdm

alphabets = string.ascii_letters + string.digits
def filter_line(line):
    try:
        cond = [
            "..." in line,
            line.upper() == line,
            line.lower() == line,
            re.sub('[,?.]', '', line) == line,
            re.sub(f'[{alphabets}]', '', line) == line,
            len(line) == 0,
            line[-1] not in [".", "?", "|", "!"],
            not line[0].isupper(),
        ]
        if any(cond):
            return True
        else:
            return False
    except:
        return False

def parse_arguments():
    parser = argparse.ArgumentParser(description='Prepare Tatoeba')
    parser.add_argument('--tsv_fn', type=str, help='TSV Data')
    parser.add_argument('--op_txt_fn', default="none", type=str, help='Output file name')
    args = parser.parse_args()
    return args

args = parse_arguments()

tsv_fn = args.tsv_fn
sent = []
with open(tsv_fn, 'r') as f:
    data = f.read().split("\n")
    for line in tqdm(data, desc="reading data"):
        line = line.split("\t")[-1].strip()
        if not filter_line(line):
            sent.append(line)
            
orig_len = len(sent)
sent = list(set(sent))
final_len = len(sent)
print(f"Original data length: {orig_len:,} - After removing duplicates: {final_len:,}")
        
with open(args.op_txt_fn, 'w') as f:
    f.write("\n".join(sent))