import re, os
import json
import string

def remove_punctuation(word):
    all_punct_marks = string.punctuation.replace("'", '')
    return re.sub('[' + all_punct_marks + ']', '', word)

def create_manifest(ip_fn, save_dir, punct_labels='O|.|,|?', labels_order='c|p', ):
    labels_order = labels_order.split("|")
    punct_labels = punct_labels.split("|")
    punct_labels.remove('O')
    
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.basename(ip_fn).replace(".txt", ".json")
    man_fn = os.path.join(save_dir, base_name)
    
    with open(ip_fn, 'r') as txt_fio:
        with open(man_fn, 'w') as man_fio:
            for line in txt_fio:
                orig_line = line.strip()
                line = line.strip().split()
                text = ''
                labels = ''
                
                for word in line:
                    label = {}
                    label['p'] = word[-1] if word[-1] in punct_labels else 'O'
                    word = remove_punctuation(word)
                    
                    if len(word) > 0:
                        if word.isupper():
                            label['c'] = 'U'
                        elif word[0].isupper():
                            label['c'] = 'C'
                        else:
                            label['c'] = 'O'

                        word = word.lower()
                        text += word + ' '
                        labels += label[labels_order[0]] + '|' + label[labels_order[1]] + ' '

                if (len(text.strip()) > 0) and (len(labels.strip()) > 0):
                    metadata = {
                        "x_text": text.strip(),
                        "y_text": orig_line,
                        "labels": labels.strip(),
                    }

                    man_fio.write(json.dumps(metadata) + '\n')

    print(f'[create_manifest]: [{man_fn}] created from [{ip_fn}].')
    
    return man_fn