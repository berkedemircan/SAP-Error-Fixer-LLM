import json
import re


REMOVE_CHARS = '<>|"?\\\n'
REMOVE_REGEX = re.compile(f'[{re.escape(REMOVE_CHARS)}]')

def clean_text(text):
    if not isinstance(text, str):
        return text
    return REMOVE_REGEX.sub('', text)

def main():
    with open('SAP_Logs_1500.json', encoding='utf-8') as f:
        logs = json.load(f)
    for entry in logs:
        if 'error' in entry:
            entry['error'] = clean_text(entry['error'])
        if 'warning' in entry:
            entry['warning'] = clean_text(entry['warning'])
    with open('SAP_Logs_1500_cleaned.json', 'w', encoding='utf-8') as f:
        json.dump(logs, f, ensure_ascii=False, indent=4)
    print(f"Temizlenmiş dosya oluşturuldu: 'SAP_Logs_1500_cleaned.json'")

if __name__ == "__main__":
    main()